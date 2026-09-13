import asyncio
import base64
from typing import Literal

from curl_cffi.requests import AsyncSession
from fastapi import HTTPException
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from app.models.models import ImageItem
from app.services.client import GeminiClientWrapper
from app.services.pool import GeminiClientPool


class NoImageGeneratedError(Exception):
    """Raised when Gemini returns a response stream without any generated images."""
    pass


def build_image_prompt(prompt: str) -> str:
    """
    Wrap user prompt with strict instructions to guarantee the image generation tool is invoked,
    suppressing conversational chatter.
    """
    cleaned = prompt.strip()
    return (
        f"Generate an image based on this description: {cleaned}. "
        "STRICT INSTRUCTION: You MUST invoke your image generation tool and return the generated image. "
        "Do NOT output conversational text, explanations, or questions."
    )


def enhance_cdn_url(url: str) -> str:
    """
    Upgrade Google thumbnail / preview URLs to high-resolution (2048px).
    """
    if not url:
        return url
    if "=s1024-rj" in url:
        return url.replace("=s1024-rj", "=s2048-rj")
    if "=s2048-rj" not in url and "googleusercontent.com" in url:
        return f"{url}=s2048-rj"
    return url


async def format_image_output(
    image: Image,
    client: GeminiClientWrapper,
    response_format: Literal["url", "b64_json"] = "url",
) -> ImageItem:
    """
    Format image result with zero local disk I/O:
    - 'url': Direct Google CDN URL with 2K resolution enhancement.
    - 'b64_json': In-memory stream download and base64 encoding without saving to disk.
    """
    high_res_url = enhance_cdn_url(image.url)

    if response_format == "url":
        # Direct CDN: 0 disk IO, 0 server bandwidth egress
        return ImageItem(url=high_res_url)

    # b64_json: Pure in-memory streaming
    impersonate = getattr(client, "impersonate", "chrome131")
    cookies = getattr(client, "cookies", None)
    effective_proxy = getattr(client, "proxy", None)

    async with AsyncSession(
        impersonate=impersonate,
        cookies=cookies,
        proxy=effective_proxy,
        headers={"Referer": "https://gemini.google.com/"},
    ) as req_session:
        # First attempt high-res URL
        resp = await req_session.get(high_res_url)
        if resp.status_code != 200 and high_res_url != image.url:
            resp = await req_session.get(image.url)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch image bytes from CDN: HTTP {resp.status_code} {resp.reason}"
            )

        b64_data = base64.b64encode(resp.content).decode("utf-8")
        return ImageItem(b64_json=b64_data)


async def stream_and_early_return(
    session,
    full_prompt: str,
    target_count: int = 1,
    timeout: float = 60.0,
) -> list[Image]:
    """
    Stream from Gemini with early-return cut-off:
    As soon as target image count is acquired, abort the rest of the text stream immediately.

    Edge Cases Handled:
    1. Early cut-off: Discontinue reading subsequent chunks as soon as images appear.
    2. Generator cleanup: Explicitly aclose() the async generator to avoid lingering HTTP/2 streams.
    3. Delayed images: Continues consuming chunks until images arrive or stream ends.
    4. Refusals / Safety block: Detects empty image result and raises NoImageGeneratedError for failover.
    5. Timeout protection: Uses asyncio.timeout to prevent indefinite hangs.
    """
    found_images: list[Image] = []

    # temporary=True guarantees:
    # 1. Flag 45 is sent to Google, so conversation is NOT stored in Google Cloud account history.
    # 2. Truly stateless execution.
    stream = session.send_message_stream(full_prompt, temporary=True)

    try:
        async with asyncio.timeout(timeout):
            async for chunk in stream:
                chunk_images = getattr(chunk, "images", None)
                if chunk_images:
                    for img in chunk_images:
                        if isinstance(img, (GeneratedImage, Image)):
                            found_images.append(img)

                    # Early return: we got what we need, stop waiting for lengthy text descriptions!
                    if len(found_images) >= target_count:
                        logger.info(
                            f"[Early Return] Intercepted {len(found_images)} images. Cutting off stream early."
                        )
                        break
    except TimeoutError:
        logger.warning(f"[Early Return] Stream timed out after {timeout}s.")
        if not found_images:
            raise
    finally:
        # Gracefully close the generator to free resources and avoid connection leaks
        if hasattr(stream, "aclose"):
            try:
                await stream.aclose()
            except Exception as e:
                logger.debug(f"[Early Return] Generator aclose note: {e}")

    if not found_images:
        raise NoImageGeneratedError("Gemini stream completed without generating any images.")

    return found_images


async def generate_images_with_failover(
    pool: GeminiClientPool,
    prompt: str,
    n: int = 1,
    model: str | None = None,
    response_format: Literal["url", "b64_json"] = "url",
    timeout: float = 60.0,
    max_retries: int | None = None,
) -> list[ImageItem]:
    """
    High-reliability image generation pipeline:
    - Bypasses LMDB conversation persistence entirely.
    - Uses Google temporary chat mode (stateless, not saved to cloud).
    - Early return upon image arrival (no waiting for extra text).
    - Zero local disk IO (CDN direct URL or in-memory Base64).
    - Multi-account pool failover retry.
    """
    total_clients = len(pool.clients)
    if total_clients == 0:
        raise HTTPException(status_code=500, detail="No Gemini clients configured in pool")

    # Retry across accounts up to total_clients (capped at 3 for reasonable latency)
    retries = max_retries if max_retries is not None else max(1, min(total_clients, 3))
    wrapped_prompt = build_image_prompt(prompt)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            client = await pool.acquire()
        except Exception as e:
            logger.error(f"[ImageGen] Failed to acquire client from pool: {e}")
            raise HTTPException(status_code=503, detail="No available Gemini client in pool") from e

        logger.info(
            f"[ImageGen] Attempt {attempt}/{retries} on client [{client.id}], format={response_format}"
        )

        try:
            # Bypass LMDB: Start a fresh stateless chat session
            session = client.start_chat(model=model)

            # Stream with early return
            images = await stream_and_early_return(
                session=session,
                full_prompt=wrapped_prompt,
                target_count=n,
                timeout=timeout,
            )

            results: list[ImageItem] = []
            for img in images[:n]:
                item = await format_image_output(
                    image=img,
                    client=client,
                    response_format=response_format,
                )
                results.append(item)

            logger.info(
                f"[ImageGen] Successfully generated {len(results)} image(s) via client [{client.id}]."
            )
            return results

        except NoImageGeneratedError as e:
            logger.warning(
                f"[ImageGen] Client [{client.id}] did not generate image ({e}). Triggering failover to next client..."
            )
            last_error = e
        except Exception as e:
            logger.warning(
                f"[ImageGen] Client [{client.id}] failed with {type(e).__name__}: {e}. Triggering failover to next client..."
            )
            last_error = e

    logger.error(
        f"[ImageGen] All {retries} failover attempts failed. Last error: {last_error}"
    )
    raise HTTPException(
        status_code=502,
        detail=f"Image generation failed after {retries} client attempts: {last_error}",
    ) from last_error
