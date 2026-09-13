import asyncio
import base64
import io
from pathlib import Path
from typing import Any, Literal

from curl_cffi.requests import AsyncSession
from fastapi import HTTPException
from gemini_webapi.types.image import GeneratedImage, Image
from loguru import logger

from app.models.models import ImageItem
from app.services.client import GeminiClientWrapper
from app.services.pool import GeminiClientPool
from app.utils.helper import save_url_to_tempfile


class NoImageGeneratedError(Exception):
    """Raised when Gemini returns a response stream without any generated images."""
    pass


def build_image_prompt(prompt: str, has_reference_image: bool = False) -> str:
    """
    Wrap user prompt with strict instructions to guarantee the image generation tool is invoked.
    If a reference image is attached, guide the model into image-to-image modification mode.
    """
    cleaned = prompt.strip()
    if has_reference_image:
        return (
            f"Based on the attached reference image, generate a modified image following this description: {cleaned}. "
            "STRICT INSTRUCTION: You MUST invoke your image generation tool and return the newly generated image. "
            "Do NOT output conversational text, explanations, or questions."
        )
    return (
        f"Generate an image based on this description: {cleaned}. "
        "STRICT INSTRUCTION: You MUST invoke your image generation tool and return the generated image. "
        "Do NOT output conversational text, explanations, or questions."
    )


def enhance_cdn_url(url: str) -> str:
    """Upgrade Google thumbnail/preview URLs to high-resolution (2048px)."""
    if not url:
        return url
    if "=s1024-rj" in url:
        return url.replace("=s1024-rj", "=s2048-rj")
    if "=s2048-rj" not in url and "googleusercontent.com" in url:
        return f"{url}=s2048-rj"
    return url


async def prepare_reference_files(
    reference_image: str | None,
    tempdir: Path | None = None,
) -> list[Any]:
    """Convert input image URL or Base64 into Gemini uploadable file objects."""
    if not reference_image:
        return []
    ref = reference_image.strip()
    if ref.startswith("http://") or ref.startswith("https://"):
        file_path = await save_url_to_tempfile(ref, tempdir)
        return [file_path]

    # Base64 string
    b64_data = ref.split(",", 1)[1] if "," in ref else ref
    try:
        raw_bytes = base64.b64decode(b64_data)
        file_obj = io.BytesIO(raw_bytes)
        file_obj.name = "reference_image.png"
        return [file_obj]
    except Exception as e:
        logger.warning(f"Failed to decode reference image base64: {e}")
        return []


async def format_image_output(
    image: Image,
    client: GeminiClientWrapper,
    response_format: Literal["url", "b64_json"] = "url",
) -> ImageItem:
    """Format image output with zero local disk I/O."""
    high_res_url = enhance_cdn_url(image.url)

    if response_format == "url":
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
    files: list[Any] | None = None,
    target_count: int = 1,
    timeout: float = 60.0,
) -> list[Image]:
    """Stream from Gemini with early-return cut-off upon image detection."""
    found_images: list[Image] = []
    stream = session.send_message_stream(full_prompt, files=files, temporary=True)

    try:
        async with asyncio.timeout(timeout):
            async for chunk in stream:
                chunk_images = getattr(chunk, "images", None)
                if chunk_images:
                    for img in chunk_images:
                        if isinstance(img, (GeneratedImage, Image)):
                            found_images.append(img)

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
    reference_image: str | None = None,
    n: int = 1,
    model: str | None = None,
    response_format: Literal["url", "b64_json"] = "url",
    timeout: float = 60.0,
    max_retries: int | None = None,
) -> list[ImageItem]:
    """High-reliability image generation pipeline supporting text-to-image and image-to-image."""
    total_clients = len(pool.clients)
    if total_clients == 0:
        raise HTTPException(status_code=500, detail="No Gemini clients configured in pool")

    retries = max_retries if max_retries is not None else max(1, min(total_clients, 3))
    has_ref = bool(reference_image)
    wrapped_prompt = build_image_prompt(prompt, has_reference_image=has_ref)
    ref_files = await prepare_reference_files(reference_image)
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            client = await pool.acquire()
        except Exception as e:
            logger.error(f"[ImageGen] Failed to acquire client from pool: {e}")
            raise HTTPException(status_code=503, detail="No available Gemini client in pool") from e

        logger.info(
            f"[ImageGen] Attempt {attempt}/{retries} on client [{client.id}], format={response_format}, img2img={has_ref}"
        )

        try:
            target_model = model or "gemini-pro"
            session = client.start_chat(model=target_model)

            images = await stream_and_early_return(
                session=session,
                full_prompt=wrapped_prompt,
                files=ref_files,
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
