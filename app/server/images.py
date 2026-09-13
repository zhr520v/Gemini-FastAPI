from typing import Literal

from curl_cffi.requests import AsyncSession
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.models.models import ImageGenerationRequest, ImageGenerationResponse
from app.server.middleware import get_image_store_dir, verify_api_key, verify_image_token
from app.services.image_pipeline import generate_images_with_failover
from app.services.pool import GeminiClientPool

router = APIRouter()


@router.post(
    "/v1/images/generations",
    response_model=ImageGenerationResponse,
    tags=["Images"],
    summary="Stateless Image Generation (OpenAI-compatible)",
    description="""
High-performance, stateless image generation pipeline for Gemini:
- **Zero Local Disk I/O**: Direct high-res Google CDN URLs or pure in-memory Base64 strings.
- **Early Return**: Intercepts generated images immediately from the stream, skipping trailing text generation.
- **Bypasses LMDB Storage**: Fully stateless, runs in Google Temporary Chat mode (not saved to account history).
- **Multi-Account Failover Retry**: Automatically retries across available accounts in the pool if an account fails or refuses to draw.
""",
)
async def generate_images(
    request: ImageGenerationRequest,
    _auth=Depends(verify_api_key),
):
    pool = GeminiClientPool()
    format_type: Literal["url", "b64_json"] = request.response_format or "url"

    items = await generate_images_with_failover(
        pool=pool,
        prompt=request.prompt,
        n=request.n or 1,
        model=request.model,
        response_format=format_type,
    )
    return ImageGenerationResponse(data=items)


@router.get(
    "/v1/images/proxy",
    tags=["Images"],
    summary="Memory-only Stream Proxy for CDN Images",
    description="Directly stream image bytes from Google CDN to client without touching local disk.",
)
async def proxy_image_stream(
    url: str = Query(..., description="Google CDN Image URL"),
    _auth=Depends(verify_api_key),
):
    """
    Optional stream proxy for clients that cannot reach Google CDN directly.
    Streams directly in memory without writing to disk.
    """
    if "googleusercontent.com" not in url:
        raise HTTPException(status_code=400, detail="Only Google CDN URLs are allowed")

    async with AsyncSession(headers={"Referer": "https://gemini.google.com/"}) as session:
        resp = await session.get(url)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="Failed to fetch image from CDN")

        content_type = resp.headers.get("content-type", "image/png")
        return Response(content=resp.content, media_type=content_type)


@router.get("/images/{filename}", tags=["Images"])
async def get_image(filename: str, token: str | None = Query(default=None)):
    """Legacy file endpoint for stored images (if any)."""
    if not verify_image_token(filename, token):
        raise HTTPException(status_code=403, detail="Invalid token")

    image_store = get_image_store_dir()
    file_path = image_store / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(file_path)
