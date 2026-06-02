from __future__ import annotations

import base64
from pathlib import Path


DEFAULT_MAX_IMAGE_BYTES = 20 * 1024 * 1024


def _detect_mime(data: bytes, path: Path) -> str | None:
    suffix = path.suffix.lower()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if suffix == ".png":
        return None
    if suffix in {".jpg", ".jpeg"}:
        return None
    if suffix == ".webp":
        return None
    return None


def image_path_to_content_block(path: str | Path, *, max_bytes: int = DEFAULT_MAX_IMAGE_BYTES) -> dict:
    image_path = Path(path)
    try:
        data = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read image: {path}") from exc
    if len(data) > max_bytes:
        raise ValueError(f"image exceeds {max_bytes} bytes: {path}")
    mime = _detect_mime(data, image_path)
    if not mime:
        raise ValueError(f"unsupported image type: {path}")
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{encoded}"},
    }
