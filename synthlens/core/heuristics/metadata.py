from __future__ import annotations

import re

from ..types import HeuristicResult
from ..image_utils import ImageData

WEIGHT = 0.25

CAMERA_TAGS = {"Make", "Model", "LensModel", "LensSpecification", "GPSInfo",
               "DateTimeOriginal", "FocalLength", "FNumber", "ExposureTime",
               "ISOSpeedRatings", "Orientation", "Software"}

GENERATOR_PATTERNS = [
    r"midjourney", r"dall-?e", r"stable.?diffusion", r"sdxl", r"sd ?xl",
    r"automatic1111", r"comfyui", r"novelai", r"leonardo", r"firefly",
    r"imagen", r"gpt-?image", r"openai", r"bing.?image", r"creator",
    r"photoshop", r"lightroom", r"affinity", r"gimp",
]

GENERATOR_NAMES = {"midjourney", "dalle", "stable diffusion", "sdxl", "comfyui",
                   "novelai", "leonardo", "firefly", "imagen", "openai"}


def _match_generator(text: str) -> str | None:
    if not text:
        return None
    low = text.lower()
    for pat in GENERATOR_PATTERNS:
        m = re.search(pat, low)
        if m:
            return m.group(0)
    return None


def _chunk_text(info: dict) -> str:
    parts = []
    for v in info.values():
        if isinstance(v, bytes):
            try:
                parts.append(v.decode("utf-8", "replace"))
            except Exception:
                pass
        elif isinstance(v, str):
            parts.append(v)
    return " ".join(parts)


def analyze_metadata(img: ImageData) -> HeuristicResult:
    exif = img.exif
    readable = img.readable_exif
    info = img.info
    details = {
        "format": img.fmt,
        "dimensions": f"{img.width}x{img.height}",
        "file_size": img.file_size,
        "exif_tag_count": len(readable),
        "exif": readable,
        "chunks": {k: _short(v) for k, v in info.items()},
    }

    present = CAMERA_TAGS & set(exif.keys())
    details["camera_tags_present"] = sorted(present)

    software = str(exif.get("Software", "")) if exif else ""
    make = str(exif.get("Make", "")) if exif else ""
    chunk_text = _chunk_text(info)
    gen_match = (_match_generator(software) or _match_generator(make)
                 or _match_generator(chunk_text))
    if gen_match:
        details["generator_detected"] = gen_match
    if software:
        details["software"] = software

    has_camera = bool({"Make", "Model", "LensModel"} & set(exif.keys()))
    is_editing_software = bool(gen_match and gen_match in {
        "photoshop", "lightroom", "affinity", "gimp"})

    score = 0.5
    if gen_match and gen_match in GENERATOR_NAMES:
        score = 0.95
    elif gen_match and is_editing_software:
        score = 0.3
    elif has_camera:
        score = 0.05
    elif len(readable) == 0 and img.is_jpeg:
        score = 0.7
    elif len(readable) == 0:
        score = 0.6
    else:
        score = 0.45

    common_ai_sizes = {512, 768, 1024, 1280, 1536, 2048}
    if img.width in common_ai_sizes and img.height in common_ai_sizes:
        score = min(1.0, score + 0.1)
        details["square_ai_size"] = True

    return HeuristicResult(
        name="metadata",
        score=score,
        weight=WEIGHT,
        details=details,
    )


def _short(v) -> str:
    if isinstance(v, bytes):
        try:
            s = v.decode("utf-8", "replace").strip()
        except Exception:
            s = v.hex()
        return s[:200] + ("..." if len(s) > 200 else "")
    s = str(v)
    return s[:200] + ("..." if len(s) > 200 else "")