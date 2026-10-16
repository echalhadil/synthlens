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


def analyze_metadata(img: ImageData) -> HeuristicResult:
    exif = img.exif
    details = {"format": img.fmt, "dimensions": f"{img.width}x{img.height}",
               "exif_tag_count": len(exif)}

    present = CAMERA_TAGS & set(exif.keys())
    details["camera_tags_present"] = sorted(present)

    software = str(exif.get("Software", "")) if exif else ""
    gen_match = _match_generator(software) or _match_generator(str(exif.get("Make", "")))

    has_camera = bool({"Make", "Model", "LensModel"} & set(exif.keys()))
    is_editing_software = bool(gen_match and gen_match in {
        "photoshop", "lightroom", "affinity", "gimp"})

    score = 0.5
    if gen_match and gen_match in GENERATOR_NAMES:
        score = 0.95
        details["generator_detected"] = gen_match
    elif gen_match and is_editing_software:
        score = 0.3
        details["software"] = software
    elif has_camera:
        score = 0.05
    elif len(exif) == 0 and img.is_jpeg:
        score = 0.7
    elif len(exif) == 0:
        score = 0.6
    else:
        score = 0.45

    common_ai_sizes = {512, 768, 1024, 1024, 1280, 1536, 2048}
    if img.width in common_ai_sizes and img.height in common_ai_sizes:
        score = min(1.0, score + 0.1)
        details["square_ai_size"] = True

    return HeuristicResult(
        name="metadata",
        score=score,
        weight=WEIGHT,
        details=details,
    )