from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags


@dataclass
class ImageData:
    path: Optional[str]
    fmt: str
    width: int
    height: int
    mode: str
    exif: dict
    color: np.ndarray
    gray: np.ndarray

    @property
    def is_jpeg(self) -> bool:
        return self.fmt.upper() in {"JPEG", "JPG"}


def _exif_to_dict(raw_exif) -> dict:
    out: dict = {}
    if not raw_exif:
        return out
    for tag, value in raw_exif.items():
        name = ExifTags.TAGS.get(tag, str(tag))
        out[name] = value
    return out


def load_image(source) -> ImageData:
    if isinstance(source, (bytes, bytearray)):
        from io import BytesIO

        pil = Image.open(BytesIO(bytes(source)))
        path = None
    elif isinstance(source, str):
        pil = Image.open(source)
        path = source
    else:
        pil = Image.open(source)
        path = getattr(source, "name", None)

    pil.load()
    fmt = pil.format or "UNKNOWN"
    mode = pil.mode

    exif_raw = pil.getexif() if hasattr(pil, "getexif") else None
    exif = _exif_to_dict(exif_raw)

    rgb = pil.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0

    gray = cv2.cvtColor((arr * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0

    h, w = arr.shape[:2]
    return ImageData(
        path=path,
        fmt=fmt,
        width=w,
        height=h,
        mode=mode,
        exif=exif,
        color=arr,
        gray=gray,
    )