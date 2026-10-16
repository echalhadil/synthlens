from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags, TiffTags

try:
    from PIL.ExifTags import IFD as EXIF_IFD
except Exception:
    EXIF_IFD = {}

SUB_IFDS = {
    "ExifIFD": 0x8769,
    "GPSIFD": 0x8825,
    "InteropIFD": 0xA005,
}


@dataclass
class ImageData:
    path: Optional[str]
    fmt: str
    width: int
    height: int
    mode: str
    file_size: int
    exif: dict
    readable_exif: dict
    info: dict
    color: np.ndarray
    gray: np.ndarray

    @property
    def is_jpeg(self) -> bool:
        return self.fmt.upper() in {"JPEG", "JPG"}

    @property
    def header(self) -> dict:
        return {
            "format": self.fmt,
            "dimensions": f"{self.width}x{self.height}",
            "width": self.width,
            "height": self.height,
            "mode": self.mode,
            "file_size": self.file_size,
            "exif": self.readable_exif,
            "chunks": {k: _readable_value(v) for k, v in self.info.items()},
        }


def _readable_value(v) -> str:
    if isinstance(v, bytes):
        try:
            return v.decode("utf-8", "replace").strip()
        except Exception:
            return v.hex()
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        d = v.denominator
        return f"{v.numerator}/{d}" if d not in (0, 1) else f"{int(v)}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_readable_value(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}={_readable_value(x)}" for k, x in v.items())
    return str(v)


def _tag_name(tag: int) -> str:
    return TiffTags.lookup(tag).name or ExifTags.TAGS.get(tag, str(tag))


def _readable_exif(exif_obj) -> dict:
    out: dict = {}
    if not exif_obj:
        return out
    for tag, value in exif_obj.items():
        name = _tag_name(tag)
        out[name] = _readable_value(value)
    for label, ifd_id in SUB_IFDS.items():
        try:
            sub = exif_obj.get_ifd(ifd_id) if hasattr(exif_obj, "get_ifd") else None
        except Exception:
            sub = None
        if not sub:
            continue
        for tag, value in sub.items():
            name = _tag_name(tag)
            out[f"{label}.{name}"] = _readable_value(value)
    return out


def load_image(source) -> ImageData:
    if isinstance(source, (bytes, bytearray)):
        from io import BytesIO

        data = bytes(source)
        pil = Image.open(BytesIO(data))
        path = None
        file_size = len(data)
    elif isinstance(source, str):
        pil = Image.open(source)
        path = source
        try:
            file_size = os.path.getsize(source)
        except OSError:
            file_size = 0
    else:
        data = source.read()
        pil = Image.open(__import__("io").BytesIO(data))
        path = getattr(source, "name", None)
        file_size = len(data)

    pil.load()
    fmt = pil.format or "UNKNOWN"
    mode = pil.mode

    exif_obj = pil.getexif() if hasattr(pil, "getexif") else None
    readable = _readable_exif(exif_obj)
    raw_names = {}
    if exif_obj:
        for tag, value in exif_obj.items():
            raw_names[_tag_name(tag)] = value

    info = {}
    for k, v in (pil.info or {}).items():
        if k in ("exif", "icc_profile", "Photoshop"):
            continue
        info[str(k)] = v

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
        file_size=file_size,
        exif=raw_names,
        readable_exif=readable,
        info=info,
        color=arr,
        gray=gray,
    )