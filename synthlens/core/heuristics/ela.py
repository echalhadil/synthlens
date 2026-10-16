from __future__ import annotations

import cv2
import numpy as np

from ..types import HeuristicResult
from ..image_utils import ImageData
from ..viz import heatmap_to_base64

WEIGHT = 0.22
ELA_QUALITY = 90


def _reencode_jpeg(bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return bgr
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _blockiness(error: np.ndarray) -> float:
    h, w = error.shape
    eh, ew = h - (h % 8), w - (w % 8)
    e = error[:eh, :ew].astype(np.float32)
    interior = np.zeros_like(e, dtype=bool)
    interior[1::8, :] = True
    interior[:, 1::8] = True
    boundary = ~interior
    bi = float(e[interior].mean()) if interior.any() else 0.0
    bb = float(e[boundary].mean()) if boundary.any() else 0.0
    base = bi + bb + 1e-6
    return abs(bb - bi) / base


def analyze_ela(img: ImageData) -> HeuristicResult:
    bgr = cv2.cvtColor((img.color * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    re = _reencode_jpeg(bgr, ELA_QUALITY)
    h = min(bgr.shape[0], re.shape[0])
    w = min(bgr.shape[1], re.shape[1])
    diff = np.abs(bgr[:h, :w].astype(np.int16) - re[:h, :w].astype(np.int16))
    gray_err = diff.mean(axis=2).astype(np.float32)

    mean_e = float(gray_err.mean())
    std_e = float(gray_err.std())
    cv = std_e / (mean_e + 1e-3)
    block = _blockiness(gray_err)

    score = 0.5
    if mean_e < 1.5 and std_e < 1.5:
        score = 0.85
    elif cv < 0.45:
        score = 0.72
    elif cv > 1.1 and mean_e > 3.0:
        score = 0.22
    elif block > 0.35:
        score = 0.68
    else:
        score = 0.5 + 0.2 * (0.5 - min(cv, 1.0) / 2.0)

    score = float(np.clip(score, 0.0, 1.0))

    viz = {"ela_heatmap": heatmap_to_base64(gray_err, enhance=1.4)}

    details = {
        "mean_error": round(mean_e, 3),
        "std_error": round(std_e, 3),
        "coef_variation": round(cv, 3),
        "blockiness": round(block, 3),
        "reencode_quality": ELA_QUALITY,
    }
    return HeuristicResult(name="ela", score=score, weight=WEIGHT,
                           details=details, viz=viz)