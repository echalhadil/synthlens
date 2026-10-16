from __future__ import annotations

import cv2
import numpy as np

from ..types import HeuristicResult
from ..image_utils import ImageData

WEIGHT = 0.1


def _noise_residual(g8: np.ndarray) -> np.ndarray:
    den = cv2.GaussianBlur(g8, (5, 5), 0)
    return g8.astype(np.float32) - den.astype(np.float32)


def _block_correlation(resid: np.ndarray, block: int = 64) -> float:
    h, w = resid.shape
    eh, ew = h - (h % block), w - (w % block)
    a = resid[:eh, :ew]
    grid = a.reshape(eh // block, block, ew // block, block).mean(axis=(1, 3))
    g = grid - grid.mean()
    s = g.std()
    if s < 1e-6:
        return 0.0
    gnorm = g / s
    rows = gnorm.shape[0]
    corrs = []
    for i in range(rows - 1):
        denom = gnorm.shape[1]
        c = float((gnorm[i] * gnorm[i + 1]).sum() / denom)
        corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


def analyze_prnu(img: ImageData) -> HeuristicResult:
    g8 = (img.gray * 255).astype(np.uint8)
    resid = _noise_residual(g8)

    block_corr = _block_correlation(resid, block=64)
    abs_mean = float(np.abs(resid).mean())
    tail = float((np.abs(resid) > 3 * resid.std()).mean())

    score = 0.5
    if 0.15 < block_corr < 0.6:
        score = 0.25
    elif block_corr > 0.75:
        score = 0.7
    elif abs_mean < 0.8 and tail < 0.01:
        score = 0.72
    else:
        score = 0.5
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "block_correlation": round(block_corr, 3),
        "residual_abs_mean": round(abs_mean, 3),
        "outlier_fraction": round(tail, 4),
    }
    return HeuristicResult(name="prnu", score=score, weight=WEIGHT,
                           details=details)