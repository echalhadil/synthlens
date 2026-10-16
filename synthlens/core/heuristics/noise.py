from __future__ import annotations

import cv2
import numpy as np

from ..types import HeuristicResult
from ..image_utils import ImageData
from ..viz import heatmap_to_base64

WEIGHT = 0.17


def _local_std_map(arr: np.ndarray, block: int = 8) -> np.ndarray:
    h, w = arr.shape
    eh, ew = h - (h % block), w - (w % block)
    a = arr[:eh, :ew].reshape(eh // block, block, ew // block, block)
    return a.std(axis=(1, 3))


def analyze_noise(img: ImageData) -> HeuristicResult:
    g8 = (img.gray * 255).astype(np.uint8)
    den = cv2.fastNlMeansDenoising(g8, None, h=7, templateWindowSize=7,
                                   searchWindowSize=21)
    resid = g8.astype(np.float32) - den.astype(np.float32)

    r_std = float(resid.std())
    r_mean_abs = float(np.abs(resid).mean())

    std_map = _local_std_map(resid, block=16)
    spatial_cv = float(std_map.std() / (std_map.mean() + 1e-6))

    hist, _ = np.histogram(resid, bins=64, range=(-30, 30), density=True)
    hist = hist / (hist.sum() + 1e-9)
    entropy = float(-(hist * np.log(hist + 1e-12)).sum())

    score = 0.5
    if r_std < 2.0:
        score = 0.8
    elif r_std > 6.0 and spatial_cv > 0.6:
        score = 0.2
    elif spatial_cv < 0.25:
        score = 0.72
    elif entropy < 3.0:
        score = 0.65
    else:
        score = 0.45

    score = float(np.clip(score, 0.0, 1.0))
    viz = {"noise_residual": heatmap_to_base64(resid, enhance=1.8)}

    details = {
        "residual_std": round(r_std, 3),
        "residual_abs_mean": round(r_mean_abs, 3),
        "spatial_cv": round(spatial_cv, 3),
        "entropy": round(entropy, 3),
    }
    return HeuristicResult(name="noise", score=score, weight=WEIGHT,
                           details=details, viz=viz)