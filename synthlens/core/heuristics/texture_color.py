from __future__ import annotations

import cv2
import numpy as np

from ..types import HeuristicResult
from ..image_utils import ImageData

WEIGHT = 0.06


def _lbp_uniformity(gray8: np.ndarray) -> float:
    g = gray8.astype(np.int32)
    h, w = g.shape
    center = g[1:-1, 1:-1]
    shifts = [
        g[0:-2, 0:-2], g[0:-2, 1:-1], g[0:-2, 2:],
        g[1:-1, 0:-2], g[1:-1, 2:], g[2:, 0:-2],
        g[2:, 1:-1], g[2:, 2:],
    ]
    code = np.zeros_like(center)
    for i, s in enumerate(shifts):
        code += (s >= center).astype(np.int32) << i
    hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
    hist /= hist.sum() + 1e-9
    top = np.sort(hist)[::-1]
    return float(top[:9].sum())


def analyze_texture_color(img: ImageData) -> HeuristicResult:
    g8 = (img.gray * 255).astype(np.uint8)

    sx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_density = float((edge_mag > 30).mean())

    lbp_u = _lbp_uniformity(g8)

    hsv = cv2.cvtColor((img.color * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255.0
    sat_mean = float(sat.mean())
    sat_std = float(sat.std())

    r, gg, b = img.color[..., 0], img.color[..., 1], img.color[..., 2]
    corr_rg = float(np.corrcoef(r.ravel(), gg.ravel())[0, 1])
    corr_rb = float(np.corrcoef(r.ravel(), b.ravel())[0, 1])

    score = 0.5
    if lbp_u > 0.75:
        score += 0.15
    if edge_density < 0.04:
        score += 0.1
    if sat_std < 0.12:
        score += 0.08
    if corr_rg > 0.95 and corr_rb > 0.9:
        score += 0.07
    score = float(np.clip(score, 0.0, 1.0))

    details = {
        "edge_density": round(edge_density, 4),
        "lbp_uniformity": round(lbp_u, 3),
        "sat_mean": round(sat_mean, 3),
        "sat_std": round(sat_std, 3),
        "corr_rg": round(corr_rg, 3),
        "corr_rb": round(corr_rb, 3),
    }
    return HeuristicResult(name="texture_color", score=score, weight=WEIGHT,
                           details=details)