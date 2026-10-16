from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ..types import HeuristicResult
from ..image_utils import ImageData
from ..viz import figure_to_base64, heatmap_to_base64

WEIGHT = 0.2
MAX_SIDE = 1024


def _radial_profile(mag: np.ndarray, nr: int) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h / 2, w / 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rbin = (r * nr / max(h, w)).astype(np.int32)
    tbin = np.bincount(rbin.ravel(), mag.ravel())
    nrng = np.bincount(rbin.ravel())
    prof = np.zeros_like(tbin, dtype=np.float64)
    nz = nrng > 0
    prof[nz] = tbin[nz] / nrng[nz]
    return prof


def _resize(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    if max(h, w) <= MAX_SIDE:
        return g
    scale = MAX_SIDE / max(h, w)
    return cv2.resize(g, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def analyze_frequency(img: ImageData) -> HeuristicResult:
    g = _resize(img.gray.astype(np.float32))
    g = g - g.mean()
    h, w = g.shape

    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F) ** 2
    mag = np.log1p(mag)

    nr = min(h, w) // 2
    prof = _radial_profile(mag, nr)
    r = np.arange(2, len(prof))
    p = prof[2:]
    logr = np.log(r + 1e-9)
    logp = np.log(p + 1e-9)
    slope = float(np.polyfit(logr, logp, 1)[0]) if len(r) > 4 else 0.0

    window = max(3, len(p) // 20)
    kernel = np.ones(window) / window
    smooth = np.convolve(p, kernel, mode="same")
    prom = float(np.max((p - smooth) / (smooth + 1e-9)))

    total = p.sum() + 1e-9
    hf_ratio = float(p[len(p) // 3:].sum() / total)

    score = 0.5
    if prom > 0.5:
        score = 0.82
    elif hf_ratio < 0.12:
        score = 0.72
    elif -2.6 < slope < -1.4 and prom < 0.25:
        score = 0.22
    elif hf_ratio > 0.45:
        score = 0.6
    else:
        score = 0.5
    score = float(np.clip(score, 0.0, 1.0))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.semilogy(r, p, lw=0.8)
    ax.semilogy(r, smooth, lw=1.2, ls="--")
    ax.set_xlabel("radius")
    ax.set_ylabel("power")
    ax.set_title("radial power spectrum")
    viz = {
        "fft_spectrum": figure_to_base64(fig),
        "fft_heatmap": heatmap_to_base64(mag, enhance=0.6),
    }

    details = {
        "slope": round(slope, 3),
        "peak_prominence": round(prom, 3),
        "hf_ratio": round(hf_ratio, 3),
    }
    return HeuristicResult(name="frequency", score=score, weight=WEIGHT,
                           details=details, viz=viz)