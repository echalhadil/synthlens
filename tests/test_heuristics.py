import io

import numpy as np
from PIL import Image

from synthlens.core.image_utils import load_image
from synthlens.core.heuristics import (
    analyze_metadata, analyze_ela, analyze_noise,
    analyze_frequency, analyze_prnu, analyze_texture_color,
)

ALL = [analyze_metadata, analyze_ela, analyze_noise,
       analyze_frequency, analyze_prnu, analyze_texture_color]


def _img(arr, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return load_image(buf.getvalue())


def _natural():
    rng = np.random.default_rng(1)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(xx / 15.0) * np.cos(yy / 19.0) + 1) / 2
    g = np.clip(base * 255 + rng.normal(0, 10, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _synthetic():
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / max(h, w)
    g = np.clip(1 - r, 0, 1)
    arr = (np.stack([g, g * 0.5, g * 0.2], axis=-1) * 255).astype(np.uint8)
    return arr


def test_each_heuristic_returns_valid():
    for arr in (_natural(), _synthetic()):
        img = _img(arr)
        for fn in ALL:
            r = fn(img)
            assert r.name
            assert 0.0 <= r.score <= 1.0
            assert r.weight > 0
            assert isinstance(r.details, dict)


def test_ela_produces_viz():
    img = _img(_natural())
    r = analyze_ela(img)
    assert "ela_heatmap" in r.viz and r.viz["ela_heatmap"]


def test_frequency_produces_viz():
    img = _img(_natural())
    r = analyze_frequency(img)
    assert "fft_spectrum" in r.viz and r.viz["fft_spectrum"]