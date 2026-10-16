import io

import numpy as np
from PIL import Image

from synthlens.core.detector import detect
from synthlens.core.image_utils import load_image

EXPECTED = {"metadata", "ela", "noise", "frequency", "prnu", "texture_color"}


def _natural_image(h=256, w=256, noise=12):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.sin(xx / 17.0) * np.cos(yy / 23.0)
    base = (base + 1) / 2
    img = (base * 255 + rng.normal(0, noise, (h, w)))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _synthetic_image(h=256, w=256):
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / max(h, w)
    g = np.clip(1 - r, 0, 1)
    arr = np.stack([g, g * 0.6, g * 0.3], axis=-1)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def _to_bytes(arr, fmt="PNG", exif=None):
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def _camera_exif():
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "EOS R6"
    exif[0x9003] = "2024:05:01 12:00:00"
    return exif.tobytes()


def test_detect_structure():
    data = _to_bytes(_natural_image(), "PNG")
    res = detect(data)
    assert res.verdict in {"Likely Real", "Uncertain", "Likely AI-Generated"}
    assert 0.0 <= res.probability <= 1.0
    assert set(res.heuristics) == EXPECTED
    for h in res.heuristics.values():
        assert 0.0 <= h.score <= 1.0


def test_metadata_exif_lowers_score():
    natural = _natural_image()
    with_exif = _to_bytes(natural, "JPEG", exif=_camera_exif())
    no_exif = _to_bytes(natural, "JPEG")

    loaded = load_image(with_exif)
    assert "Make" in loaded.exif and "Model" in loaded.exif

    r_with = detect(with_exif).heuristics["metadata"].score
    r_none = detect(no_exif).heuristics["metadata"].score
    assert r_with < r_none


def test_jsonable_dict():
    data = _to_bytes(_synthetic_image(), "PNG")
    d = detect(data).to_dict(include_viz=False)
    assert "verdict" in d and "probability" in d and "heuristics" in d
    assert "viz" not in d["heuristics"]["ela"]