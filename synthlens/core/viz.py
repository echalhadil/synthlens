from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def heatmap_to_base64(arr: np.ndarray, cmap: int = cv2.COLORMAP_INFERNO,
                      enhance: float = 1.0) -> str:
    a = arr.astype(np.float32)
    lo, hi = float(np.percentile(a, 2)), float(np.percentile(a, 98))
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0, 1) ** enhance
    u = (a * 255).astype(np.uint8)
    colored = cv2.applyColorMap(u, cmap)
    ok, buf = cv2.imencode(".png", colored)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def figure_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")