from __future__ import annotations

from typing import Union

from .types import HeuristicResult, DetectionResult
from .image_utils import ImageData, load_image
from .scoring import fuse_scores, verdict_from_probability
from .heuristics import ALL_HEURISTICS

__all__ = ["detect", "DetectionResult", "HeuristicResult"]

Source = Union[str, bytes, "object"]


def detect(source: Source, include_viz: bool = True) -> DetectionResult:
    img: ImageData = load_image(source)
    results: dict[str, HeuristicResult] = {}
    for fn in ALL_HEURISTICS:
        try:
            r = fn(img)
        except Exception as exc:
            r = HeuristicResult(name=fn.__name__.replace("analyze_", ""),
                                score=0.5, weight=0.0,
                                details={"error": str(exc)})
        if not include_viz:
            r.viz = {}
        results[r.name] = r

    p = fuse_scores(results)
    verdict = verdict_from_probability(p)
    return DetectionResult(verdict=verdict, probability=p, heuristics=results)