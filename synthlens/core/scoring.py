from __future__ import annotations

from .types import HeuristicResult

REAL_BAND = 0.35
AI_BAND = 0.65


def fuse_scores(results: dict[str, HeuristicResult]) -> float:
    total_w = sum(r.weight for r in results.values())
    if total_w <= 0:
        return 0.5
    return sum(r.weighted for r in results.values()) / total_w


def verdict_from_probability(p: float) -> str:
    if p < REAL_BAND:
        return "Likely Real"
    if p > AI_BAND:
        return "Likely AI-Generated"
    return "Uncertain"