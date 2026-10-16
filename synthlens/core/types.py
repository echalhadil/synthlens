from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class HeuristicResult:
    name: str
    score: float
    weight: float
    details: Dict = field(default_factory=dict)
    viz: Dict[str, str] = field(default_factory=dict)

    @property
    def weighted(self) -> float:
        return self.score * self.weight


@dataclass
class DetectionResult:
    verdict: str
    probability: float
    heuristics: Dict[str, HeuristicResult] = field(default_factory=dict)

    def to_dict(self, include_viz: bool = False) -> dict:
        out = {
            "verdict": self.verdict,
            "probability": float(self.probability),
            "heuristics": {},
        }
        for name, h in self.heuristics.items():
            hd = {
                "score": float(h.score),
                "weight": float(h.weight),
                "details": {k: _jsonable(v) for k, v in h.details.items()},
            }
            if include_viz:
                hd["viz"] = dict(h.viz)
            out["heuristics"][name] = hd
        return out


def _jsonable(v):
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)