"""ROI / reviewer metrics: precision, recall, F1 on labeled synthetic scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .detect import Candidate


@dataclass
class MatchResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    matched_pairs: List[Tuple[int, int]]  # (gt_idx, det_idx)


def _dist(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def match_detections(
    gt: Sequence[dict],
    dets: Sequence[Candidate],
    radius_px: float = 40.0,
) -> MatchResult:
    """Greedy nearest-neighbor matching within radius (pixel space)."""
    used_det = set()
    pairs: List[Tuple[int, int]] = []
    for gi, g in enumerate(gt):
        gx, gy = float(g["cx"]), float(g["cy"])
        best_j, best_d = -1, float("inf")
        for j, d in enumerate(dets):
            if j in used_det:
                continue
            dist = _dist(gx, gy, d.cx, d.cy)
            if dist < best_d:
                best_d, best_j = dist, j
        if best_j >= 0 and best_d <= radius_px:
            used_det.add(best_j)
            pairs.append((gi, best_j))

    tp = len(pairs)
    fp = len(dets) - tp
    fn = len(gt) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return MatchResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, matched_pairs=pairs)


def aggregate_metrics(results: List[MatchResult]) -> dict:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "scenes": len(results),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_scene": [
            {
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
                "precision": round(r.precision, 4),
                "recall": round(r.recall, 4),
                "f1": round(r.f1, 4),
            }
            for r in results
        ],
    }
