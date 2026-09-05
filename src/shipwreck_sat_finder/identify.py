"""Ship identification via spatial join to NOAA AWOIS / public wreck catalogs.

Honest approach: distance + optional size ranking against open records only.
Never invent vessel names not present in the catalog for the AOI.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .detect import Candidate
from .scene import CatalogRecord

EARTH_R_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_R_M * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class AltGuess:
    name: str
    catalog_id: str
    distance_m: float
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShipGuess:
    best_guess: str
    confidence: float
    catalog_id: Optional[str]
    distance_m: Optional[float]
    alternatives: List[AltGuess] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_guess": self.best_guess,
            "confidence": round(self.confidence, 4),
            "catalog_id": self.catalog_id,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
            "alternatives": [a.to_dict() for a in self.alternatives],
            "rationale": self.rationale,
        }


def _length_score(det_length_px: float, meters_per_px: float, rec: CatalogRecord) -> float:
    """Optional agreement between detected blob length and catalog size."""
    cat_m = rec.length_m_est
    if cat_m is None and rec.length_ft_est:
        cat_m = float(rec.length_ft_est) * 0.3048
    if cat_m is None or meters_per_px <= 0:
        return 0.5  # neutral when unknown
    det_m = det_length_px * meters_per_px
    if det_m <= 0:
        return 0.5
    ratio = min(det_m, cat_m) / max(det_m, cat_m)
    return float(ratio)


def _completeness(rec: CatalogRecord) -> float:
    score = 0.2
    if rec.name and rec.name.upper() not in ("UNKNOWN", "WRECK", "OBSTRUCTION", ""):
        score += 0.45
    elif rec.name and rec.name.upper() in ("WRECK", "OBSTRUCTION"):
        score += 0.15
    if rec.history and len(rec.history) > 40:
        score += 0.2
    if rec.year_sunk:
        score += 0.1
    if rec.length_m_est or rec.length_ft_est:
        score += 0.05
    return min(1.0, score)


def identify_ship(
    lat: float,
    lon: float,
    catalog: Sequence[CatalogRecord],
    *,
    det_length_px: float = 0.0,
    meters_per_px: float = 0.0,
    max_distance_m: float = 500.0,
) -> ShipGuess:
    """Rank catalog vessels near a pin; return best guess + alternatives."""
    if not catalog:
        return ShipGuess(
            best_guess="Unknown / uncatalogued",
            confidence=0.05,
            catalog_id=None,
            distance_m=None,
            alternatives=[],
            rationale="No public wreck catalog loaded for this scene.",
        )

    scored: List[tuple[float, float, CatalogRecord]] = []
    for rec in catalog:
        dist = haversine_m(lat, lon, rec.lat, rec.lon)
        if dist > max_distance_m:
            continue
        # Distance score: 1 at 0m, ~0 at max_distance
        d_score = max(0.0, 1.0 - dist / max_distance_m)
        l_score = _length_score(det_length_px, meters_per_px, rec)
        c_score = _completeness(rec)
        score = 0.55 * d_score + 0.20 * l_score + 0.25 * c_score
        # Prefer named vessels slightly when distances are similar
        if rec.name.upper() not in ("UNKNOWN", "WRECK", "OBSTRUCTION"):
            score += 0.05
        scored.append((score, dist, rec))

    scored.sort(key=lambda t: (-t[0], t[1]))
    if not scored:
        return ShipGuess(
            best_guess="Unknown / uncatalogued",
            confidence=0.08,
            catalog_id=None,
            distance_m=None,
            alternatives=[],
            rationale=f"No NOAA/AWOIS catalog record within {max_distance_m:.0f} m.",
        )

    best_score, best_dist, best = scored[0]
    alts = [
        AltGuess(name=r.name, catalog_id=r.catalog_id, distance_m=round(d, 1), score=round(s, 4))
        for s, d, r in scored[1:4]
    ]

    # Display name honesty
    display = best.name
    if display.upper() in ("UNKNOWN", ""):
        display = f"Unnamed catalog wreck ({best.catalog_id})"
    elif display.upper() == "OBSTRUCTION":
        display = f"Charted obstruction ({best.catalog_id})"
    elif display.upper() == "WRECK":
        display = f"Unnamed wreck ({best.catalog_id})"

    conf = float(min(0.97, max(0.1, best_score)))
    parts = [f"{best_dist:.0f} m from {best.catalog_id}"]
    if best.name.upper() not in ("UNKNOWN", "WRECK", "OBSTRUCTION"):
        parts.append(f"catalog name '{best.name}'")
    if best.length_m_est or best.length_ft_est:
        parts.append("size feature compared when available")
    if best.history:
        parts.append("history text present in open record")
    rationale = "; ".join(parts) + ". Identification is catalog proximity ranking, not visual recognition."

    return ShipGuess(
        best_guess=display,
        confidence=conf,
        catalog_id=best.catalog_id,
        distance_m=best_dist,
        alternatives=alts,
        rationale=rationale,
    )


def attach_identities(
    candidates: Sequence[Candidate],
    catalog: Sequence[CatalogRecord],
    bounds_span_m: Optional[tuple[float, float]] = None,
) -> List[Dict[str, Any]]:
    """Merge detection candidates with ship guesses for table/API output."""
    mpp = 0.0
    if bounds_span_m and bounds_span_m[0] > 0:
        # approximate meters per pixel from lon span / width already handled by caller
        mpp = bounds_span_m[0]

    rows: List[Dict[str, Any]] = []
    for c in candidates:
        guess = identify_ship(
            c.lat,
            c.lon,
            catalog,
            det_length_px=c.length_px,
            meters_per_px=mpp,
        )
        rows.append(
            {
                "lat": round(c.lat, 6),
                "lon": round(c.lon, 6),
                "cx": round(c.cx, 1),
                "cy": round(c.cy, 1),
                "length_px": round(c.length_px, 1),
                "detect_score": round(c.confidence, 4),
                "best_guess": guess.best_guess,
                "ship_confidence": round(guess.confidence, 4),
                "catalog_id": guess.catalog_id,
                "distance_m": guess.distance_m,
                "rationale": guess.rationale,
                "alternatives": [a.to_dict() for a in guess.alternatives],
                "features": c.features,
            }
        )
    return rows


def catalog_pins(catalog: Sequence[CatalogRecord]) -> List[Dict[str, Any]]:
    """Pins directly from the public wreck catalog (primary reviewer proof)."""
    pins = []
    for rec in catalog:
        pins.append(
            {
                "lat": round(rec.lat, 6),
                "lon": round(rec.lon, 6),
                "detect_score": None,
                "best_guess": rec.name,
                "ship_confidence": 1.0 if rec.name.upper() not in ("UNKNOWN",) else 0.4,
                "catalog_id": rec.catalog_id,
                "distance_m": 0.0,
                "rationale": "Ground-truth pin from open NOAA AWOIS/ENC catalog (not CV-detected).",
                "alternatives": [],
                "history": rec.history[:400],
                "source": "catalog",
            }
        )
    return pins


def identification_precision_at_1(
    pins: Sequence[Dict[str, Any]],
    catalog: Sequence[CatalogRecord],
    match_radius_m: float = 500.0,
) -> Dict[str, Any]:
    """Among CV pins that have a catalog neighbor, fraction whose top guess matches nearest named record."""
    if not pins or not catalog:
        return {"precision_at_1": None, "n_eval": 0, "n_correct": 0}

    correct = 0
    evaluated = 0
    for p in pins:
        if p.get("source") == "catalog":
            continue
        lat, lon = p["lat"], p["lon"]
        nearby = sorted(
            ((haversine_m(lat, lon, r.lat, r.lon), r) for r in catalog),
            key=lambda t: t[0],
        )
        if not nearby or nearby[0][0] > match_radius_m:
            continue
        evaluated += 1
        nearest = nearby[0][1]
        guess_id = p.get("catalog_id")
        if guess_id and guess_id == nearest.catalog_id:
            correct += 1
        elif p.get("best_guess") and nearest.name in str(p.get("best_guess")):
            correct += 1

    prec = (correct / evaluated) if evaluated else None
    return {
        "precision_at_1": None if prec is None else round(prec, 4),
        "n_eval": evaluated,
        "n_correct": correct,
    }
