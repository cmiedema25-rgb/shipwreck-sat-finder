"""End-to-end detect → identify → metrics pipeline."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from .detect import Candidate, detect_wrecks, draw_overlay
from .identify import (
    attach_identities,
    catalog_pins,
    haversine_m,
    identification_precision_at_1,
    identify_ship,
)
from .metrics import aggregate_metrics, match_detections
from .scene import CatalogRecord, Scene, default_sample_dir, load_scene


def _meters_per_pixel(scene: Scene) -> float:
    """Approximate m/px using mid-latitude lon span."""
    mid_lat = 0.5 * (scene.bounds.lat_min + scene.bounds.lat_max)
    lon_m = haversine_m(mid_lat, scene.bounds.lon_min, mid_lat, scene.bounds.lon_max)
    return lon_m / max(scene.width - 1, 1)


def run_pipeline(
    scene: Scene,
    *,
    min_confidence: float = 0.35,
    match_radius_px: float = 45.0,
    include_catalog_pins: bool = True,
) -> Dict[str, Any]:
    """Detect candidates, identify ships, score vs catalog, build overlay."""
    candidates = detect_wrecks(scene.image, scene.bounds, min_confidence=min_confidence)
    mpp = _meters_per_pixel(scene)
    det_rows = attach_identities(candidates, scene.catalog, bounds_span_m=(mpp, mpp))
    for row in det_rows:
        row["source"] = "cv_detect"

    cat_rows = catalog_pins(scene.catalog) if include_catalog_pins else []

    # Match CV detections to catalog points in pixel space
    gt = []
    for rec in scene.catalog:
        x, y = scene.bounds.latlon_to_pixel(rec.lat, rec.lon)
        gt.append({"cx": x, "cy": y, "catalog_id": rec.catalog_id, "name": rec.name})

    match = match_detections(gt, candidates, radius_px=match_radius_px)
    id_metrics = identification_precision_at_1(det_rows, scene.catalog)

    overlay = draw_overlay(scene.image, candidates)
    # Also mark catalog pins in cyan
    from PIL import ImageDraw

    draw = ImageDraw.Draw(overlay)
    for rec in scene.catalog:
        x, y = scene.bounds.latlon_to_pixel(rec.lat, rec.lon)
        r = 7
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(80, 220, 255), width=2)
        label = rec.name if rec.name.upper() not in ("UNKNOWN", "OBSTRUCTION") else rec.catalog_id
        draw.text((x + r + 2, y - 6), label[:22], fill=(80, 220, 255))

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    overlay_bytes = buf.getvalue()

    return {
        "scene_id": scene.scene_id,
        "meta": scene.meta,
        "bounds": {
            "lat_min": scene.bounds.lat_min,
            "lat_max": scene.bounds.lat_max,
            "lon_min": scene.bounds.lon_min,
            "lon_max": scene.bounds.lon_max,
            "width": scene.width,
            "height": scene.height,
        },
        "n_catalog": len(scene.catalog),
        "n_detections": len(candidates),
        "detections": det_rows,
        "catalog_pins": cat_rows,
        "pins_table": cat_rows + det_rows,
        "detection_metrics": {
            "tp": match.tp,
            "fp": match.fp,
            "fn": match.fn,
            "precision": round(match.precision, 4),
            "recall": round(match.recall, 4),
            "f1": round(match.f1, 4),
            "match_radius_px": match_radius_px,
        },
        "identification_metrics": id_metrics,
        "overlay_png": overlay_bytes,
        "meters_per_pixel": round(mpp, 2),
        "notes": (
            "Catalog pins are NOAA AWOIS/ENC open records (primary proof). "
            "CV detections on optical Sentinel-2 of submerged wrecks are often partial — "
            "metrics reflect that honesty."
        ),
    }


def run_default_sample(min_confidence: float = 0.35) -> Dict[str, Any]:
    scene = load_scene(default_sample_dir())
    return run_pipeline(scene, min_confidence=min_confidence)


def evaluate_sample_scenes(sample_dir: Optional[Path] = None) -> Dict[str, Any]:
    sample_dir = Path(sample_dir or default_sample_dir())
    metas = sorted(sample_dir.glob("*_meta.json"))
    results = []
    match_list = []
    id_list = []
    for meta_path in metas:
        sid = meta_path.name.replace("_meta.json", "")
        scene = load_scene(sample_dir, sid)
        out = run_pipeline(scene)
        dm = out["detection_metrics"]
        from .metrics import MatchResult

        match_list.append(
            MatchResult(
                tp=dm["tp"],
                fp=dm["fp"],
                fn=dm["fn"],
                precision=dm["precision"],
                recall=dm["recall"],
                f1=dm["f1"],
                matched_pairs=[],
            )
        )
        id_list.append(out["identification_metrics"])
        results.append(
            {
                "scene_id": sid,
                "n_catalog": out["n_catalog"],
                "n_detections": out["n_detections"],
                "detection_metrics": dm,
                "identification_metrics": out["identification_metrics"],
                "named_examples": [
                    p["best_guess"]
                    for p in out["catalog_pins"]
                    if p["best_guess"]
                    and str(p["best_guess"]).upper()
                    not in ("UNKNOWN", "WRECK", "OBSTRUCTION")
                    and not str(p["best_guess"]).startswith("Unnamed")
                    and not str(p["best_guess"]).startswith("Charted")
                ][:8],
            }
        )
    agg = aggregate_metrics(match_list) if match_list else {}
    # Micro ID precision
    n_eval = sum(i.get("n_eval") or 0 for i in id_list)
    n_correct = sum(i.get("n_correct") or 0 for i in id_list)
    id_p = (n_correct / n_eval) if n_eval else None
    return {
        "scenes": results,
        "aggregate_detection": agg,
        "aggregate_identification": {
            "precision_at_1": None if id_p is None else round(id_p, 4),
            "n_eval": n_eval,
            "n_correct": n_correct,
        },
    }
