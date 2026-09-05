"""FastAPI backend + mobile-first PWA frontend."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .pipeline import run_pipeline
from .scene import default_sample_dir, load_scene

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Shipwreck Sat Finder",
    description="Sentinel-2 + NOAA AWOIS shipwreck pin & identity demo",
    version="0.1.0",
)

if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC / "index.html"
    if not index_path.exists():
        raise HTTPException(500, "UI missing")
    return HTMLResponse(index_path.read_text())


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenes")
def list_scenes() -> Dict[str, Any]:
    d = default_sample_dir()
    scenes = []
    for meta in sorted(d.glob("*_meta.json")):
        import json

        m = json.loads(meta.read_text())
        scenes.append(
            {
                "scene_id": m.get("scene_id", meta.name.replace("_meta.json", "")),
                "n_wrecks": m.get("n_wrecks"),
                "bbox_wgs84": m.get("bbox_wgs84"),
                "sentinel2_item_id": m.get("sentinel2_item_id"),
                "named_vessels": m.get("named_vessels", []),
            }
        )
    return {"scenes": scenes}


@app.get("/api/detect")
def api_detect(
    scene_id: Optional[str] = Query(None),
    min_confidence: float = Query(0.35, ge=0.0, le=1.0),
) -> JSONResponse:
    try:
        scene = load_scene(default_sample_dir(), scene_id)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    result = run_pipeline(scene, min_confidence=min_confidence)
    overlay_b64 = base64.b64encode(result["overlay_png"]).decode("ascii")
    payload = {
        "scene_id": result["scene_id"],
        "meta": result["meta"],
        "bounds": result["bounds"],
        "n_catalog": result["n_catalog"],
        "n_detections": result["n_detections"],
        "detection_metrics": result["detection_metrics"],
        "identification_metrics": result["identification_metrics"],
        "pins_table": result["pins_table"],
        "catalog_pins": result["catalog_pins"],
        "detections": result["detections"],
        "overlay_png_base64": overlay_b64,
        "meters_per_pixel": result["meters_per_pixel"],
        "notes": result["notes"],
    }
    return JSONResponse(payload)


@app.get("/api/preview/{scene_id}")
def preview(scene_id: str) -> Response:
    path = default_sample_dir() / f"{scene_id}_preview.png"
    if not path.exists():
        raise HTTPException(404, "preview not found")
    return FileResponse(path, media_type="image/png")
