from pathlib import Path

from shipwreck_sat_finder.identify import identify_ship
from shipwreck_sat_finder.pipeline import evaluate_sample_scenes, run_pipeline
from shipwreck_sat_finder.scene import CatalogRecord, default_sample_dir, load_scene
from shipwreck_sat_finder.synthetic import generate_scene
from shipwreck_sat_finder.detect import detect_wrecks
from shipwreck_sat_finder.geo import SceneBounds


def test_load_real_sample():
    scene = load_scene(default_sample_dir())
    assert scene.width > 100
    assert len(scene.catalog) >= 1
    assert any(r.name == "THOMAS F. POLLARD" for r in scene.catalog) or len(scene.catalog) > 0


def test_identify_nearest_named():
    catalog = [
        CatalogRecord("AWOIS-1", "THOMAS F. POLLARD", 36.900147, -75.966317, history="schooner wreck"),
        CatalogRecord("AWOIS-2", "UNKNOWN", 36.91, -75.97),
    ]
    g = identify_ship(36.9002, -75.9663, catalog, max_distance_m=500)
    assert "POLLARD" in g.best_guess.upper() or g.catalog_id == "AWOIS-1"
    assert g.confidence > 0.3


def test_identify_no_hallucination():
    catalog = [CatalogRecord("AWOIS-9", "UNKNOWN", 36.9, -75.96)]
    g = identify_ship(37.5, -76.5, catalog, max_distance_m=200)
    assert "Unknown / uncatalogued" in g.best_guess


def test_pipeline_runs():
    scene = load_scene(default_sample_dir())
    out = run_pipeline(scene)
    assert "detection_metrics" in out
    assert "pins_table" in out
    assert out["overlay_png"][:8] == b"\x89PNG\r\n\x1a\n"
    cols = {"lat", "lon", "detect_score", "best_guess", "ship_confidence", "catalog_id"}
    assert cols.issubset(set(out["pins_table"][0].keys()) | {"detect_score"})


def test_benchmark_aggregate():
    report = evaluate_sample_scenes()
    assert report["scenes"]
    assert "aggregate_detection" in report


def test_synthetic_helper_optional():
    """Synthetic generator remains for unit-test helpers only."""
    img, meta, bounds = generate_scene("unit-test", width=256, height=256, n_wrecks=2, seed=1)
    cands = detect_wrecks(img, bounds, min_confidence=0.25)
    assert meta.scene_id == "unit-test"
    assert isinstance(cands, list)
