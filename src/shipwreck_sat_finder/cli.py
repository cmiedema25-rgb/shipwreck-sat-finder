"""CLI: shipwreck-sat-finder detect|benchmark|serve."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from .pipeline import evaluate_sample_scenes, run_pipeline
from .scene import default_sample_dir, load_scene


def cmd_detect(args: argparse.Namespace) -> int:
    scene = load_scene(Path(args.scene_dir), args.scene_id)
    result = run_pipeline(scene, min_confidence=args.min_confidence)
    overlay_path = Path(args.overlay_out) if args.overlay_out else Path(f"{scene.scene_id}_overlay.png")
    overlay_path.write_bytes(result["overlay_png"])
    table = []
    for p in result["pins_table"]:
        table.append(
            {
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "detect_score": p.get("detect_score"),
                "best_guess": p.get("best_guess"),
                "ship_confidence": p.get("ship_confidence"),
                "catalog_id": p.get("catalog_id"),
                "source": p.get("source"),
            }
        )
    payload = {
        "scene_id": result["scene_id"],
        "detection_metrics": result["detection_metrics"],
        "identification_metrics": result["identification_metrics"],
        "pins": table,
        "overlay": str(overlay_path),
        "notes": result["notes"],
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"\nOverlay written to {overlay_path}", file=sys.stderr)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    report = evaluate_sample_scenes(Path(args.scene_dir) if args.scene_dir else None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}", file=sys.stderr)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    from .webapp import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="shipwreck-sat-finder",
        description="Detect and identify shipwreck candidates on Sentinel-2 coastal chips using NOAA AWOIS/ENC catalogs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("detect", help="Run detection + ship ID on a cached scene")
    p_det.add_argument("--scene-dir", default=str(default_sample_dir()))
    p_det.add_argument("--scene-id", default=None)
    p_det.add_argument("--min-confidence", type=float, default=0.35)
    p_det.add_argument("--overlay-out", default=None)
    p_det.add_argument("--json-out", default=None)
    p_det.set_defaults(func=cmd_detect)

    p_bench = sub.add_parser("benchmark", help="Evaluate metrics on labeled sample scenes")
    p_bench.add_argument("--scene-dir", default=None)
    p_bench.add_argument("--out", default="evidence/benchmark-report.json")
    p_bench.set_defaults(func=cmd_benchmark)

    p_serve = sub.add_parser("serve", help="Start FastAPI + mobile PWA UI")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=7860)
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
