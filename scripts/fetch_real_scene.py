#!/usr/bin/env python3
"""Refresh cached Sentinel-2 RGB chip + NOAA AWOIS wreck GeoJSON for the demo AOI.

Requires network. Writes under data/samples/. Offline CI uses the committed cache.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
import requests
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as tfm_from_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds


DEFAULT_BBOX = [-76.02, 36.88, -75.94, 36.96]
SCENE_ID = "cape_henry_approaches"
AWOIS_QUERY = (
    "https://services5.arcgis.com/HDRa0B57OVrv2E1q/arcgis/rest/services/"
    "Wrecks_and_Obstructions/FeatureServer/0/query"
)


def fetch_wrecks(bbox, out: Path) -> list:
    url = (
        f"{AWOIS_QUERY}?geometry={bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
        "&geometryType=esriGeometryEnvelope&inSR=4326&spatialRel=esriSpatialRelIntersects"
        "&outFields=*&returnGeometry=true&f=geojson&outSR=4326&resultRecordCount=500"
    )
    fc = requests.get(url, timeout=90).json()
    features = []
    for f in fc.get("features", []):
        p = f.get("properties") or {}
        lon = p.get("longitudeD") or f["geometry"]["coordinates"][0]
        lat = p.get("latitudeDD") or f["geometry"]["coordinates"][1]
        name = (p.get("vesselTerm") or "UNKNOWN").strip()
        hist = (p.get("history") or "")[:1200]
        props = {
            "catalog_id": f"AWOIS-{p.get('record') or p.get('OBJECTID')}",
            "source": "NOAA AWOIS / ENC wrecks & obstructions (public GIS synthesis)",
            "name": name,
            "vessel_type": None,
            "year_sunk": p.get("yearSunk") if p.get("yearSunk") not in (0, None, "0") else None,
            "depth": p.get("depth"),
            "depth_units": (p.get("soundingTy") or "").strip() or None,
            "position_quality": p.get("positionQu"),
            "chart": p.get("chart"),
            "history": hist,
            "length_ft_est": None,
            "lat": lat,
            "lon": lon,
        }
        m = re.search(r"(\d+)\s*(?:ft|feet|foot)\s*long", hist, re.I)
        if m:
            props["length_ft_est"] = int(m.group(1))
        m2 = re.search(r"(\d+)\s*[-–]\s*(\d+)\s*-?\s*meter", hist, re.I)
        if m2:
            props["length_m_est"] = (int(m2.group(1)) + int(m2.group(2))) / 2
        features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": props})
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": {
            "aoi_bbox_wgs84": bbox,
            "source": "NOAA OCS AWOIS (2016) + ENC wrecks/obstructions public GIS synthesis",
            "license_note": "U.S. Government public data; research/demo only — NOT for navigation.",
            "service": AWOIS_QUERY.rsplit("/query", 1)[0],
        },
    }
    out.write_text(json.dumps(payload, indent=2))
    return features


def fetch_s2(bbox, out_tif: Path, out_png: Path, size: int = 896) -> dict:
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = list(
        catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime="2024-01-01/2026-12-31",
            query={"eo:cloud_cover": {"lt": 5}},
            max_items=25,
        ).items()
    )
    items = sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 99))
    prefer = [it for it in items if any(t in it.id for t in ("T18SUF", "T18SVF", "T18SUG"))]
    item = pc.sign((prefer or items)[0])
    href = item.assets["visual"].href
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        data = src.read(window=window, out_shape=(src.count, size, size), resampling=Resampling.bilinear)
        rgb = data[:3]
        if rgb.dtype != np.uint8:
            out = np.zeros_like(rgb, dtype=np.uint8)
            for i in range(3):
                band = rgb[i].astype(np.float32)
                lo, hi = np.percentile(band, (2, 98))
                out[i] = np.clip((band - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
            rgb = out
        transform = tfm_from_bounds(left, bottom, right, top, size, size)
        profile = {
            "driver": "GTiff",
            "height": size,
            "width": size,
            "count": 3,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(rgb)
    Image.fromarray(np.transpose(rgb, (1, 2, 0))).save(out_png)
    return {
        "sentinel2_item_id": item.id,
        "sentinel2_datetime": item.properties.get("datetime"),
        "cloud_cover": item.properties.get("eo:cloud_cover"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, default=DEFAULT_BBOX, metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/samples"))
    ap.add_argument("--scene-id", default=SCENE_ID)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wrecks_path = args.out_dir / f"{args.scene_id}_wrecks.geojson"
    feats = fetch_wrecks(args.bbox, wrecks_path)
    tif = args.out_dir / f"{args.scene_id}_s2_rgb.tif"
    png = args.out_dir / f"{args.scene_id}_preview.png"
    s2 = fetch_s2(args.bbox, tif, png)
    named = [f["properties"]["name"] for f in feats if f["properties"]["name"] not in ("UNKNOWN",)]
    meta = {
        "scene_id": args.scene_id,
        "bbox_wgs84": list(args.bbox),
        **s2,
        "stac": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "license": "Sentinel-2 Copernicus open access (with attribution)",
        "width": 896,
        "height": 896,
        "n_wrecks": len(feats),
        "wrecks_file": wrecks_path.name,
        "geotiff": tif.name,
        "preview": png.name,
        "named_vessels": named,
    }
    (args.out_dir / f"{args.scene_id}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
