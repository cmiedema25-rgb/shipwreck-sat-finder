"""Load georeferenced Sentinel-2 chips and NOAA wreck catalogs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .geo import SceneBounds

try:
    import rasterio
    from rasterio.warp import transform as rio_transform
except ImportError:  # pragma: no cover
    rasterio = None


@dataclass
class CatalogRecord:
    catalog_id: str
    name: str
    lat: float
    lon: float
    year_sunk: Optional[Any] = None
    depth: Optional[float] = None
    depth_units: Optional[str] = None
    history: str = ""
    length_m_est: Optional[float] = None
    length_ft_est: Optional[float] = None
    position_quality: Optional[str] = None
    chart: Optional[str] = None
    source: str = ""
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "catalog_id": self.catalog_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "year_sunk": self.year_sunk,
            "depth": self.depth,
            "depth_units": self.depth_units,
            "history": self.history,
            "length_m_est": self.length_m_est,
            "length_ft_est": self.length_ft_est,
            "position_quality": self.position_quality,
            "chart": self.chart,
            "source": self.source,
        }


@dataclass
class Scene:
    scene_id: str
    image: Image.Image
    bounds: SceneBounds
    catalog: List[CatalogRecord]
    meta: Dict[str, Any]
    geotiff_path: Optional[Path] = None

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]


def load_catalog(geojson_path: Path) -> List[CatalogRecord]:
    data = json.loads(Path(geojson_path).read_text())
    records: List[CatalogRecord] = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates") or [props.get("lon"), props.get("lat")]
        lon, lat = float(coords[0]), float(coords[1])
        length_m = props.get("length_m_est")
        if length_m is None and props.get("length_ft_est"):
            length_m = float(props["length_ft_est"]) * 0.3048
        records.append(
            CatalogRecord(
                catalog_id=str(props.get("catalog_id") or props.get("objectid") or f"rec-{len(records)}"),
                name=str(props.get("name") or "UNKNOWN").strip() or "UNKNOWN",
                lat=lat,
                lon=lon,
                year_sunk=props.get("year_sunk"),
                depth=props.get("depth"),
                depth_units=props.get("depth_units"),
                history=str(props.get("history") or "")[:2000],
                length_m_est=float(length_m) if length_m is not None else None,
                length_ft_est=props.get("length_ft_est"),
                position_quality=props.get("position_quality"),
                chart=props.get("chart"),
                source=str(props.get("source") or ""),
                raw=props,
            )
        )
    return records


def bounds_from_geotiff(path: Path) -> Tuple[SceneBounds, Image.Image]:
    """Read RGB GeoTIFF and compute WGS84 SceneBounds via corner transform."""
    if rasterio is None:
        raise ImportError("rasterio is required to load GeoTIFF scenes")
    with rasterio.open(path) as src:
        data = src.read()
        if data.shape[0] >= 3:
            rgb = np.transpose(data[:3], (1, 2, 0))
        else:
            band = data[0]
            rgb = np.stack([band, band, band], axis=-1)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        h, w = rgb.shape[0], rgb.shape[1]
        # pixel corners: (0,0), (w-1,0), (0,h-1), (w-1,h-1) in raster row/col
        xs = np.array([0, w - 1, 0, w - 1], dtype=float)
        ys = np.array([0, 0, h - 1, h - 1], dtype=float)
        # rasterio xy uses center of pixel convention with offset
        lons_src, lats_src = rasterio.transform.xy(src.transform, ys, xs, offset="center")
        if src.crs and src.crs.to_string() != "EPSG:4326":
            lons, lats = rio_transform(src.crs, "EPSG:4326", lons_src, lats_src)
        else:
            lons, lats = lons_src, lats_src
        bounds = SceneBounds(
            lat_min=float(min(lats)),
            lat_max=float(max(lats)),
            lon_min=float(min(lons)),
            lon_max=float(max(lons)),
            width=w,
            height=h,
        )
        return bounds, Image.fromarray(rgb, mode="RGB")


def load_scene(
    scene_dir: Path,
    scene_id: Optional[str] = None,
) -> Scene:
    """Load a cached sample scene (GeoTIFF + wrecks GeoJSON + meta JSON)."""
    scene_dir = Path(scene_dir)
    if scene_id is None:
        metas = sorted(scene_dir.glob("*_meta.json"))
        if not metas:
            raise FileNotFoundError(f"No *_meta.json in {scene_dir}")
        meta_path = metas[0]
        scene_id = meta_path.name.replace("_meta.json", "")
    else:
        meta_path = scene_dir / f"{scene_id}_meta.json"

    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"scene_id": scene_id}
    tif = scene_dir / meta.get("geotiff", f"{scene_id}_s2_rgb.tif")
    preview = scene_dir / meta.get("preview", f"{scene_id}_preview.png")
    wrecks = scene_dir / meta.get("wrecks_file", f"{scene_id}_wrecks.geojson")

    if tif.exists() and rasterio is not None:
        bounds, image = bounds_from_geotiff(tif)
    elif preview.exists():
        image = Image.open(preview).convert("RGB")
        bbox = meta.get("bbox_wgs84") or [0, 0, 1, 1]
        bounds = SceneBounds(
            lat_min=bbox[1],
            lat_max=bbox[3],
            lon_min=bbox[0],
            lon_max=bbox[2],
            width=image.size[0],
            height=image.size[1],
        )
    else:
        raise FileNotFoundError(f"No imagery for scene {scene_id}")

    catalog = load_catalog(wrecks) if wrecks.exists() else []
    return Scene(
        scene_id=scene_id,
        image=image,
        bounds=bounds,
        catalog=catalog,
        meta=meta,
        geotiff_path=tif if tif.exists() else None,
    )


def default_sample_dir() -> Path:
    here = Path(__file__).resolve().parents[2]
    return here / "data" / "samples"
