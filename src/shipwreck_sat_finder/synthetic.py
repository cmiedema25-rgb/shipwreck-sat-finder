"""Generate synthetic coastal satellite-like scenes with known wreck markers."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .geo import SceneBounds


@dataclass
class WreckLabel:
    """Ground-truth wreck annotation in pixel + geo space."""

    id: str
    cx: float
    cy: float
    length_px: float
    angle_deg: float
    lat: float
    lon: float


@dataclass
class SceneMeta:
    scene_id: str
    width: int
    height: int
    bounds: dict
    wrecks: List[dict]
    seed: int


def _ocean_base(w: int, h: int, rng: random.Random) -> np.ndarray:
    """Blue-green ocean with gentle noise and depth gradient."""
    yy, xx = np.mgrid[0:h, 0:w]
    depth = 0.35 + 0.25 * (yy / h) + 0.1 * np.sin(xx / 40.0)
    r = (20 + 15 * depth + rng.uniform(-3, 3)).astype(np.float32)
    g = (80 + 40 * depth + rng.uniform(-5, 5)).astype(np.float32)
    b = (140 + 50 * (1 - depth) + rng.uniform(-8, 8)).astype(np.float32)
    noise = rng.gauss(0, 1)
    # Per-pixel noise
    n = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, 6, (h, w, 3))
    rgb = np.stack([r, g, b], axis=-1) + n
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _draw_coastline(img: Image.Image, rng: random.Random) -> None:
    """Draw a sandy/rocky coastline along one edge."""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    side = rng.choice(["left", "right", "top", "bottom"])
    sand = (194, 178, 128)
    rock = (110, 100, 90)
    if side in ("left", "right"):
        x0 = 0 if side == "left" else w - 1
        pts = []
        for y in range(0, h + 20, 12):
            offset = int(40 + 35 * math.sin(y / 55.0) + rng.uniform(-12, 12))
            x = x0 + offset if side == "left" else x0 - offset
            pts.append((x, y))
        if side == "left":
            poly = [(0, 0)] + pts + [(0, h)]
        else:
            poly = [(w, 0)] + pts + [(w, h)]
        draw.polygon(poly, fill=sand)
        # Rocks along edge
        for _ in range(rng.randint(8, 18)):
            rx = rng.randint(5, 80) if side == "left" else w - rng.randint(5, 80)
            ry = rng.randint(0, h)
            rr = rng.randint(4, 14)
            draw.ellipse([rx - rr, ry - rr, rx + rr, ry + rr], fill=rock)
    else:
        y0 = 0 if side == "top" else h - 1
        pts = []
        for x in range(0, w + 20, 12):
            offset = int(35 + 30 * math.sin(x / 50.0) + rng.uniform(-10, 10))
            y = y0 + offset if side == "top" else y0 - offset
            pts.append((x, y))
        if side == "top":
            poly = [(0, 0)] + pts + [(w, 0)]
        else:
            poly = [(0, h)] + pts + [(w, h)]
        draw.polygon(poly, fill=sand)


def _draw_wreck(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    length: float,
    angle_deg: float,
    rng: random.Random,
) -> None:
    """Draw an elongated dark hull-like blob with optional debris."""
    angle = math.radians(angle_deg)
    half = length / 2
    width = length * rng.uniform(0.18, 0.28)
    # Hull polygon (rotated rectangle tapered)
    corners = [
        (-half, -width / 2),
        (half * 0.85, -width / 2.5),
        (half, 0),
        (half * 0.85, width / 2.5),
        (-half, width / 2),
    ]
    pts = []
    for px, py in corners:
        rx = px * math.cos(angle) - py * math.sin(angle) + cx
        ry = px * math.sin(angle) + py * math.cos(angle) + cy
        pts.append((rx, ry))
    hull_color = (45 + rng.randint(0, 25), 50 + rng.randint(0, 20), 55 + rng.randint(0, 20))
    draw.polygon(pts, fill=hull_color)
    # Shadow / edge darkening
    draw.line([pts[0], pts[1], pts[2], pts[3], pts[4], pts[0]], fill=(25, 28, 30), width=1)
    # Debris nearby
    for _ in range(rng.randint(1, 4)):
        dx = rng.uniform(-length * 0.6, length * 0.6)
        dy = rng.uniform(-length * 0.4, length * 0.4)
        rr = rng.uniform(2, 5)
        draw.ellipse(
            [cx + dx - rr, cy + dy - rr, cx + dx + rr, cy + dy + rr],
            fill=(60, 55, 50),
        )


def generate_scene(
    scene_id: str,
    width: int = 512,
    height: int = 512,
    n_wrecks: Optional[int] = None,
    seed: Optional[int] = None,
    lat_center: float = 36.8,
    lon_center: float = -122.4,
    span_deg: float = 0.08,
) -> Tuple[Image.Image, SceneMeta, SceneBounds]:
    """Create a synthetic coastal optical scene with labeled wrecks."""
    seed = seed if seed is not None else abs(hash(scene_id)) % (2**31)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    # Re-seed python random via numpy bridge for ocean noise consistency
    random.seed(seed)

    base = _ocean_base(width, height, rng)
    img = Image.fromarray(base, mode="RGB")
    _draw_coastline(img, rng)

    # Soft blur then slight sharpen for "satellite" look
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    bounds = SceneBounds(
        lat_min=lat_center - span_deg / 2,
        lat_max=lat_center + span_deg / 2,
        lon_min=lon_center - span_deg / 2,
        lon_max=lon_center + span_deg / 2,
        width=width,
        height=height,
    )

    n = n_wrecks if n_wrecks is not None else rng.randint(1, 4)
    draw = ImageDraw.Draw(img)
    wrecks: List[WreckLabel] = []
    margin = 60
    for i in range(n):
        cx = rng.uniform(margin, width - margin)
        cy = rng.uniform(margin, height - margin)
        length = rng.uniform(28, 55)
        angle = rng.uniform(0, 180)
        # Avoid putting wrecks on obvious land (rough: skip near edges if sand-like)
        _draw_wreck(draw, cx, cy, length, angle, rng)
        lat, lon = bounds.pixel_to_latlon(cx, cy)
        wrecks.append(
            WreckLabel(
                id=f"{scene_id}-w{i}",
                cx=cx,
                cy=cy,
                length_px=length,
                angle_deg=angle,
                lat=lat,
                lon=lon,
            )
        )

    # Wave / wake-like streaks (clutter)
    for _ in range(rng.randint(2, 6)):
        x0, y0 = rng.randint(0, width), rng.randint(0, height)
        length = rng.randint(20, 60)
        ang = rng.uniform(0, math.pi)
        x1 = x0 + length * math.cos(ang)
        y1 = y0 + length * math.sin(ang)
        wake = (int(90 + rng.randint(0, 40)), int(130 + rng.randint(0, 40)), int(160 + rng.randint(0, 30)))
        draw.line([(x0, y0), (x1, y1)], fill=wake, width=rng.randint(1, 2))

    meta = SceneMeta(
        scene_id=scene_id,
        width=width,
        height=height,
        bounds={
            "lat_min": bounds.lat_min,
            "lat_max": bounds.lat_max,
            "lon_min": bounds.lon_min,
            "lon_max": bounds.lon_max,
        },
        wrecks=[asdict(w) for w in wrecks],
        seed=seed,
    )
    return img, meta, bounds


def write_scene(out_dir: Path, scene_id: str, **kwargs) -> Path:
    """Generate and write PNG + JSON label sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img, meta, _ = generate_scene(scene_id, **kwargs)
    png_path = out_dir / f"{scene_id}.png"
    json_path = out_dir / f"{scene_id}.json"
    img.save(png_path)
    json_path.write_text(json.dumps(asdict(meta) if hasattr(meta, "__dataclass_fields__") else meta.__dict__, indent=2))
    # Fix: SceneMeta is dataclass — use asdict via converting
    from dataclasses import asdict as _asdict

    json_path.write_text(json.dumps(_asdict(meta), indent=2))
    return png_path
