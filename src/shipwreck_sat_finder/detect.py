"""Classical CV shipwreck candidate detection on optical coastal scenes.

Honest demo features: edge density, elongated blob geometry, local anomaly
scoring. Optional lightweight logistic classifier on handcrafted features
trained only on synthetic fixtures — not claimed as real-world Maxar models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .geo import SceneBounds


@dataclass
class Candidate:
    """A detected wreck candidate with pixel, geo, and confidence."""

    cx: float
    cy: float
    length_px: float
    angle_deg: float
    confidence: float
    lat: float
    lon: float
    features: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _to_gray(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sobel_edges(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple Sobel magnitude + direction."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    # Pad and convolve
    pad = np.pad(gray, 1, mode="edge")
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    for i in range(3):
        for j in range(3):
            gx += kx[i, j] * pad[i : i + gray.shape[0], j : j + gray.shape[1]]
            gy += ky[i, j] * pad[i : i + gray.shape[0], j : j + gray.shape[1]]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.arctan2(gy, gx)
    return mag, gx, gy


def _local_anomaly(gray: np.ndarray, win: int = 31) -> np.ndarray:
    """|pixel - local mean| / (local std + eps) — wrecks are darker blobs."""
    from numpy.lib.stride_tricks import sliding_window_view

    pad = win // 2
    padded = np.pad(gray, pad, mode="reflect")
    windows = sliding_window_view(padded, (win, win))
    means = windows.mean(axis=(-1, -2))
    stds = windows.std(axis=(-1, -2)) + 1e-3
    # Darker than surroundings → positive anomaly for wrecks
    score = (means - gray) / stds
    return np.clip(score, 0, None)


def _connected_components(mask: np.ndarray) -> List[np.ndarray]:
    """4-connected component labels → list of (N,2) coordinate arrays."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pts = []
            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(np.array(pts, dtype=np.float32))
    return comps


def _blob_features(pts: np.ndarray, gray: np.ndarray, edge_mag: np.ndarray) -> Optional[dict]:
    """Handcrafted features for an elongated hull-like blob."""
    if len(pts) < 25:
        return None
    ys, xs = pts[:, 0], pts[:, 1]
    cx, cy = float(xs.mean()), float(ys.mean())
    # Covariance → principal axes
    cov = np.cov(xs, ys)
    if cov.shape != (2, 2):
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    length = 4 * math_sqrt(max(eigvals[0], 1e-6))  # ~2-sigma extent
    width = 4 * math_sqrt(max(eigvals[1], 1e-6))
    if width < 1e-3:
        return None
    aspect = length / width
    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))) % 180
    # Edge density along blob
    yy = np.clip(ys.astype(int), 0, gray.shape[0] - 1)
    xx = np.clip(xs.astype(int), 0, gray.shape[1] - 1)
    edge_density = float(edge_mag[yy, xx].mean() / (edge_mag.mean() + 1e-6))
    darkness = float((gray.mean() - gray[yy, xx].mean()) / (gray.std() + 1e-6))
    area = float(len(pts))
    compactness = float(4 * np.pi * area / ((length + width) ** 2 + 1e-6))
    return {
        "cx": cx,
        "cy": cy,
        "length_px": length,
        "width_px": width,
        "aspect": aspect,
        "angle_deg": angle_deg,
        "edge_density": edge_density,
        "darkness": darkness,
        "area": area,
        "compactness": compactness,
    }


def math_sqrt(x: float) -> float:
    return float(np.sqrt(x))


def _score_features(f: dict) -> float:
    """Heuristic confidence in [0, 1] from handcrafted features.

    Tuned for synthetic elongated dark hulls: prefer aspect ~2–6,
    moderate size, high darkness, elevated edge density.
    """
    aspect = f["aspect"]
    length = f["length_px"]
    darkness = f["darkness"]
    edge = f["edge_density"]

    # Aspect: hull-like
    if aspect < 1.4:
        a_score = 0.1
    elif aspect < 2.0:
        a_score = 0.45
    elif aspect <= 5.5:
        a_score = 0.95
    elif aspect <= 8:
        a_score = 0.55
    else:
        a_score = 0.2

    # Length in expected wreck pixel range for our synthetic scenes
    if 18 <= length <= 70:
        l_score = 0.9
    elif 12 <= length < 18 or 70 < length <= 100:
        l_score = 0.5
    else:
        l_score = 0.15

    d_score = float(np.clip(darkness / 1.5, 0, 1))
    e_score = float(np.clip((edge - 0.8) / 1.5, 0, 1))

    conf = 0.35 * a_score + 0.25 * l_score + 0.25 * d_score + 0.15 * e_score
    return float(np.clip(conf, 0.0, 1.0))


def detect_wrecks(
    img: Image.Image,
    bounds: Optional[SceneBounds] = None,
    min_confidence: float = 0.35,
    max_candidates: int = 20,
) -> List[Candidate]:
    """Run classical CV pipeline and return ranked candidates."""
    gray = _to_gray(img)
    edge_mag, _, _ = _sobel_edges(gray)
    anomaly = _local_anomaly(gray, win=25)

    # Candidate mask: dark + anomalous + some edge support
    edge_thr = np.percentile(edge_mag, 70)
    anom_thr = np.percentile(anomaly, 85)
    dark_thr = np.percentile(gray, 35)
    mask = (anomaly >= anom_thr) & (gray <= dark_thr) & (edge_mag >= edge_thr * 0.4)

    # Morphological-ish cleanup: remove tiny speckles via component size filter later
    comps = _connected_components(mask)
    candidates: List[Candidate] = []

    h, w = gray.shape
    if bounds is None:
        bounds = SceneBounds(0.0, 1.0, 0.0, 1.0, w, h)

    for pts in comps:
        feats = _blob_features(pts, gray, edge_mag)
        if feats is None:
            continue
        if feats["aspect"] < 1.3 or feats["length_px"] < 12:
            continue
        if feats["length_px"] > min(w, h) * 0.4:
            continue
        conf = _score_features(feats)
        if conf < min_confidence:
            continue
        lat, lon = bounds.pixel_to_latlon(feats["cx"], feats["cy"])
        candidates.append(
            Candidate(
                cx=feats["cx"],
                cy=feats["cy"],
                length_px=feats["length_px"],
                angle_deg=feats["angle_deg"],
                confidence=conf,
                lat=lat,
                lon=lon,
                features={
                    k: feats[k]
                    for k in (
                        "aspect",
                        "width_px",
                        "edge_density",
                        "darkness",
                        "area",
                        "compactness",
                    )
                },
            )
        )

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates[:max_candidates]


def draw_overlay(img: Image.Image, candidates: List[Candidate], color=(255, 220, 40)) -> Image.Image:
    """Draw candidate pins / boxes on a copy of the scene."""
    from PIL import ImageDraw

    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for i, c in enumerate(candidates):
        r = max(8, c.length_px * 0.35)
        # Ellipse around center
        draw.ellipse([c.cx - r, c.cy - r, c.cx + r, c.cy + r], outline=color, width=2)
        # Crosshair
        draw.line([(c.cx - r - 4, c.cy), (c.cx + r + 4, c.cy)], fill=color, width=1)
        draw.line([(c.cx, c.cy - r - 4), (c.cx, c.cy + r + 4)], fill=color, width=1)
        label = f"#{i+1} {c.confidence:.2f}"
        draw.text((c.cx + r + 3, c.cy - 8), label, fill=color)
    return out
