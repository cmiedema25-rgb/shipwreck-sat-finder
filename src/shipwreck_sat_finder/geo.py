"""Simple affine pixel ↔ lat/lon mapping for synthetic scene bounds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SceneBounds:
    """Geographic bounds of a synthetic scene (north-up, lon west→east)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    width: int
    height: int

    def pixel_to_latlon(self, x: float, y: float) -> Tuple[float, float]:
        """Map pixel (x, y) to (lat, lon). y=0 is north (top of image)."""
        lon = self.lon_min + (x / max(self.width - 1, 1)) * (self.lon_max - self.lon_min)
        lat = self.lat_max - (y / max(self.height - 1, 1)) * (self.lat_max - self.lat_min)
        return lat, lon

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """Map (lat, lon) to pixel (x, y)."""
        x = (lon - self.lon_min) / max(self.lon_max - self.lon_min, 1e-12) * (self.width - 1)
        y = (self.lat_max - lat) / max(self.lat_max - self.lat_min, 1e-12) * (self.height - 1)
        return x, y
