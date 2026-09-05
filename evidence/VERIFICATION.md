# Verification

## 60-second reviewer path
```bash
pip install -e ".[dev]"
make verify
make serve   # http://0.0.0.0:7860
```

Offline CI uses `data/samples/cape_henry_approaches_*` (Sentinel-2 RGB GeoTIFF + AWOIS GeoJSON).

Refresh samples (network):
```bash
python scripts/fetch_real_scene.py
```

## Honesty
Optical Sentinel-2 often cannot resolve submerged wrecks; catalog pins are the primary proof. CV recall may be low - report actual numbers.
