# Shipwreck Sat Finder

Research/demo app that uses **real Copernicus Sentinel-2** imagery and **public NOAA AWOIS/ENC wreck catalogs** to pin wreck locations and **guess vessel identity** via catalog matching.

> Not a guarantee of real-world wreck discovery. Real deployment needs licensed imagery and validated models. **Not for navigation.**

## What it does
1. Loads a georeferenced Sentinel-2 RGB chip (cached sample: Cape Henry approaches).
2. Overlays open NOAA wreck records (e.g. THOMAS F. POLLARD, WESTMORELAND, E H BLUM).
3. Runs classical CV candidate detection (edge density, elongated blobs, anomaly scoring).
4. For each pin, ranks nearby catalog vessels by distance / optional size / completeness.
5. Serves a **mobile-first PWA** (Android Chrome Install/Add to Home Screen) via FastAPI.
6. Includes an **Android WebView / Capacitor** project to build a Play Store **AAB** (not claimed published).

Ship ID is **catalog proximity ranking**, not magical visual recognition of every wreck worldwide. Unknown if no catalog match.

## Quick start
```bash
pip install -e ".[dev]"
make verify
make serve   # http://0.0.0.0:7860
```

CLI:
```bash
python -m shipwreck_sat_finder detect
python -m shipwreck_sat_finder benchmark --out evidence/benchmark-report.json
```

## Data (open / legal)
| Asset | Source |
| --- | --- |
| Imagery | Sentinel-2 L2A via Microsoft Planetary Computer STAC |
| Wrecks | NOAA AWOIS / ENC public GIS synthesis (vesselTerm, history, positions) |
| Cache | `data/samples/cape_henry_approaches_*` for offline CI |

Refresh: `python scripts/fetch_real_scene.py`

Synthetic Pillow scenes exist only as optional unit-test helpers (`synthetic.py`).

## Run on Android
1. Start server on your PC: `make serve`
2. On phone Chrome open `http://<lan-ip>:7860` (same Wi-Fi).
3. Menu → **Install app** / Add to Home Screen (PWA).
4. Optional: `adb reverse tcp:7860 tcp:7860` then `http://127.0.0.1:7860`.

### Play Store path
See `docs/PLAY_STORE.md` and `android/`. Application id: `com.cmiedema25.shipwrecksatfinder`.
Build AAB on a machine with Android Studio (`./gradlew bundleRelease`). **This repo does not claim the app is on Google Play.**

## Honesty / legality
- No Google Earth / Maxar scraping.
- Metrics on a small AOI may show limited CV recall for submerged wrecks — catalog pins are primary proof.
- Identification never invents names absent from the open AOI catalog.

## License
MIT. Sentinel-2: Copernicus open access. NOAA catalog: U.S. public data (non-navigational use).
