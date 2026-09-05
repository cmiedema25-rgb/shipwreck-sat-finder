# Rework Submission

## Title
Shipwreck Sat Finder

## Category
Computer Vision

## Skills
Computer Vision, Python, Multimodal AI (image+geo), Data Pipeline

## One-paragraph paste description
Shipwreck Sat Finder is a research demo that loads a real Copernicus Sentinel-2 coastal chip (Cape Henry approaches, cached for offline CI) and overlays public NOAA AWOIS/ENC wreck records with vessel names such as THOMAS F. POLLARD and WESTMORELAND. A classical CV detector proposes candidate pins; ship identity is guessed only by ranking nearby open catalog records (distance, optional size, record completeness) never by inventing famous ships. Mobile-first FastAPI PWA works on Android Chrome with Add to Home Screen; an Android WebView/Capacitor project is included for building a Play Store AAB (not claimed published). MIT licensed with make verify, CI, and honest metrics.

## Reviewer 60s table
| Step | Action | Expect |
| --- | --- | --- |
| 1 | `pip install -e .[dev]` | installs |
| 2 | `make verify` | tests + benchmark JSON |
| 3 | `make serve` then open phone | PWA UI |
| 4 | Detect and identify | pins + Likely vessel |
| 5 | Check evidence/benchmark-report.json | real metrics |
