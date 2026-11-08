# Capture and front/back pairing

The main difficulty is not the CV but getting users to take photos the pipeline can use,
so that every front is matched to the right back. Principle: **make the setup itself
structured instead of explaining rules better.**

## Current setup (personal pipeline)

- Coins in rows on plain paper; photo `..._v.jpg`; flip every coin; photo `..._r.jpg`.
- `main.py` detects coins (Canny + contours), assigns rows and pairs by horizontal order.
- Weak points: row assignment fails on uneven rows; mirrored order if the user flips
  the sheet instead of the coins; no scale, so coins of the same design but different
  size (25 vs 50 Schilling) look identical.

## 1. Printable capture sheet

A4 and US Letter PDFs (later: sold as printed sheets, see [05-monetization.md](05-monetization.md)).

- **Numbered circles** (slots) for 20–30 coins, maybe 2–3 slot sizes per sheet.
- **ArUco markers** in the corners (OpenCV `cv2.aruco`), each with a different ID.

What this gives us:

| Feature | How |
|---|---|
| Pairing | Slot number decides which front belongs to which back. No row heuristic. |
| Perspective correction | Homography from the 4 markers. |
| Scale | Known paper size → mm per pixel → **coin diameter in mm**. Strong identification feature; pass it to the LLM and use it to rank Numista candidates (`size` field). |
| Orientation | Markers are asymmetric → detect if the whole sheet was turned or flipped. |
| Colour | Small grey/white patch → white balance → helps silver vs nickel vs brass. |

Tasks:

- [ ] Generate the sheet PDF (Python: reportlab or matplotlib), slots + markers + short instructions printed on it.
- [ ] `detect_sheet(image) -> homography, orientation, mm_per_px`.
- [ ] Slot assignment: coin centre → nearest slot; warn on empty / double slots.
- [ ] Diameter in mm per coin, written to `extraction_results.json`.
- [ ] Add diameter to the LLM prompt and to Numista candidate ranking.

## 2. Plain paper fallback

Keep the current method working (no printer needed), with extra checks:

- Diameter in pixels must match between front and back of a pair (within a few %).
- The LLM sees both sides together and reports `front_is_obverse`; add a field
  "do these two sides belong to the same coin?" and flag mismatches.
- Detect mirrored order (sheet flipped): try both pairings, keep the one whose diameters match better.

## 3. Guided two-photo flow (browser camera, later native)

1. Photo 1. The app shows the detected coins with numbers.
2. "Flip each coin where it lies." While taking photo 2, faded outlines of the photo 1
   detections are shown so the user lines up the same positions.
3. The app refuses to continue if coin count or positions don't match, and shows which coin is off.

## 4. Instant quality checks (before upload)

| Check | Method | Message |
|---|---|---|
| Blur | Variance of Laplacian | "Hold still / tap to focus" |
| Glare | Saturated pixel blobs on coins | "Turn off flash, avoid direct light" |
| Coins touching | Contours merged / non-circular | "Leave a finger's width between coins" |
| Too small | < ~250 px coin diameter | "Move closer or use fewer coins per photo" |
| Shadows | Dark halo around coins | "Use daylight, not a lamp from the side" |
| Tilt | Marker homography far from frontal | "Hold the phone flat above the sheet" |

## 5. Instructions for users

One card plus a 20-second video / GIF:

1. Plain matte paper (or the capture sheet).
2. Daylight, no flash.
3. Phone flat above the sheet.
4. A finger's width between coins.
5. Flip each coin where it lies; don't move the sheet.

## Later

- Single-coin quick-scan mode (competitor parity).
- Video mode: flip coins in front of the camera, frames picked automatically.
