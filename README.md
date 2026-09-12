# Personalized Climbing Difficulty (HackCMU 2026)

Climbing grades are set by whoever established the route, using their own
body. A move that's easy for a tall climber can be nearly impossible and even 
dangerous for a shorter one, but no current tool quantifies that gap. This 
project uses phone-camera computer vision to compute a **personalized,
reach-adjusted difficulty score** for each move, alongside the gym's official grade.

Track: **Multiplayer**

## What it does

Given a video of someone climbing, the pipeline:

1. **Estimates body proportions** (arm span, torso, leg segments) from
   MediaPipe pose landmarks, robust to camera-perspective foreshortening
   (`src/pose_pipeline.py`).
2. **Detects the route's holds** two ways: color/LED detection for lit
   boards like Kilter (`src/hold_detection.py`), or hand-dwell clustering
   for any normal wall — a hold is wherever climbers' hands actually stop
   and grip, no color assumptions needed (`src/multi_climber.py`).
3. **Computes reach-adjusted difficulty**: each move's distance as a % of
   *that climber's own* arm span — the core personalization metric.
4. **Recovers real climb order** from wrist-dwell contact events instead
   of assuming holds are climbed bottom-to-top (`src/climb_order.py`).
5. **Estimates wall overhang** from the board's trapezoidal perspective
   distortion (`src/wall_angle.py`).
6. **Assembles Strava-style session stats**: distance climbed, exertion
   index, duration (`src/session_stats.py`).
7. **Cross-climber ghost overlay**: when multiple climbers are filmed from
   the same fixed camera, overlays one climber's real pose (rescaled to
   another climber's body) at a shared hold — a real cross-body form
   comparison, not a synthetic "ideal form" (`src/ghost_overlay.py`).
8. **Limiting-factor assessment**: classifies each move as reach-,
   strength-, or flexibility-limited from support-arm bend angle and leg
   spread, and surfaces the session's overall limiting factor
   (`src/limiting_factor.py`).

Everything above is wired into one entry point:

```bash
python3 src/run_demo.py path/to/climb.mp4 --out-dir demo_output
```

which produces a skeleton-overlay video, route/move visualization, wall
angle diagram, limiting-factor breakdown, and a `report.json` /
`report.txt` summary.

## Demo assets

`demo_assets/` has example output from a real test clip:

- `skeleton_overlay.mp4` — pose tracked through a full climb
- `route_profile.png` — detected holds with reach-adjusted % per move,
  crux move flagged
- `detected_holds.png` — LED hold detection on a Kilter board
- `wall_angle.png` — overhang measurement
- `climb_order.png` — real hand-contact climb order vs. naive height sort
- `limiting_factor.png` — per-move limiting-factor classification
- `ghost_overlay.png` — cross-climber pose comparison at a shared hold

## Setup

```bash
pip install -r requirements.txt
```

Note: this project uses an older version of a face/body-tracking library 
(MediaPipe) on purpose. The old version works instantly, no internet needed o
nce installed. Newer versions changed how they work; you need to download extra 
files from Google's servers the first time you run the code.


## Known limitations 

- **No camera-shake compensation.** Hold detection assumes holds stay in
  the same pixel position across sampled frames. A handheld/shaky camera
  breaks that assumption. Frame-to-frame stabilization (feature-matching
  + homography alignment) would fix this but isn't implemented yet.
- **Light-colored walls**: hold/wall detection that relies on finding a
  dark board silhouette doesn't work on light gym walls — `wall_angle.py`
  now detects this case and returns `None` instead of a random number, but
  doesn't yet have a working fallback for that wall type.
- **Reach-ratio comparisons across different videos** assume the climbers
  were filmed at a comparable distance from the wall. Within one climber's
  own footage this doesn't matter (the pixel-to-real-world scale cancels
  out); across different recording setups it would need a shared
  calibration reference.
- Body measurements are reported primarily in **arm-span-relative units**
  (calibration-free). Converting to real meters currently uses a
  population-average arm span assumption unless a real measurement is
  supplied.

## Credits

Early pipeline testing used publicly available climbing pose-estimation
footage from the [CruxCam](https://github.com/scobi7/CruxCam) project on
GitHub for prototyping before running on our own footage.
