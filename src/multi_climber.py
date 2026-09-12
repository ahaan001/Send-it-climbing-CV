"""
Cross-climber pipeline for a normal (non-lit) gym wall, using multiple
climbers' attempts at the SAME route filmed from the SAME camera angle.

Key idea: we don't need to detect holds by color/shape at all. A hold is,
by definition, wherever climbers' hands stop and grip. So:

  1. For each climber, track wrist trajectories and find "dwell" frames
     where wrist speed drops (a grip, not a hand in transit).
  2. Pool dwell points across ALL climbers and cluster by proximity.
     A cluster that shows up across MULTIPLE climbers is a real hold;
     one-off dwells (adjusting a foot, brushing chalk, pausing mid-air)
     don't recur at the same pixel spot for other climbers and get
     filtered out.
  3. That gives a shared hold set for the route, in this shared camera's
     pixel space -- which is exactly what's needed to compare climbers'
     reach ratios on truly the same moves, and to build a ghost overlay
     using one climber's pose at a hold as the reference for another.
"""

import numpy as np

from pose_pipeline import extract_trajectories, process_video, summarize_body


def wrist_dwell_points(trajectories, speed_thresh=14, min_dwell=4):
    """Frame-to-frame wrist speed; a dwell = several consecutive frames
    below speed_thresh. Returns list of (x, y, frame_idx) touch points,
    one per dwell segment, taken at the segment's slowest frame."""
    frame_idxs = sorted(trajectories.keys())
    points = []

    for wrist_name in ("LEFT_WRIST", "RIGHT_WRIST"):
        series = []
        for f in frame_idxs:
            pts = trajectories[f]
            if wrist_name in pts:
                x, y, vis = pts[wrist_name]
                series.append((f, x, y, vis))

        speeds = [0.0]
        for i in range(1, len(series)):
            _, x0, y0, _ = series[i - 1]
            _, x1, y1, _ = series[i]
            speeds.append(np.hypot(x1 - x0, y1 - y0))

        run_start = None
        for i, s in enumerate(speeds):
            slow = s < speed_thresh and series[i][3] > 0.5
            if slow:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None and i - run_start >= min_dwell:
                    seg = series[run_start:i]
                    slow_speeds = speeds[run_start:i]
                    best = int(np.argmin(slow_speeds))
                    f, x, y, vis = seg[best]
                    points.append((x, y, f))
                run_start = None
        if run_start is not None and len(series) - run_start >= min_dwell:
            seg = series[run_start:]
            f, x, y, vis = seg[0]
            points.append((x, y, f))

    return points


def discover_shared_holds(climber_dwells, merge_dist=28, min_climbers=2):
    """climber_dwells: {climber_name: [(x, y, frame_idx), ...]}.
    Clusters dwell points across climbers; keeps clusters touched by
    at least min_climbers distinct climbers as real shared holds."""
    all_points = []
    for name, pts in climber_dwells.items():
        for (x, y, f) in pts:
            all_points.append({"x": x, "y": y, "climber": name, "frame": f})

    clusters = []
    for p in all_points:
        placed = False
        for cl in clusters:
            mx = np.mean([q["x"] for q in cl])
            my = np.mean([q["y"] for q in cl])
            if np.hypot(p["x"] - mx, p["y"] - my) < merge_dist:
                cl.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])

    holds = []
    for cl in clusters:
        climbers_here = {q["climber"] for q in cl}
        if len(climbers_here) >= min_climbers:
            holds.append({
                "x": float(np.mean([q["x"] for q in cl])),
                "y": float(np.mean([q["y"] for q in cl])),
                "climbers": sorted(climbers_here),
                "n_climbers": len(climbers_here),
                "per_climber_frame": {q["climber"]: q["frame"] for q in cl},
            })
    return holds


if __name__ == "__main__":
    import json

    climbers = {
        "Aiden": "CruxCam/posevids/AidenRDK.AVI",
        "Malachi": "CruxCam/posevids/MalachiRDK.AVI",
        "Jasper": "CruxCam/posevids/JasperRDK.AVI",
        "Kaia": "CruxCam/posevids/KaiaRDK.AVI",
        "Kat": "CruxCam/posevids/KatRDK.AVI",
        "Mike": "CruxCam/posevids/MikeRDK.AVI",
    }

    climber_dwells = {}
    climber_bodies = {}
    for name, path in climbers.items():
        print(f"Processing {name} ({path}) ...")
        traj, w, h = extract_trajectories(path)
        climber_dwells[name] = wrist_dwell_points(traj)
        segments, total, detected = process_video(path, out_path=None)
        climber_bodies[name] = summarize_body(segments)
        print(f"  {detected}/{total} frames with pose, "
              f"{len(climber_dwells[name])} dwell points, "
              f"arm_span_px={climber_bodies[name]['arm_span_px']:.0f}")

    holds = discover_shared_holds(climber_dwells)
    holds.sort(key=lambda h: -h["y"])

    print(f"\n{len(holds)} shared holds found (touched by >=2 climbers):")
    for i, h in enumerate(holds):
        print(f"  #{i}: ({h['x']:.0f},{h['y']:.0f})  touched by {h['n_climbers']}/{len(climbers)}: "
              f"{', '.join(h['climbers'])}")

    print("\nArm span estimates (pixel units, shared camera frame):")
    for name, body in climber_bodies.items():
        print(f"  {name}: {body['arm_span_px']:.0f}px")

    with open("multi_climber_data.json", "w") as f:
        json.dump({
            "holds": holds,
            "bodies": climber_bodies,
        }, f, indent=2)
    print("\nSaved shared holds + body data to multi_climber_data.json")
