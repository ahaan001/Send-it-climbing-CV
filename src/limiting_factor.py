"""
Tier 3: limiting-factor assessment.

For each move (the gap between two consecutive hand-dwell events, in
chronological order, for ONE climber's own clip -- no shared hold set
needed), compute three heuristic load proxies:

  - reach_pct:      distance to the next hold / the climber's arm span.
                     High => the move itself demands a lot of reach.
  - support_bend:   how bent the OTHER (supporting) arm stays while the
                     reaching hand travels. A straight arm is a cheap,
                     skeletal hang; a bent arm is expensive, muscled
                     isometric load. Low angle (very bent) => strength
                     demand.
  - leg_spread:     the widest the ankles get during the move, relative
                     to the climber's own leg length. A high ratio means
                     a wide stem/high step => flexibility demand.

Each move gets whichever of the three is most extreme RELATIVE TO THAT
CLIMBER'S OWN moves in the session (a percentile within their own data,
not an absolute clinical threshold -- we have no ground truth for what
counts as "hard" in absolute terms, only what's unusual for them).

This is a heuristic demo-stage proxy, not a validated biomechanical
model -- worth saying exactly that in the pitch if asked how rigorous it is.
"""

import numpy as np

from pose_pipeline import extract_trajectories, process_video, summarize_body


def wrist_dwell_events(trajectories, speed_thresh=14, min_dwell=4):
    """Like multi_climber.wrist_dwell_points, but keeps wrist identity
    and returns events merged & sorted chronologically across both
    wrists -- i.e. this IS the climber's own move sequence."""
    frame_idxs = sorted(trajectories.keys())
    events = []

    for wrist_name in ("LEFT_WRIST", "RIGHT_WRIST"):
        series = [(f, *trajectories[f][wrist_name]) for f in frame_idxs if wrist_name in trajectories[f]]
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
                    best = int(np.argmin(speeds[run_start:i]))
                    f, x, y, vis = seg[best]
                    events.append({"frame": f, "x": x, "y": y, "wrist": wrist_name})
                run_start = None

    events.sort(key=lambda e: e["frame"])
    return events


def angle_at_joint(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def analyze_moves(video_path):
    print(f"Analyzing moves in {video_path} ...")
    trajectories, w, h = extract_trajectories(video_path)
    segments, total, detected = process_video(video_path, out_path=None)
    body = summarize_body(segments)
    leg_len_px = body["thigh_px"] + body["shin_px"]

    events = wrist_dwell_events(trajectories)
    frame_idxs = sorted(trajectories.keys())

    moves = []
    for i in range(len(events) - 1):
        e0, e1 = events[i], events[i + 1]
        reach_pct = np.hypot(e1["x"] - e0["x"], e1["y"] - e0["y"]) / body["arm_span_px"]

        support_side = "RIGHT" if e1["wrist"] == "LEFT_WRIST" else "LEFT"
        s_sh, s_el, s_wr = f"{support_side}_SHOULDER", f"{support_side}_ELBOW", f"{support_side}_WRIST"

        window = [f for f in frame_idxs if e0["frame"] <= f <= e1["frame"]]
        elbow_angles = []
        ankle_spreads = []
        for f in window:
            p = trajectories[f]
            if s_sh in p and s_el in p and s_wr in p:
                if p[s_sh][2] > 0.5 and p[s_el][2] > 0.5 and p[s_wr][2] > 0.5:
                    elbow_angles.append(angle_at_joint(p[s_sh][:2], p[s_el][:2], p[s_wr][:2]))
            if "LEFT_ANKLE" in p and "RIGHT_ANKLE" in p:
                if p["LEFT_ANKLE"][2] > 0.5 and p["RIGHT_ANKLE"][2] > 0.5:
                    la, ra = p["LEFT_ANKLE"][:2], p["RIGHT_ANKLE"][:2]
                    ankle_spreads.append(np.hypot(la[0] - ra[0], la[1] - ra[1]))

        support_bend = float(np.mean(elbow_angles)) if elbow_angles else None  # lower = more bent = harder
        leg_spread_ratio = float(np.max(ankle_spreads)) / leg_len_px if ankle_spreads else None

        moves.append({
            "from": (e0["x"], e0["y"]), "to": (e1["x"], e1["y"]),
            "frame_range": (e0["frame"], e1["frame"]),
            "reach_pct": reach_pct,
            "support_bend_deg": support_bend,
            "leg_spread_ratio": leg_spread_ratio,
        })

    return body, moves


def classify_limiting_factors(moves):
    reach_vals = [m["reach_pct"] for m in moves]
    bend_vals = [m["support_bend_deg"] for m in moves if m["support_bend_deg"] is not None]
    spread_vals = [m["leg_spread_ratio"] for m in moves if m["leg_spread_ratio"] is not None]

    def pct_rank(val, arr):
        if val is None or not arr:
            return 0.0
        return float(np.mean(np.array(arr) <= val))

    for m in moves:
        reach_score = pct_rank(m["reach_pct"], reach_vals)
        # lower bend angle = more bent = harder, so invert before ranking
        strength_score = pct_rank(-m["support_bend_deg"] if m["support_bend_deg"] is not None else None,
                                   [-v for v in bend_vals])
        flex_score = pct_rank(m["leg_spread_ratio"], spread_vals)

        scores = {"reach": reach_score, "strength": strength_score, "flexibility": flex_score}
        m["scores"] = scores
        m["limiting_factor"] = max(scores, key=scores.get)

    counts = {"reach": 0, "strength": 0, "flexibility": 0}
    for m in moves:
        counts[m["limiting_factor"]] += 1
    overall = max(counts, key=counts.get)
    return moves, counts, overall


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/KatRDK.AVI"
    body, moves = analyze_moves(video_path)
    moves, counts, overall = classify_limiting_factors(moves)

    print(f"\n{len(moves)} moves detected:")
    for i, m in enumerate(moves):
        bend = f"{m['support_bend_deg']:.0f}deg" if m["support_bend_deg"] is not None else "n/a"
        spread = f"{m['leg_spread_ratio']:.2f}" if m["leg_spread_ratio"] is not None else "n/a"
        print(f"  move {i}: reach={m['reach_pct']:.1%}  support_arm_bend={bend}  "
              f"leg_spread={spread}  -> {m['limiting_factor'].upper()}")

    print(f"\nMove counts by limiting factor: {counts}")
    print(f"Overall limiting factor this session: {overall.upper()}")
    labels = {
        "reach": "your limiting factor this session is reach -- several moves needed a large "
                 "fraction of your wingspan.",
        "strength": "your limiting factor this session is strength -- you spent a lot of time "
                    "on bent-arm holds instead of straight-arm hangs.",
        "flexibility": "your limiting factor this session is flexibility -- your hardest moves "
                       "involved wide stems/high steps relative to your leg length.",
    }
    print(f"\nDemo takeaway line: \"{labels[overall]}\"")
