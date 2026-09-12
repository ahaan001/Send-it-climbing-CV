"""Observed beta: recover the climber's actual hand sequence from pose.

For every frame and each hand, the hand is "on" the nearest hold if it is
within a contact radius (a fraction of the climber's own arm span). A run
of >= min_frames consecutive frames on the same hold is a contact event.
Events from both hands are merged chronologically into placements.
"""
from __future__ import annotations

import numpy as np

from .pose import hand_point, foot_point


def contact_events(pose: dict, holds: list, arm_span_px: float,
                   contact_radius_frac: float = 0.12, min_frames: int = 4, min_vis: float = 0.5,
                   gap_frames: int = 3, point_fn=hand_point, limb_kind: str = "hand"):
    """Contact events for both limbs of one kind. limb_kind='hand' uses the
    hand centre against hand-usable holds; 'foot' uses the toe/ankle against
    every on-route hold (foot-only ones included) and labels limbs LEFT_FOOT/RIGHT_FOOT."""
    radius = contact_radius_frac * arm_span_px
    if limb_kind == "foot":
        route = [h for h in holds if h.get("on_route", True)]
    else:
        route = [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]
    if not route:
        return []
    hx = np.array([h["x"] for h in route])
    hy = np.array([h["y"] for h in route])
    hid = [h["id"] for h in route]
    idxs = sorted(pose["frames"].keys())
    events = []
    for side in ("LEFT", "RIGHT"):
        label = side if limb_kind == "hand" else f"{side}_FOOT"
        cur = None  # [hold_id, start, end, count]
        last_f = None
        for f in idxs:
            hp = point_fn(pose["frames"][f], side, min_vis)
            on = None
            if hp is not None:
                d = np.hypot(hx - hp[0], hy - hp[1])
                k = int(np.argmin(d))
                if d[k] <= radius:
                    on = hid[k]
            if cur is not None and (on != cur[0] or (last_f is not None and f - last_f > gap_frames)):
                if cur[3] >= min_frames:
                    events.append({"hand": label, "limb": label, "hold_id": cur[0], "start": cur[1], "end": cur[2], "frames": cur[3]})
                cur = None
            if on is not None:
                if cur is None:
                    cur = [on, f, f, 1]
                else:
                    cur[2] = f
                    cur[3] += 1
            last_f = f
        if cur is not None and cur[3] >= min_frames:
            events.append({"hand": label, "limb": label, "hold_id": cur[0], "start": cur[1], "end": cur[2], "frames": cur[3]})
    events.sort(key=lambda e: e["start"])
    return events


def observed_feet(pose: dict, holds: list, arm_span_px: float, contact_radius_frac: float = 0.12,
                  min_frames: int = 4, min_vis: float = 0.5):
    """Foot contact events measured from toe/ankle landmarks (LEFT_FOOT/RIGHT_FOOT)."""
    return contact_events(pose, holds, arm_span_px, contact_radius_frac, min_frames, min_vis,
                          point_fn=foot_point, limb_kind="foot")


def feet_at_placements(foot_events: list, placements: list):
    """For each hand placement frame, which hold (if any) each foot was on:
    [{"LEFT_FOOT": id|None, "RIGHT_FOOT": id|None}], aligned with placements.
    None = not on a detected hold (smearing, on the mat, or not visible)."""
    out = []
    for p in placements:
        f = p["frame"]
        feet = {"LEFT_FOOT": None, "RIGHT_FOOT": None}
        for e in foot_events:
            if e["start"] <= f <= e["end"] + 2:
                feet[e["limb"]] = e["hold_id"]
        out.append(feet)
    return out


def observed_placements(events: list, flicker_frames: int = 12):
    """Chronological list of hand placements: [{hand, hold_id, frame}].
    Cleans two kinds of tracking noise: a hand re-registering on the hold
    it already held, and a brief A -> B -> A flicker (B held for fewer
    than `flicker_frames` frames before the hand is back on A)."""
    per_hand = {"LEFT": [], "RIGHT": []}
    for e in events:
        per_hand[e["hand"]].append(dict(e))
    kept = []
    for hand, evs in per_hand.items():
        changed = True
        while changed:
            changed = False
            for i in range(1, len(evs) - 1):
                a, b, c = evs[i - 1], evs[i], evs[i + 1]
                if a["hold_id"] == c["hold_id"] != b["hold_id"] and b["frames"] < flicker_frames:
                    a["end"] = c["end"]
                    a["frames"] += c["frames"]
                    del evs[i:i + 2]
                    changed = True
                    break
        kept.extend(evs)
    kept.sort(key=lambda e: e["start"])
    placements = []
    last_on = {"LEFT": None, "RIGHT": None}
    for e in kept:
        if last_on[e["hand"]] == e["hold_id"]:
            continue
        placements.append({"hand": e["hand"], "hold_id": e["hold_id"], "frame": e["start"], "end": e["end"]})
        last_on[e["hand"]] = e["hold_id"]
    return placements


def initial_state(placements: list):
    """First (left, right) hand-hold pair. If one hand is never seen on a
    hold before the other moves, that hand starts on the same hold (matched)."""
    left = right = None
    consumed = 0
    for p in placements:
        if p["hand"] == "LEFT" and left is None:
            left = p["hold_id"]
        elif p["hand"] == "RIGHT" and right is None:
            right = p["hold_id"]
        consumed += 1
        if left is not None and right is not None:
            break
    if left is None and right is None:
        return None, 0
    if left is None:
        left = right
    if right is None:
        right = left
    return (left, right), consumed


def default_roles(holds: list, placements: list):
    """Start = hold(s) of the initial state; finish = highest hold the
    climber reached. User can override in the editor."""
    hid = {h["id"]: h for h in holds}
    for h in holds:
        h["role"] = None
    state, _ = initial_state(placements)
    if state is not None:
        for s in set(state):
            if s in hid:
                hid[s]["role"] = "start"
        touched = [hid[p["hold_id"]] for p in placements if p["hold_id"] in hid]
        if touched:
            top = min(touched, key=lambda h: h["y"])
            if top["role"] != "start":
                top["role"] = "finish"
    else:  # no observed sequence: lowest/highest route hold
        route = [h for h in holds if h.get("on_route", True)]
        if route:
            max(route, key=lambda h: h["y"])["role"] = "start"
            min(route, key=lambda h: h["y"])["role"] = "finish"
    return holds


def measured_move_context(pose: dict, placements: list, min_vis=0.5):
    """Diagnostics measured from the video for each observed move (NOT
    part of the objective, so observed and optimized paths are scored
    identically): support-arm elbow angle and hip travel during the move."""
    idxs = sorted(pose["frames"].keys())

    def angle(a, b, c):
        a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
        ba, bc = a - b, c - b
        cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

    out = []
    for i in range(1, len(placements)):
        p0, p1 = placements[i - 1], placements[i]
        window = [f for f in idxs if p0["frame"] <= f <= p1["frame"]]
        support = "RIGHT" if p1["hand"] == "LEFT" else "LEFT"
        elbows, hips = [], []
        for f in window:
            fr = pose["frames"][f]
            s, e, w = fr.get(f"{support}_SHOULDER"), fr.get(f"{support}_ELBOW"), fr.get(f"{support}_WRIST")
            if s and e and w and min(s[2], e[2], w[2]) >= min_vis:
                elbows.append(angle(s, e, w))
            lh, rh = fr.get("LEFT_HIP"), fr.get("RIGHT_HIP")
            if lh and rh and min(lh[2], rh[2]) >= min_vis:
                hips.append(((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2))
        hip_travel = 0.0
        for j in range(1, len(hips)):
            hip_travel += float(np.hypot(hips[j][0] - hips[j - 1][0], hips[j][1] - hips[j - 1][1]))
        out.append({
            "support_elbow_min_deg": float(np.min(elbows)) if elbows else None,
            "support_elbow_mean_deg": float(np.mean(elbows)) if elbows else None,
            "hip_travel_px": hip_travel,
            "duration_frames": p1["frame"] - p0["frame"],
        })
    return out


def truncate_at_finish(placements: list, finish_ids, arm_span_px: float, drop_frac: float = 0.5, hid=None):
    """Once a hand has reached a finish hold, later placements that drop far
    below it are the dismount, not beta. Also drops trailing placements after
    the finish that never go higher (e.g. a hand flailing while jumping off)."""
    finish_ids = set(finish_ids)
    if not placements or not finish_ids:
        return placements
    out = []
    reached = False
    finish_y = None
    for p in placements:
        if reached and hid is not None:
            h = hid.get(p["hold_id"])
            if h is None or (finish_y is not None and h["y"] - finish_y > drop_frac * arm_span_px):
                break
        out.append(p)
        if p["hold_id"] in finish_ids:
            reached = True
            if hid is not None and p["hold_id"] in hid:
                finish_y = hid[p["hold_id"]]["y"]
    return out
