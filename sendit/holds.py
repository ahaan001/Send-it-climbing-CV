"""Hold detection + hold-set schema.

Reliability hierarchy (the UI exposes every level):
  1. automatic detection (lit LEDs on a Kilter/Moon-style board, or
     saturated-colour blobs on a normal gym wall)
  2. automatic detection + manual correction (add / move / remove / rate)
  3. holds inferred from where the climber's hands actually dwelled
  4. fully manual hold definition

A hold is a dict:
  {"id": int, "x": px, "y": px, "grip": 1..5, "on_route": bool,
   "role": "start" | "finish" | None, "source": "led"|"color"|"dwell"|"manual",
   "label": str}
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

GRIP_NEUTRAL = 3
GRIP_LABELS = {1: "excellent", 2: "good", 3: "average", 4: "poor", 5: "terrible"}


def make_hold(hid, x, y, source="manual", grip=GRIP_NEUTRAL, on_route=True, role=None, label=None, **extra):
    h = {"id": int(hid), "x": float(x), "y": float(y), "grip": int(grip), "on_route": bool(on_route),
         "role": role, "source": source, "label": label or f"H{hid}"}
    h.update(extra)
    return h


def next_id(holds):
    return (max((h["id"] for h in holds), default=-1) + 1)


def background_frame(video_path: str, n_samples: int = 21) -> np.ndarray:
    """Per-pixel median over frames sampled across the clip. The climber
    moves, the wall doesn't, so the median is a clean, climber-free wall
    image -- ideal both for hold detection and as the UI canvas."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(0, total - 1), min(n_samples, max(1, total))).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {video_path}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read {video_path}")
    return fr


def detect_led_holds(video_path: str):
    """Lit-LED holds on a Kilter/Moon/Tension style board (HSV threshold,
    temporally consistent across sampled frames). Returns [] on a normal
    wall, which the caller uses to fall back to colour detection."""
    from hold_detection import detect_route_holds  # existing repo module
    holds, _ = detect_route_holds(video_path)
    out = []
    for i, h in enumerate(sorted(holds, key=lambda h: -h["y"])):
        out.append(make_hold(i, h["x"], h["y"], source="led", hits=h["hits"], hue=h["hue"]))
    return out


def _hue_label(hue):
    if hue < 8 or hue > 170:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 130:
        return "blue"
    return "purple"


def detect_color_holds(bg_bgr: np.ndarray, sat_min=95, val_min=60,
                       min_area_frac=2e-5, max_area_frac=4e-3):
    """Saturated colour blobs on the climber-free background image.
    Works for typical gym holds (bright plastics on grey/beige/black
    panels). Painted wall features are rejected by the max-area filter;
    the human-in-the-loop editor handles what slips through."""
    h_img, w_img = bg_bgr.shape[:2]
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, sat_min, val_min), (179, 255, 255))
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = h_img * w_img
    blobs = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < min_area_frac * area_img or a > max_area_frac * area_img:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
        x, y, bw, bh = cv2.boundingRect(c)
        if bw > 0 and bh > 0 and max(bw / bh, bh / bw) > 4.0:  # elongated = tape/edge, not a hold
            continue
        hue = int(np.median(hsv[y:y + bh, x:x + bw, 0][mask[y:y + bh, x:x + bw] > 0]))
        blobs.append((cx, cy, a, hue))
    blobs.sort(key=lambda b: -b[1])  # bottom of wall first
    return [make_hold(i, cx, cy, source="color", area=float(a), hue=hue, color=_hue_label(hue))
            for i, (cx, cy, a, hue) in enumerate(blobs)]


def detect_holds(video_path: str, bg: np.ndarray | None = None):
    """Auto-detect: LED holds first (objective route membership on a lit
    board); otherwise colour blobs on the background image."""
    led = detect_led_holds(video_path)
    if len(led) >= 3:
        return led, "led"
    if bg is None:
        bg = background_frame(video_path)
    return detect_color_holds(bg), "color"


def dwell_points(pose: dict, arm_span_px: float, speed_frac=0.02, min_frames=5, min_vis=0.5):
    """Where each hand stopped moving (a grip, not a hand in transit).
    Returns [{"hand": "LEFT"/"RIGHT", "x", "y", "frame", "n_frames"}]."""
    from .pose import hand_point
    thresh = speed_frac * arm_span_px
    out = []
    idxs = sorted(pose["frames"].keys())
    for side in ("LEFT", "RIGHT"):
        series = []
        for f in idxs:
            hp = hand_point(pose["frames"][f], side, min_vis)
            if hp is not None:
                series.append((f, hp[0], hp[1]))
        run = []
        for i in range(len(series)):
            if i > 0 and series[i][0] - series[i - 1][0] <= 2:
                sp = np.hypot(series[i][1] - series[i - 1][1], series[i][2] - series[i - 1][2])
            else:
                sp = np.inf
            if sp < thresh:
                run.append(series[i])
            else:
                if len(run) >= min_frames:
                    out.append({"hand": side, "x": float(np.median([r[1] for r in run])),
                                "y": float(np.median([r[2] for r in run])), "frame": run[0][0], "n_frames": len(run)})
                run = [series[i]]
        if len(run) >= min_frames:
            out.append({"hand": side, "x": float(np.median([r[1] for r in run])),
                        "y": float(np.median([r[2] for r in run])), "frame": run[0][0], "n_frames": len(run)})
    out.sort(key=lambda e: e["frame"])
    return out


def dwell_inferred_holds(pose: dict, arm_span_px: float, holds: list, radius_frac=0.12, merge_frac=0.06):
    """Hand dwell locations that are not near any known hold -> proposed
    holds (source='dwell'). This is the fallback that keeps a normal wall
    usable when colour detection misses the holds the climber actually used."""
    pts = dwell_points(pose, arm_span_px)
    radius = radius_frac * arm_span_px
    merge = merge_frac * arm_span_px
    # ignore dwells below the wall (climber standing on the mat / dismount)
    floor_y = (max(h["y"] for h in holds) + 0.25 * arm_span_px) if holds else float("inf")
    proposed = []
    for p in pts:
        if p["y"] > floor_y:
            continue
        near_known = any(np.hypot(p["x"] - h["x"], p["y"] - h["y"]) < radius for h in holds)
        if near_known:
            continue
        for q in proposed:
            if np.hypot(p["x"] - q["x"], p["y"] - q["y"]) < merge:
                q["n"] += 1
                q["x"] = (q["x"] * (q["n"] - 1) + p["x"]) / q["n"]
                q["y"] = (q["y"] * (q["n"] - 1) + p["y"]) / q["n"]
                break
        else:
            proposed.append({"x": p["x"], "y": p["y"], "n": 1})
    nid = next_id(holds)
    return [make_hold(nid + i, q["x"], q["y"], source="dwell", dwell_count=q["n"]) for i, q in enumerate(proposed)]


def by_id(holds):
    return {h["id"]: h for h in holds}


def route_holds(holds):
    return [h for h in holds if h.get("on_route", True)]


def merge_close_holds(holds: list, min_dist: float):
    """Merge detections closer than min_dist px (duplicate blobs of one hold)."""
    out = []
    for h in sorted(holds, key=lambda h: (h["source"] != "led", -h.get("hits", 0))):
        for q in out:
            if np.hypot(h["x"] - q["x"], h["y"] - q["y"]) < min_dist:
                break
        else:
            out.append(dict(h))
    for i, h in enumerate(out):
        h["id"], h["label"] = i, f"H{i}"
    return out
