"""Pose extraction + morphology estimation (MediaPipe Pose).

Everything is in pixel units of the source video. Reach ratios are
calibration-free because pixel scale cancels within one video.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import cv2
import numpy as np

LANDMARK_NAMES = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_INDEX", "RIGHT_INDEX",
    "LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
    "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
]

SKELETON_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"), ("LEFT_WRIST", "LEFT_INDEX"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"), ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]


def extract_pose(video_path: str, max_frames: Optional[int] = None,
                 progress: Optional[Callable[[float], None]] = None,
                 model_complexity: int = 1) -> dict:
    """Run MediaPipe Pose over every frame. Returns
    {"w","h","fps","n_frames","frames": {frame_idx: {name: [x, y, visibility]}}}
    with x, y in pixels of the source video."""
    import mediapipe as mp  # local import: slow, and optional for cached mode

    mp_pose = mp.solutions.pose
    LM = mp_pose.PoseLandmark
    keep = [(name, LM[name].value) for name in LANDMARK_NAMES]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    frames = {}
    idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=model_complexity,
                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and idx >= max_frames):
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm = res.pose_landmarks.landmark
                frames[idx] = {name: [round(lm[v].x * w, 1), round(lm[v].y * h, 1), round(lm[v].visibility, 3)]
                               for name, v in keep}
            idx += 1
            if progress and idx % 10 == 0:
                progress(min(1.0, idx / total))
    cap.release()
    if progress:
        progress(1.0)
    return {"w": w, "h": h, "fps": fps, "n_frames": idx, "frames": frames}


def _robust_max(vals, pct=92):
    return float(np.percentile(vals, pct)) if len(vals) else None


def morphology_from_pose(pose: dict, min_vis: float = 0.5) -> dict:
    """Estimate body-segment lengths (pixels) with a robust percentile-max
    across the clip: a single frame under-estimates a segment whenever it
    is foreshortened, but across a whole climb each limb passes through
    many orientations, so a high percentile approximates the true length.
    """
    segs = {k: [] for k in ["shoulder", "l_upper", "r_upper", "l_fore", "r_fore",
                            "torso", "l_thigh", "r_thigh", "l_shin", "r_shin", "l_hand", "r_hand"]}

    def d(p, a, b):
        pa, pb = p.get(a), p.get(b)
        if pa is None or pb is None or pa[2] < min_vis or pb[2] < min_vis:
            return None
        return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))

    for p in pose["frames"].values():
        pairs = {
            "shoulder": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
            "l_upper": ("LEFT_SHOULDER", "LEFT_ELBOW"), "r_upper": ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
            "l_fore": ("LEFT_ELBOW", "LEFT_WRIST"), "r_fore": ("RIGHT_ELBOW", "RIGHT_WRIST"),
            "l_thigh": ("LEFT_HIP", "LEFT_KNEE"), "r_thigh": ("RIGHT_HIP", "RIGHT_KNEE"),
            "l_shin": ("LEFT_KNEE", "LEFT_ANKLE"), "r_shin": ("RIGHT_KNEE", "RIGHT_ANKLE"),
            "l_hand": ("LEFT_WRIST", "LEFT_INDEX"), "r_hand": ("RIGHT_WRIST", "RIGHT_INDEX"),
        }
        for k, (a, b) in pairs.items():
            v = d(p, a, b)
            if v is not None:
                segs[k].append(v)
        # torso: shoulder midpoint to hip midpoint
        ls, rs, lh, rh = p.get("LEFT_SHOULDER"), p.get("RIGHT_SHOULDER"), p.get("LEFT_HIP"), p.get("RIGHT_HIP")
        if all(x is not None and x[2] >= min_vis for x in (ls, rs, lh, rh)):
            sm = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
            hm = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
            segs["torso"].append(float(np.hypot(sm[0] - hm[0], sm[1] - hm[1])))

    est = {k: _robust_max(v) for k, v in segs.items()}

    def mean2(a, b):
        vals = [x for x in (est[a], est[b]) if x is not None]
        return float(np.mean(vals)) if vals else None

    upper, fore, thigh, shin, hand = (mean2("l_upper", "r_upper"), mean2("l_fore", "r_fore"),
                                      mean2("l_thigh", "r_thigh"), mean2("l_shin", "r_shin"),
                                      mean2("l_hand", "r_hand"))
    shoulder, torso = est["shoulder"], est["torso"]
    if None in (upper, fore, shoulder):
        raise ValueError("Not enough visible upper-body landmarks to estimate morphology")
    hand = hand or 0.35 * fore
    arm_span = shoulder + 2 * (upper + fore + hand)  # fingertip to fingertip
    leg = (thigh or 0) + (shin or 0)
    return {
        "arm_span_px": float(arm_span),
        "shoulder_width_px": float(shoulder),
        "upper_arm_px": float(upper),
        "forearm_px": float(fore),
        "hand_px": float(hand),
        "torso_px": float(torso) if torso else None,
        "thigh_px": float(thigh) if thigh else None,
        "shin_px": float(shin) if shin else None,
        "leg_len_px": float(leg) if leg else None,
        "height_proxy_px": float((torso or 0) + leg) if torso and leg else None,
        "n_pose_frames": len(pose["frames"]),
        "pose_detection_rate": len(pose["frames"]) / max(1, pose["n_frames"]),
        # dimensionless ratios (calibration-free, comparable across climbers)
        "ratios": {
            "arm_span_over_height_proxy": float(arm_span / (torso + leg)) if torso and leg else None,
            "forearm_over_upper_arm": float(fore / upper),
            "leg_over_arm_span": float(leg / arm_span) if leg else None,
        },
    }


def hand_point(frame_landmarks: dict, side: str, min_vis: float = 0.5):
    """Best estimate of the hand centre for 'LEFT'/'RIGHT': midpoint of
    wrist and index fingertip when both visible, else whichever is visible."""
    w, i = frame_landmarks.get(f"{side}_WRIST"), frame_landmarks.get(f"{side}_INDEX")
    pts = [p for p in (w, i) if p is not None and p[2] >= min_vis]
    if not pts:
        return None
    x = float(np.mean([p[0] for p in pts]))
    y = float(np.mean([p[1] for p in pts]))
    vis = float(min(p[2] for p in pts))
    return (x, y, vis)


def save_pose(pose: dict, path: str):
    with open(path, "w") as f:
        json.dump(pose, f)


def load_pose(path: str) -> dict:
    with open(path) as f:
        pose = json.load(f)
    pose["frames"] = {int(k): v for k, v in pose["frames"].items()}
    return pose
