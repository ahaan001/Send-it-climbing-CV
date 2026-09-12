"""Camera-motion compensation.

Handheld climbing footage pans to follow the climber, so a hold's pixel
position drifts over the clip. We register every frame to one reference
frame with an ORB-feature homography (RANSAC rejects the moving climber),
then express holds, hand trajectories and the background in that single
"wall coordinate" frame. A static camera is detected and left untouched.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

import cv2
import numpy as np


def _orb_frames(video_path: str, max_side: int = 720):
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    sc = min(1.0, max_side / max(w, h))
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=12)
    feats = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if sc < 1.0:
            g = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        k, d = orb.detectAndCompute(g, None)
        feats.append((k, d))
    cap.release()
    return feats, w, h, sc


def _match_h(k_src, d_src, k_dst, d_dst, min_inliers=30):
    if d_src is None or d_dst is None or len(k_src) < 12 or len(k_dst) < 12:
        return None, 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    m = bf.match(d_src, d_dst)
    if len(m) < 12:
        return None, 0
    m = sorted(m, key=lambda x: x.distance)[:900]
    p_src = np.float32([k_src[x.queryIdx].pt for x in m])
    p_dst = np.float32([k_dst[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(p_src, p_dst, cv2.RANSAC, 4.0)
    n = int(inl.sum()) if inl is not None else 0
    if H is None or n < min_inliers:
        return None, n
    return H, n


def compute_homographies(video_path: str, ref_idx: Optional[int] = None,
                         progress: Optional[Callable[[float], None]] = None) -> dict:
    """H[i] maps pixel coords of frame i -> pixel coords of the reference frame
    (full-resolution). Falls back to chaining through the previous frame when a
    frame has too little overlap with the reference."""
    feats, w, h, sc = _orb_frames(video_path)
    n = len(feats)
    if ref_idx is None:
        ref_idx = n // 2
    S = np.diag([sc, sc, 1.0])
    S_inv = np.diag([1 / sc, 1 / sc, 1.0])
    k_ref, d_ref = feats[ref_idx]
    Hs = [None] * n
    inliers = [0] * n
    Hs[ref_idx] = np.eye(3)
    # forward and backward passes so chaining always has a known neighbour
    for order in (range(ref_idx + 1, n), range(ref_idx - 1, -1, -1)):
        prev = ref_idx
        for i in order:
            H, ninl = _match_h(feats[i][0], feats[i][1], k_ref, d_ref)
            if H is None:  # chain via neighbour
                H2, ninl2 = _match_h(feats[i][0], feats[i][1], feats[prev][0], feats[prev][1], min_inliers=20)
                if H2 is not None and Hs[prev] is not None:
                    H = Hs[prev] @ H2 if sc == 1.0 else (Hs[prev] @ (S_inv @ H2 @ S))
                    Hs[i] = H
                    inliers[i] = ninl2
                else:
                    Hs[i] = Hs[prev]  # last resort: reuse
                    inliers[i] = 0
            else:
                Hs[i] = S_inv @ H @ S
                inliers[i] = ninl
            prev = i
            if progress and i % 20 == 0:
                progress(i / n)
    # motion magnitude: where the frame centre lands in the reference frame
    c = np.array([[[w / 2, h / 2]]], np.float32)
    shifts = []
    for H in Hs:
        p = cv2.perspectiveTransform(c, H.astype(np.float64))[0, 0]
        shifts.append(float(np.hypot(p[0] - w / 2, p[1] - h / 2)))
    static = max(shifts) < 0.012 * max(w, h)
    return {"w": w, "h": h, "ref_idx": ref_idx, "H": [H.tolist() for H in Hs], "inliers": inliers,
            "max_shift_px": max(shifts), "static": static}


def canvas_geometry(stab: dict, max_scale: float = 2.2):
    """Union of all warped frame corners -> canvas size + translation T such
    that every warped frame has non-negative coordinates."""
    w, h = stab["w"], stab["h"]
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32)
    allc = []
    for H in stab["H"]:
        allc.append(cv2.perspectiveTransform(corners, np.array(H, np.float64))[0])
    allc = np.concatenate(allc)
    x0, y0 = np.floor(allc.min(axis=0))
    x1, y1 = np.ceil(allc.max(axis=0))
    # clamp runaway extents (bad homographies)
    x0, y0 = max(x0, -w * (max_scale - 1)), max(y0, -h * (max_scale - 1))
    x1, y1 = min(x1, w * max_scale), min(y1, h * max_scale)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
    return int(x1 - x0), int(y1 - y0), T


def warp_points(H, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def stabilize_pose(pose: dict, stab: dict, T) -> dict:
    """Transform every landmark into canvas (wall) coordinates."""
    out = {k: v for k, v in pose.items() if k != "frames"}
    frames = {}
    for i, lm in pose["frames"].items():
        H = T @ np.array(stab["H"][int(i)], np.float64)
        names = list(lm.keys())
        pts = warp_points(H, [lm[n][:2] for n in names])
        frames[int(i)] = {n: [float(pts[j][0]), float(pts[j][1]), lm[n][2]] for j, n in enumerate(names)}
    out["frames"] = frames
    return out


def stabilized_background(video_path: str, stab: dict, T, canvas_w: int, canvas_h: int,
                          n_samples: int = 17, max_side: int = 1400) -> np.ndarray:
    """Median over warped sample frames (masked to valid pixels) -> sharp,
    climber-free panorama of the wall in wall coordinates."""
    sc = min(1.0, max_side / max(canvas_w, canvas_h))
    cw, ch = int(canvas_w * sc), int(canvas_h * sc)
    S = np.diag([sc, sc, 1.0])
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(7))
    idxs = np.linspace(0, total - 1, min(n_samples, total)).astype(int)
    warped, masks = [], []
    for i in idxs:
        cap.set(1, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        H = S @ T @ np.array(stab["H"][int(i)], np.float64)
        warped.append(cv2.warpPerspective(fr, H, (cw, ch)))
        masks.append(cv2.warpPerspective(np.full(fr.shape[:2], 255, np.uint8), H, (cw, ch)) > 0)
    cap.release()
    out = np.zeros((ch, cw, 3), np.uint8)
    chunk = 64
    stack = np.stack(warped)                 # n x ch x cw x 3 uint8
    mstack = np.stack(masks)                 # n x ch x cw
    for y in range(0, ch, chunk):
        blk = stack[:, y:y + chunk].astype(np.float32)
        blk[~mstack[:, y:y + chunk]] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(blk, axis=0)
        med = np.nan_to_num(med, nan=0.0)
        out[y:y + chunk] = np.clip(med, 0, 255).astype(np.uint8)
    if sc < 1.0:
        out = cv2.resize(out, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)
    return out


def save(stab: dict, path: str):
    with open(path, "w") as f:
        json.dump(stab, f)


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
