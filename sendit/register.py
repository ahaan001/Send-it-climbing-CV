"""Align the video's wall canvas to a separate route image.

The pipeline writes a climber-free background (background.png) in the
video's wall/canvas coordinates. Holds may instead come from a separate
route image -- a photo of the lit board, or a Kilter app screenshot. This
module finds the homography between the two pictures so the holds from
the route image and the pose from the video live in one frame.

Feature matching reuses the ORB + RANSAC helpers from ``sendit.stabilize``
(the same trick that cancels camera motion between video frames). Images
are matched on downscaled copies for speed and the scaling is undone so
the returned homography maps full-resolution pixels to full-resolution
pixels.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .stabilize import _match_h, warp_points

_CORNER_REACH = 3.0   # a corner further than this * max(w, h) from the origin is a bad fit


def orb_features(gray: np.ndarray, nfeatures: int = 4000):
    """ORB keypoints + descriptors of a grayscale image (keypoints, descriptors)."""
    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    return orb.detectAndCompute(gray, None)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _downscale(gray: np.ndarray, max_side: int):
    """Shrink so the long side is at most max_side; returns (image, scale)."""
    sc = min(1.0, max_side / max(gray.shape[:2]))
    if sc < 1.0:
        gray = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    return gray, sc


def align_images(src_bgr: np.ndarray, dst_bgr: np.ndarray, min_inliers: int = 30,
                 max_side: int = 900) -> tuple[Optional[np.ndarray], int]:
    """Homography H mapping full-resolution src pixels to full-resolution dst
    pixels, plus the number of RANSAC inliers. ORB features are matched with
    a cross-checked Hamming brute-force matcher and the homography fit with
    RANSAC on copies downscaled to ``max_side``; the scaling is undone with
    S / S_inv like ``stabilize.compute_homographies``. Returns (None, n)
    when matching fails, has fewer than ``min_inliers`` inliers, or the
    result is not :func:`plausible`."""
    if src_bgr is None or dst_bgr is None or src_bgr.size == 0 or dst_bgr.size == 0:
        return None, 0
    g_src, sc_src = _downscale(_to_gray(src_bgr), max_side)
    g_dst, sc_dst = _downscale(_to_gray(dst_bgr), max_side)
    k_src, d_src = orb_features(g_src)
    k_dst, d_dst = orb_features(g_dst)
    H_small, n = _match_h(k_src, d_src, k_dst, d_dst, min_inliers=min_inliers)
    if H_small is None:
        return None, n
    # full src px --S_src--> small src px --H_small--> small dst px --S_dst_inv--> full dst px
    S_src = np.diag([sc_src, sc_src, 1.0])
    S_dst_inv = np.diag([1.0 / sc_dst, 1.0 / sc_dst, 1.0])
    H = S_dst_inv @ np.asarray(H_small, np.float64) @ S_src
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    h, w = src_bgr.shape[:2]
    if not plausible(H, w, h):
        return None, n
    return H, n


def plausible(H, w: int, h: int, min_area_ratio: float = 0.2, max_area_ratio: float = 5.0) -> bool:
    """True when the src image corners map to a convex, positively oriented
    quad (no fold or mirror) whose area is within [min, max] * w * h and no
    corner is further than 3 * max(w, h) from the origin."""
    if H is None:
        return False
    H = np.asarray(H, np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64)
    hom = corners @ H.T
    z = hom[:, 2]
    # every corner must stay on the same side of the camera (no wrap through infinity)
    if np.any(np.abs(z) < 1e-12) or not (np.all(z > 0) or np.all(z < 0)):
        return False
    q = hom[:, :2] / z[:, None]
    if not np.all(np.isfinite(q)):
        return False
    if np.max(np.hypot(q[:, 0], q[:, 1])) > _CORNER_REACH * max(w, h):
        return False
    # convex + same winding as the source corners: all consecutive-edge crosses positive
    e = np.roll(q, -1, axis=0) - q
    e_next = np.roll(e, -1, axis=0)
    cross = e[:, 0] * e_next[:, 1] - e[:, 1] * e_next[:, 0]
    if not np.all(cross > 0):
        return False
    area = 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))
    ratio = area / float(w * h)
    return min_area_ratio <= ratio <= max_area_ratio


def homography_from_corners(src_pts, dst_pts) -> np.ndarray:
    """Exact homography from four corresponding points (same order in both
    lists), e.g. the wall corners clicked on the background and on the
    route image."""
    src = np.asarray(src_pts, np.float32).reshape(4, 2)
    dst = np.asarray(dst_pts, np.float32).reshape(4, 2)
    return np.asarray(cv2.getPerspectiveTransform(src, dst), np.float64)


def transform_points(H, pts) -> np.ndarray:
    """Apply H to an (n, 2) array of points; returns (n, 2) float32."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), np.float32)
    return warp_points(H, pts)


def transform_pose(pose: dict, H) -> dict:
    """Copy of a pose dict (``sendit.pose.load_pose`` layout: {"w","h","fps",
    "n_frames","frames": {idx: {name: [x, y, visibility]}}}) with every
    landmark's x, y mapped through H. Visibility and all other keys are
    kept unchanged."""
    out = {k: v for k, v in pose.items() if k != "frames"}
    frames = {}
    for i, lm in pose.get("frames", {}).items():
        names = list(lm.keys())
        if not names:
            frames[i] = {}
            continue
        pts = transform_points(H, [lm[n][:2] for n in names])
        frames[i] = {n: [float(pts[j][0]), float(pts[j][1]), *lm[n][2:]] for j, n in enumerate(names)}
    out["frames"] = frames
    return out


def alignment_error(H, H_true, w: int, h: int) -> float:
    """Mean displacement (pixels) of the four image corners under H versus H_true."""
    corners = [[0, 0], [w, 0], [w, h], [0, h]]
    a = transform_points(H, corners)
    b = transform_points(H_true, corners)
    return float(np.mean(np.hypot(a[:, 0] - b[:, 0], a[:, 1] - b[:, 1])))
