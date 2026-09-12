"""
Wall angle (overhang) estimation.

The board is a flat rectangular panel in real life. When it's overhung
and filmed from a camera near the base of the wall (the typical setup
for a mounted gym camera or a phone propped on the floor/crash pad),
the top of the panel is farther from the camera and recedes, so it
projects narrower than the bottom -- the board looks like a trapezoid
instead of a rectangle. The more the wall overhangs, the more
pronounced that narrowing is.

This gives a genuinely useful RELATIVE steepness index for free (no
extra hardware). Turning it into calibrated degrees needs to know the
camera's field of view, which we don't -- so estimate_overhang() reports
the raw trapezoid ratio, and calibrate_angle() lets you convert that to
degrees IF you know the true angle for one reference climb (e.g. many
gyms/Kilter boards post their angle -- 20, 25, 40 degrees etc). Once
calibrated for a given camera setup, reuse that calibration for any
other climb filmed from the same rig.

Caveat worth keeping in the pitch: this assumes the camera itself is
held level and at a consistent position relative to the wall base.
Camera tilt or a very off-axis placement will bias the estimate.
"""

import math

import cv2
import numpy as np

from hold_detection import detect_board_bbox


def estimate_overhang(frame_bgr, top_frac=0.12, bottom_frac=0.88, min_area_frac=0.15):
    """Measure the board's width near its top and bottom by scanning
    rows of the already-validated eroded board mask (from
    hold_detection.detect_board_bbox), rather than fitting exact
    polygon corners -- much more robust to the mask's boundary noise
    (light beams, background clutter) than approxPolyDP corner-fitting.

    Returns None (rather than a bogus number) if the mask is too small
    to plausibly be the wall -- this happens on light-colored walls,
    where the "find the dark rectangle" trick has nothing to grab onto
    and instead locks onto something small and dark (hair, shadow,
    a shoe) that isn't the wall at all."""
    h_img, w_img = frame_bgr.shape[:2]
    board_mask = detect_board_bbox(frame_bgr)
    if board_mask is None:
        return None

    ys, xs = np.where(board_mask > 0)
    if len(ys) == 0:
        return None

    # Sanity check: a real wall should cover a large chunk of the frame.
    # A tiny mask means we grabbed the wrong (small, dark) object.
    mask_area_frac = len(ys) / (h_img * w_img)
    bbox_w_frac = (xs.max() - xs.min()) / w_img
    if mask_area_frac < min_area_frac or bbox_w_frac < 0.3:
        return None

    y_min, y_max = ys.min(), ys.max()
    top_y = int(y_min + top_frac * (y_max - y_min))
    bottom_y = int(y_min + bottom_frac * (y_max - y_min))

    def row_width(y):
        row = np.where(board_mask[y] > 0)[0]
        if len(row) == 0:
            return None, None, None
        return float(row.max() - row.min()), int(row.min()), int(row.max())

    top_w, top_x0, top_x1 = row_width(top_y)
    bottom_w, bot_x0, bot_x1 = row_width(bottom_y)
    if not top_w or not bottom_w:
        return None

    ratio = top_w / bottom_w
    return {
        "ratio": ratio,
        "top_y": top_y, "bottom_y": bottom_y,
        "top_width_px": top_w, "bottom_width_px": bottom_w,
        "top_span": (top_x0, top_x1), "bottom_span": (bot_x0, bot_x1),
        "board_mask": board_mask,
    }


def calibrate_angle(ratio, known_angle_degrees, other_ratio=None, flat_ratio=1.0):
    """Very rough linear calibration: assumes ratio == 1.0 (perfect
    rectangle, no perspective narrowing) corresponds to 0 degrees
    overhang, and the given (ratio, known_angle_degrees) is a second
    point. Returns a function mapping new ratios to approximate degrees.
    Only valid for the SAME camera position/setup used to measure it."""
    if ratio >= flat_ratio:
        raise ValueError("ratio must be < flat_ratio to calibrate a non-zero angle")
    slope = known_angle_degrees / (flat_ratio - ratio)

    def angle_from_ratio(r):
        return max(0.0, slope * (flat_ratio - r))

    return angle_from_ratio


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/DevinRDK.AVI"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("couldn't read frame")

    result = estimate_overhang(frame)
    print(f"Board width near top:    {result['top_width_px']:.0f}px (at y={result['top_y']})")
    print(f"Board width near bottom: {result['bottom_width_px']:.0f}px (at y={result['bottom_y']})")
    print(f"Trapezoid ratio (top/bottom): {result['ratio']:.3f}")
    print("  ratio < 1.0 => top narrower than bottom, consistent with an overhang")
    print("  (this is a relative steepness index, not calibrated degrees -- see "
          "calibrate_angle() if you know this board's real angle)")

    vis = frame.copy()
    tx0, tx1 = result["top_span"]
    bx0, bx1 = result["bottom_span"]
    cv2.line(vis, (tx0, result["top_y"]), (tx1, result["top_y"]), (0, 0, 255), 4)
    cv2.line(vis, (bx0, result["bottom_y"]), (bx1, result["bottom_y"]), (0, 255, 0), 4)
    edge = cv2.Canny(result["board_mask"], 50, 150)
    vis[edge > 0] = (0, 255, 255)
    cv2.imwrite("wall_angle.png", vis)
    print("\nSaved visualization to wall_angle.png")
