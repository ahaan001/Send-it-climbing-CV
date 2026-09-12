"""
Tier 2 (cheap-win version): detect the lit LEDs on a Kilter-style board.

Insight: on a Kilter board, the holds that are "on" for a given route are
lit LEDs -- vividly saturated and bright compared to the matte black board,
the beige holds, and the surrounding gym. So instead of training a YOLO
model on a labeled hold dataset, plain HSV thresholding finds them.

The single-frame version below still picks up some noise (skin tone,
ceiling lights, lens flare) at plausible saturation/brightness. The fix
isn't a fancier color model, it's time: holds are physically fixed to the
wall, so a real hold's blob shows up in roughly the same pixel location
across many sampled frames, while skin/light noise moves or flickers.
Requiring a detection to recur near the same spot across a chunk of the
clip filters almost all of it out for free.
"""

import cv2
import numpy as np


def detect_board_bbox(frame_bgr, margin_px=12):
    """The board is a large matte-black rectangle -- much darker than the
    surrounding wood/gym wall. Build a mask that follows its actual
    (trapezoidal, due to camera angle) silhouette rather than a bounding
    box, so hold detection stays inside the true board edge and ignores
    reflections off the wood frame or mats around it."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)

    board_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(board_mask, [largest], -1, 255, thickness=-1)
    # Erode inward so the board's own edge (and any wood sliver just
    # outside it) never counts as "inside".
    board_mask = cv2.erode(board_mask, np.ones((margin_px * 2 + 1,) * 2, np.uint8))
    return board_mask


def detect_lit_blobs(frame_bgr, board_mask=None, min_area=25, max_area=2500):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Lit LEDs: high saturation AND high brightness. This one threshold
    # catches all hold colors (blue/purple/green/orange/pink) at once.
    mask = cv2.inRange(hsv, (0, 130, 150), (179, 255, 255))

    if board_mask is not None:
        mask = cv2.bitwise_and(mask, board_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            hue = int(hsv[int(cy), int(cx), 0])
            blobs.append((cx, cy, hue, area))
    return blobs


def detect_route_holds(video_path, num_samples=25, merge_dist=22, min_hit_frac=0.25):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, min(num_samples, total)).astype(int)

    # Establish the board region once, from the first frame.
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idxs[0]))
    ok, first_frame = cap.read()
    board_mask = detect_board_bbox(first_frame) if ok else None

    detections = []  # (x, y, hue, area, frame_idx)
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        for (x, y, hue, area) in detect_lit_blobs(frame, board_mask=board_mask):
            detections.append((x, y, hue, area, int(idx)))
    cap.release()

    # Cluster detections across frames by proximity (holds don't move).
    clusters = []  # each: {"pts": [(x,y,hue,area,frame_idx), ...]}
    for x, y, hue, area, fidx in detections:
        placed = False
        for cl in clusters:
            mx = np.mean([p[0] for p in cl["pts"]])
            my = np.mean([p[1] for p in cl["pts"]])
            if np.hypot(x - mx, y - my) < merge_dist:
                cl["pts"].append((x, y, hue, area, fidx))
                placed = True
                break
        if not placed:
            clusters.append({"pts": [(x, y, hue, area, fidx)]})

    min_hits = max(2, int(min_hit_frac * len(idxs)))
    holds = []
    for cl in clusters:
        distinct_frames = {p[4] for p in cl["pts"]}
        if len(distinct_frames) >= min_hits:
            xs = [p[0] for p in cl["pts"]]
            ys = [p[1] for p in cl["pts"]]
            hues = [p[2] for p in cl["pts"]]
            holds.append({
                "x": float(np.mean(xs)),
                "y": float(np.mean(ys)),
                "hue": int(np.median(hues)),
                "hits": len(distinct_frames),
                "frames_sampled": len(idxs),
            })
    return holds, board_mask


def label_color(hue):
    # Rough OpenCV-hue (0-179) bucket labels for common hold LED colors.
    if hue < 8 or hue > 170:
        return "red/pink"
    if hue < 22:
        return "orange"
    if hue < 40:
        return "yellow"
    if hue < 78:
        return "green"
    if hue < 132:
        return "blue"
    return "purple"


if __name__ == "__main__":
    import sys
    import json
    import cv2

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/DevinRDK.AVI"

    print(f"Scanning {video_path} for lit holds ...")
    holds, board_mask = detect_route_holds(video_path)
    holds.sort(key=lambda h: -h["y"])  # bottom of wall first (climb order, start -> finish)

    print(f"\nFound {len(holds)} consistent lit holds:")
    for i, h in enumerate(holds):
        print(f"  #{i}: ({h['x']:.0f}, {h['y']:.0f})  color={label_color(h['hue'])}  "
              f"hit in {h['hits']}/{h['frames_sampled']} sampled frames")

    # Draw them on a reference frame for a visual sanity check.
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        if board_mask is not None:
            edge = cv2.Canny(board_mask, 50, 150)
            frame[edge > 0] = (0, 255, 255)
        for i, h in enumerate(holds):
            cv2.circle(frame, (int(h["x"]), int(h["y"])), 16, (0, 0, 255), 2)
            cv2.putText(frame, str(i), (int(h["x"]) + 18, int(h["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite("detected_holds.png", frame)
        print("\nSaved visualization to detected_holds.png")
