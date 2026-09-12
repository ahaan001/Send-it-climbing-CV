"""
Ghost overlay using REAL cross-climber data instead of a synthetic
biomechanical rule.

Because all 6 climbers were filmed on the same route from the same fixed
camera position, their pixel coordinate systems are already shared -- a
hold at (x, y) means the same physical spot for every climber's video.
That means we can:

  1. Take a reference climber's full-body pose at the frame where they
     contact a given hold.
  2. Scale that skeleton about the hold point by the ratio of the
     target climber's arm span to the reference's arm span.
  3. Draw it translucently onto the target climber's own frame at their
     contact moment for the same hold.

No synthetic "ideal form" rules needed -- the reference IS another real
climber's real form on the exact same move, rescaled to the target's body.
"""

import json

import cv2
import numpy as np

from pose_pipeline import extract_trajectories, TRAJECTORY_LANDMARKS

SKELETON_EDGES = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"), ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"), ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"), ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"), ("LEFT_KNEE", "LEFT_ANKLE"), ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_HIP", "RIGHT_KNEE"), ("RIGHT_KNEE", "RIGHT_ANKLE"), ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]


def nearest_wrist_to_point(frame_landmarks, point_xy):
    best_name, best_d = None, np.inf
    for name in ("LEFT_WRIST", "RIGHT_WRIST"):
        if name in frame_landmarks:
            x, y, vis = frame_landmarks[name]
            d = np.hypot(x - point_xy[0], y - point_xy[1])
            if d < best_d:
                best_d, best_name = d, name
    return best_name


def transform_skeleton(frame_landmarks, anchor_landmark_name, anchor_target_xy, scale):
    ax, ay, _ = frame_landmarks[anchor_landmark_name]
    transformed = {}
    for name, (x, y, vis) in frame_landmarks.items():
        tx = anchor_target_xy[0] + (x - ax) * scale
        ty = anchor_target_xy[1] + (y - ay) * scale
        transformed[name] = (tx, ty, vis)
    return transformed


def draw_skeleton(img, landmarks, color, thickness=3, alpha=1.0):
    overlay = img.copy()
    for a, b in SKELETON_EDGES:
        if a in landmarks and b in landmarks:
            pa = (int(landmarks[a][0]), int(landmarks[a][1]))
            pb = (int(landmarks[b][0]), int(landmarks[b][1]))
            cv2.line(overlay, pa, pb, color, thickness, cv2.LINE_AA)
    for name, (x, y, vis) in landmarks.items():
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    else:
        img[:] = overlay


def build_ghost_overlay(video_paths, hold, bodies, ref_name, target_name, out_path="ghost_overlay.png"):
    ref_frame_idx = hold["per_climber_frame"][ref_name]
    target_frame_idx = hold["per_climber_frame"][target_name]

    ref_traj, w, h = extract_trajectories(video_paths[ref_name])
    target_traj, _, _ = extract_trajectories(video_paths[target_name])

    ref_landmarks = ref_traj[ref_frame_idx]
    target_landmarks = target_traj[target_frame_idx]

    hold_xy = (hold["x"], hold["y"])
    ref_anchor = nearest_wrist_to_point(ref_landmarks, hold_xy)
    target_anchor = nearest_wrist_to_point(target_landmarks, hold_xy)

    scale = bodies[target_name]["arm_span_px"] / bodies[ref_name]["arm_span_px"]
    ghost = transform_skeleton(ref_landmarks, ref_anchor,
                                (target_landmarks[target_anchor][0], target_landmarks[target_anchor][1]),
                                scale)

    cap = cv2.VideoCapture(video_paths[target_name])
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read target frame")

    # target's real pose in solid green, reference's rescaled "ghost" in translucent red
    draw_skeleton(frame, target_landmarks, (0, 220, 0), thickness=3, alpha=1.0)
    draw_skeleton(frame, ghost, (0, 0, 255), thickness=3, alpha=0.55)
    cv2.circle(frame, (int(hold_xy[0]), int(hold_xy[1])), 14, (0, 255, 255), 3)

    cv2.imwrite(out_path, frame)
    return out_path, scale


if __name__ == "__main__":
    climbers = {
        "Aiden": "CruxCam/posevids/AidenRDK.AVI",
        "Malachi": "CruxCam/posevids/MalachiRDK.AVI",
        "Jasper": "CruxCam/posevids/JasperRDK.AVI",
        "Kaia": "CruxCam/posevids/KaiaRDK.AVI",
        "Kat": "CruxCam/posevids/KatRDK.AVI",
        "Mike": "CruxCam/posevids/MikeRDK.AVI",
    }

    with open("multi_climber_data.json") as f:
        data = json.load(f)
    holds = data["holds"]
    bodies = data["bodies"]

    best_hold = max(holds, key=lambda h: h["n_climbers"])
    print(f"Using hold at ({best_hold['x']:.0f},{best_hold['y']:.0f}), "
          f"touched by: {', '.join(best_hold['climbers'])}")

    candidates = best_hold["climbers"]
    ref_name = max(candidates, key=lambda n: bodies[n]["arm_span_px"])
    target_name = min(candidates, key=lambda n: bodies[n]["arm_span_px"])
    print(f"Reference (largest arm span): {ref_name} ({bodies[ref_name]['arm_span_px']:.0f}px)")
    print(f"Target (smallest arm span): {target_name} ({bodies[target_name]['arm_span_px']:.0f}px)")

    out_path, scale = build_ghost_overlay(climbers, best_hold, bodies, ref_name, target_name)
    print(f"Scale factor applied to reference skeleton: {scale:.2f}")
    print(f"Saved ghost overlay to {out_path}")
