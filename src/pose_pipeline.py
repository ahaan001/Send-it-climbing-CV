"""
Tier 1 pipeline for the personalized climbing difficulty app.

1. Run MediaPipe Pose over a climbing video, frame by frame.
2. Track body-segment lengths (shoulder width, upper arm, forearm, torso,
   thigh, shin) across the whole clip.
3. Take a robust (percentile) max of each segment as the true bone-length
   estimate. This matters because a single frame under-estimates a
   segment whenever that limb is angled toward/away from the camera
   (foreshortening); across a whole climbing clip the climber's limbs
   pass through many orientations, so the max-ish value across frames is
   the best estimate of the segment's true length.
4. Combine segments into an estimated "arm span" (fingertip-to-fingertip)
   for the climber, entirely in pixel units.
5. reach_ratio(): given two hold positions (in the SAME video's pixel
   space) and the climber's arm span in that same pixel space, compute
   the move's required reach as a fraction of their wingspan. This is
   the number from the brief -- no real-world calibration needed AS LONG
   AS you're comparing within one climber's own footage, because the
   pixel-to-real-world scale factor cancels out in the ratio.

   Cross-climber comparison (tall climber's clip vs short climber's clip,
   filmed from different distances/zooms) DOES need a shared scale
   reference -- see NOTE at the bottom of this file.
6. Draws the skeleton overlay onto an output video (the visual "wow"
   moment from the demo flow).
"""

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LM = mp_pose.PoseLandmark


def landmark_to_vec(lm, w, h):
    # z is MediaPipe's rough relative-depth estimate; scale it by width
    # so all three axes are in comparable "pixel-ish" units.
    return np.array([lm.x * w, lm.y * h, lm.z * w])


def dist(a, b):
    return float(np.linalg.norm(a - b))


def process_video(video_path, out_path=None, max_frames=None, draw=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    segments = {
        "shoulder_width": [], "l_upper_arm": [], "l_forearm": [],
        "r_upper_arm": [], "r_forearm": [], "torso": [],
        "l_thigh": [], "l_shin": [], "r_thigh": [], "r_shin": [],
    }

    frame_idx = 0
    detected = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                detected += 1
                lm = results.pose_landmarks.landmark
                p = {name: landmark_to_vec(lm[name.value], w, h) for name in LM}

                ls, rs = p[LM.LEFT_SHOULDER], p[LM.RIGHT_SHOULDER]
                le, re = p[LM.LEFT_ELBOW], p[LM.RIGHT_ELBOW]
                lw, rw = p[LM.LEFT_WRIST], p[LM.RIGHT_WRIST]
                lh, rh = p[LM.LEFT_HIP], p[LM.RIGHT_HIP]
                lk, rk = p[LM.LEFT_KNEE], p[LM.RIGHT_KNEE]
                la, ra = p[LM.LEFT_ANKLE], p[LM.RIGHT_ANKLE]

                segments["shoulder_width"].append(dist(ls, rs))
                segments["l_upper_arm"].append(dist(ls, le))
                segments["l_forearm"].append(dist(le, lw))
                segments["r_upper_arm"].append(dist(rs, re))
                segments["r_forearm"].append(dist(re, rw))
                segments["torso"].append(dist((ls + rs) / 2, (lh + rh) / 2))
                segments["l_thigh"].append(dist(lh, lk))
                segments["l_shin"].append(dist(lk, la))
                segments["r_thigh"].append(dist(rh, rk))
                segments["r_shin"].append(dist(rk, ra))

                if writer and draw:
                    mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

            if writer:
                writer.write(frame)
            frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    return segments, frame_idx, detected


def robust_max(values, pct=92):
    return float(np.percentile(values, pct)) if values else None


def summarize_body(segments):
    est = {k: robust_max(v) for k, v in segments.items()}
    upper_arm = np.mean([est["l_upper_arm"], est["r_upper_arm"]])
    forearm = np.mean([est["l_forearm"], est["r_forearm"]])
    thigh = np.mean([est["l_thigh"], est["r_thigh"]])
    shin = np.mean([est["l_shin"], est["r_shin"]])

    arm_span_px = est["shoulder_width"] + 2 * (upper_arm + forearm)
    leg_len_px = thigh + shin
    height_proxy_px = est["torso"] + leg_len_px  # missing head/neck + foot, proxy only

    return {
        "arm_span_px": arm_span_px,
        "height_proxy_px": height_proxy_px,
        "shoulder_width_px": est["shoulder_width"],
        "upper_arm_px": upper_arm,
        "forearm_px": forearm,
        "thigh_px": thigh,
        "shin_px": shin,
    }


def reach_ratio(hold_a_px, hold_b_px, arm_span_px):
    """Required reach for a move as a fraction of the climber's arm span.
    hold_a_px / hold_b_px must be in the SAME video's pixel space as the
    arm_span_px estimate."""
    d = dist(np.array(hold_a_px, dtype=float), np.array(hold_b_px, dtype=float))
    return d / arm_span_px


# Landmarks worth keeping per-frame for climb-order detection and the
# ghost overlay (full pose, not just the segment lengths).
TRAJECTORY_LANDMARKS = [
    LM.LEFT_WRIST, LM.RIGHT_WRIST, LM.LEFT_ELBOW, LM.RIGHT_ELBOW,
    LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP,
    LM.LEFT_KNEE, LM.RIGHT_KNEE, LM.LEFT_ANKLE, LM.RIGHT_ANKLE,
    LM.LEFT_FOOT_INDEX, LM.RIGHT_FOOT_INDEX, LM.NOSE,
]


def extract_trajectories(video_path, max_frames=None):
    """Per-frame (x, y) pixel positions for a fixed set of landmarks,
    for the whole clip. Returns {frame_idx: {landmark_name: (x, y)}}."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    trajectories = {}
    frame_idx = 0
    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames and frame_idx >= max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                frame_pts = {}
                for name in TRAJECTORY_LANDMARKS:
                    l = lm[name.value]
                    frame_pts[name.name] = (l.x * w, l.y * h, l.visibility)
                trajectories[frame_idx] = frame_pts
            frame_idx += 1
    cap.release()
    return trajectories, w, h


if __name__ == "__main__":
    import sys
    import json

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/DevinRDK.AVI"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "skeleton_overlay.mp4"

    print(f"Processing {video_path} ...")
    segments, total_frames, detected_frames = process_video(video_path, out_path=out_path)
    print(f"Frames: {total_frames}, pose detected in: {detected_frames}")

    body = summarize_body(segments)
    print("\nEstimated body proportions (pixel units, this video's scale):")
    print(json.dumps(body, indent=2))

    print("\nDemo reach-ratio calc (illustrative hold coordinates in pixels):")
    hold_a = (400, 800)
    hold_b = (700, 300)
    ratio = reach_ratio(hold_a, hold_b, body["arm_span_px"])
    print(f"  Move from {hold_a} to {hold_b}: reach = {ratio:.2%} of estimated arm span")
