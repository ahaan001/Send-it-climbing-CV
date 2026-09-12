"""
Strava-style session stats, assembled from the pose trajectory + wall
angle estimate.

Distance/height are reported primarily in "arm-span units" (multiples
of the climber's own estimated arm span) since that's calibration-free
and consistent with the rest of the pipeline. A meters estimate is also
given using the average adult arm-span-approx-height rule of thumb
(~1.0x height), clearly flagged as an approximation -- swap in the
climber's real height for an exact conversion.
"""

import numpy as np

from pose_pipeline import extract_trajectories, process_video, summarize_body
from wall_angle import estimate_overhang

AVG_ADULT_ARM_SPAN_M = 1.70  # rough population average, used only for the approximate meters conversion


def hip_midpoint_trajectory(trajectories):
    frame_idxs = sorted(trajectories.keys())
    pts = []
    for f in frame_idxs:
        p = trajectories[f]
        if "LEFT_HIP" in p and "RIGHT_HIP" in p:
            lx, ly, lv = p["LEFT_HIP"]
            rx, ry, rv = p["RIGHT_HIP"]
            if lv > 0.5 and rv > 0.5:
                pts.append((f, (lx + rx) / 2, (ly + ry) / 2))
    return pts


def compute_session_stats(video_path, assumed_arm_span_m=AVG_ADULT_ARM_SPAN_M):
    print(f"Analyzing session: {video_path} ...")
    trajectories, w, h = extract_trajectories(video_path)
    segments, total_frames, detected_frames = process_video(video_path, out_path=None)
    body = summarize_body(segments)

    cap_fps = 30.0  # overridden below if we can read it
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or cap_fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()
    cap.release()

    duration_s = total_frames / fps

    hip_traj = hip_midpoint_trajectory(trajectories)
    path_len_px = 0.0
    net_vertical_px = 0.0
    if len(hip_traj) >= 2:
        for i in range(1, len(hip_traj)):
            _, x0, y0 = hip_traj[i - 1]
            _, x1, y1 = hip_traj[i]
            path_len_px += np.hypot(x1 - x0, y1 - y0)
        net_vertical_px = hip_traj[0][2] - hip_traj[-1][2]  # image y decreases going up

    arm_span_px = body["arm_span_px"]
    px_per_m = arm_span_px / assumed_arm_span_m

    # Exertion proxy: total limb-endpoint movement (wrists + ankles), which
    # scales with how much the climber was actively moving/repositioning --
    # NOT a physiologically validated calorie/effort model, just a relative
    # index for comparing sessions or climbers.
    exertion_px = 0.0
    frame_idxs = sorted(trajectories.keys())
    limb_names = ["LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE"]
    for i in range(1, len(frame_idxs)):
        p0, p1 = trajectories[frame_idxs[i - 1]], trajectories[frame_idxs[i]]
        for name in limb_names:
            if name in p0 and name in p1 and p0[name][2] > 0.5 and p1[name][2] > 0.5:
                exertion_px += np.hypot(p1[name][0] - p0[name][0], p1[name][1] - p0[name][1])

    wall = estimate_overhang(first_frame) if ok else None

    return {
        "duration_s": duration_s,
        "frames": total_frames,
        "pose_detection_rate": detected_frames / total_frames if total_frames else 0,
        "distance_climbed_armspans": path_len_px / arm_span_px,
        "distance_climbed_m_est": path_len_px / px_per_m,
        "net_vertical_armspans": net_vertical_px / arm_span_px,
        "net_vertical_m_est": net_vertical_px / px_per_m,
        "exertion_index": exertion_px / arm_span_px,  # normalized by body scale so climbers are comparable
        "wall_overhang_ratio": wall["ratio"] if wall else None,
        "arm_span_px": arm_span_px,
    }


if __name__ == "__main__":
    import sys
    import json

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/DevinRDK.AVI"
    stats = compute_session_stats(video_path)

    print(f"\nSession stats for {video_path}:")
    print(f"  Duration:              {stats['duration_s']:.1f}s ({stats['frames']} frames, "
          f"{stats['pose_detection_rate']:.0%} pose detection)")
    print(f"  Distance climbed:      {stats['distance_climbed_armspans']:.2f} arm-spans "
          f"(~{stats['distance_climbed_m_est']:.2f}m, using avg arm span assumption)")
    print(f"  Net vertical gain:     {stats['net_vertical_armspans']:.2f} arm-spans "
          f"(~{stats['net_vertical_m_est']:.2f}m)")
    print(f"  Exertion index:        {stats['exertion_index']:.2f} (relative, body-scale normalized)")
    if stats["wall_overhang_ratio"] is not None:
        print(f"  Wall trapezoid ratio:  {stats['wall_overhang_ratio']:.3f} "
              f"({'overhung' if stats['wall_overhang_ratio'] < 0.97 else 'near vertical'})")
