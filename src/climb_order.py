"""
Real climb order, from hand-contact events instead of a height sort.

For each hold, track the minimum distance from either wrist to that
hold's position, frame by frame. When that distance drops below a
contact radius and stays there for a few consecutive frames, that's a
genuine grab (not just a hand passing near the hold on the way to
somewhere else). The first such event's frame index gives the hold's
position in climb order.
"""

import numpy as np

from pose_pipeline import extract_trajectories, summarize_body, process_video
from hold_detection import detect_route_holds


def find_contact_events(trajectories, holds, contact_radius, min_dwell=3):
    """Returns holds with an added 'first_contact_frame' (None if never
    contacted), based on wrist proximity sustained over >= min_dwell frames."""
    frame_idxs = sorted(trajectories.keys())

    for h in holds:
        h["first_contact_frame"] = None

    for h_i, h in enumerate(holds):
        hxy = np.array([h["x"], h["y"]])
        below = []
        for f in frame_idxs:
            pts = trajectories[f]
            dists = []
            for wrist_name in ("LEFT_WRIST", "RIGHT_WRIST"):
                if wrist_name in pts:
                    wx, wy, vis = pts[wrist_name]
                    if vis > 0.5:
                        dists.append(np.hypot(wx - h["x"], wy - h["y"]))
            d = min(dists) if dists else np.inf
            below.append(d < contact_radius)

        # find first run of >= min_dwell consecutive True values
        run_start = None
        run_len = 0
        for i, b in enumerate(below):
            if b:
                if run_start is None:
                    run_start = i
                run_len += 1
                if run_len >= min_dwell:
                    h["first_contact_frame"] = frame_idxs[run_start]
                    break
            else:
                run_start = None
                run_len = 0

    return holds


def compute_climb_order(video_path, contact_radius_frac=0.22):
    print(f"[1/3] Extracting wrist trajectories from {video_path} ...")
    trajectories, w, h_px = extract_trajectories(video_path)
    print(f"      got landmarks for {len(trajectories)} frames")

    print(f"[2/3] Getting body scale + detected holds ...")
    segments, total_frames, detected = process_video(video_path, out_path=None)
    body = summarize_body(segments)
    holds, board_mask = detect_route_holds(video_path)

    contact_radius = contact_radius_frac * body["shoulder_width_px"] * 3  # ~ hand-to-hold tolerance
    print(f"      contact radius = {contact_radius:.0f}px (shoulder width = {body['shoulder_width_px']:.0f}px)")

    print(f"[3/3] Finding contact events for {len(holds)} holds ...")
    holds = find_contact_events(trajectories, holds, contact_radius)

    contacted = [h for h in holds if h["first_contact_frame"] is not None]
    uncontacted = [h for h in holds if h["first_contact_frame"] is None]
    contacted.sort(key=lambda h: h["first_contact_frame"])

    return body, contacted, uncontacted, holds


if __name__ == "__main__":
    import sys
    import cv2

    video_path = sys.argv[1] if len(sys.argv) > 1 else "CruxCam/posevids/DevinRDK.AVI"
    body, contacted, uncontacted, all_holds = compute_climb_order(video_path)

    print(f"\n{len(contacted)}/{len(all_holds)} holds had a detected hand-contact event.")
    print("Climb order (by first contact frame):")
    for order_i, h in enumerate(contacted):
        print(f"  {order_i}: hold at ({h['x']:.0f},{h['y']:.0f})  first touched at frame {h['first_contact_frame']}")

    if uncontacted:
        print(f"\n{len(uncontacted)} holds never registered a contact (likely feet-only holds, "
              f"or a hand pass that didn't linger long enough / MediaPipe occlusion):")
        for h in uncontacted:
            print(f"  hold at ({h['x']:.0f},{h['y']:.0f})")

    # compare to old height-sort order
    height_order = sorted(all_holds, key=lambda h: -h["y"])
    print("\nFor comparison, the height-sort order was:")
    print("  " + " -> ".join(f"({h['x']:.0f},{h['y']:.0f})" for h in height_order))
    print("Contact-based order was:")
    print("  " + " -> ".join(f"({h['x']:.0f},{h['y']:.0f})" for h in contacted))

    # visualize
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = cap.read()
    cap.release()
    if ok:
        for h in uncontacted:
            cv2.circle(frame, (int(h["x"]), int(h["y"])), 16, (128, 128, 128), 2)
        for i, h in enumerate(contacted):
            cv2.circle(frame, (int(h["x"]), int(h["y"])), 16, (0, 200, 255), 3)
            cv2.putText(frame, str(i), (int(h["x"]) + 18, int(h["y"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if i > 0:
                p1 = (int(contacted[i-1]["x"]), int(contacted[i-1]["y"]))
                p2 = (int(h["x"]), int(h["y"]))
                cv2.arrowedLine(frame, p1, p2, (255, 255, 255), 2, tipLength=0.05)
        cv2.imwrite("climb_order.png", frame)
        print("\nSaved visualization to climb_order.png")
