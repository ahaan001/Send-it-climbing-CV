"""
End-to-end demo entry point. Run the whole pipeline on one climbing
video and get every artifact the demo needs in one shot:

    python3 run_demo.py path/to/climb.mp4 [--out-dir demo_output]

Produces (in --out-dir):
  skeleton_overlay.mp4   - pose skeleton drawn over the original video
  route_or_moves.png     - detected holds/moves with reach-adjusted %
  wall_angle.png         - board trapezoid measurement
  limiting_factor.png    - per-move limiting-factor classification
  report.json            - every number, for a UI to consume later
  report.txt             - the same thing, human-readable

Works on both a lit (Kilter-style) board and a normal wall: it tries
LED hold detection first, and falls back to the dwell-based per-climber
move sequence (no fixed hold set needed) if that finds nothing.
"""

import argparse
import json
import os

import cv2
import numpy as np

from pose_pipeline import process_video, summarize_body, reach_ratio
from hold_detection import detect_route_holds
from wall_angle import estimate_overhang
from session_stats import compute_session_stats
from limiting_factor import analyze_moves, classify_limiting_factors


def run(video_path, out_dir="demo_output"):
    os.makedirs(out_dir, exist_ok=True)
    report = {"video": video_path}

    print("=" * 60)
    print(f"Running full pipeline on {video_path}")
    print("=" * 60)

    # 1. Body geometry + skeleton overlay video
    print("\n[1/5] Pose estimation + body geometry ...")
    skeleton_path = os.path.join(out_dir, "skeleton_overlay.mp4")
    segments, total_frames, detected_frames = process_video(video_path, out_path=skeleton_path)
    body = summarize_body(segments)
    report["body"] = body
    report["pose_detection_rate"] = detected_frames / total_frames if total_frames else 0
    print(f"      arm_span_px={body['arm_span_px']:.0f}, "
          f"detection={report['pose_detection_rate']:.0%}")

    # 2. Try LED hold detection (Kilter-style); fall back to dwell-based moves
    print("\n[2/5] Hold / move detection ...")
    holds, board_mask = detect_route_holds(video_path)
    route_png = os.path.join(out_dir, "route_or_moves.png")

    if len(holds) >= 2:
        print(f"      lit-hold detection found {len(holds)} holds -- using route-profile mode")
        holds.sort(key=lambda h: -h["y"])
        moves = []
        for i in range(len(holds) - 1):
            a, b = holds[i], holds[i + 1]
            ratio = reach_ratio((a["x"], a["y"]), (b["x"], b["y"]), body["arm_span_px"])
            moves.append({"from": (a["x"], a["y"]), "to": (b["x"], b["y"]), "reach_pct": ratio})
        report["mode"] = "lit_holds"
        report["holds"] = holds
        report["moves"] = [{"from": m["from"], "to": m["to"], "reach_pct": m["reach_pct"]} for m in moves]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            crux = max(moves, key=lambda m: m["reach_pct"]) if moves else None
            for h in holds:
                cv2.circle(frame, (int(h["x"]), int(h["y"])), 16, (0, 200, 255), 3)
            for m in moves:
                p1, p2 = tuple(map(int, m["from"])), tuple(map(int, m["to"]))
                is_crux = m is crux
                color = (0, 0, 255) if is_crux else (255, 255, 255)
                cv2.line(frame, p1, p2, color, 3 if is_crux else 1)
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                cv2.putText(frame, f"{m['reach_pct']:.0%}", mid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imwrite(route_png, frame)
    else:
        print("      no lit holds found -- using dwell-based move sequence (works on any wall)")
        report["mode"] = "dwell_moves"

    # 3. Wall angle
    print("\n[3/5] Wall angle ...")
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first_frame = cap.read()
    cap.release()
    wall = estimate_overhang(first_frame) if ok else None
    if wall:
        report["wall_overhang_ratio"] = wall["ratio"]
        print(f"      trapezoid ratio={wall['ratio']:.3f}")
        vis = first_frame.copy()
        tx0, tx1 = wall["top_span"]; bx0, bx1 = wall["bottom_span"]
        cv2.line(vis, (tx0, wall["top_y"]), (tx1, wall["top_y"]), (0, 0, 255), 4)
        cv2.line(vis, (bx0, wall["bottom_y"]), (bx1, wall["bottom_y"]), (0, 255, 0), 4)
        cv2.imwrite(os.path.join(out_dir, "wall_angle.png"), vis)

    # 4. Session stats
    print("\n[4/5] Session stats ...")
    stats = compute_session_stats(video_path)
    report["session_stats"] = stats
    print(f"      duration={stats['duration_s']:.1f}s, "
          f"distance={stats['distance_climbed_armspans']:.2f} arm-spans, "
          f"exertion_index={stats['exertion_index']:.2f}")

    # 5. Limiting-factor assessment
    print("\n[5/5] Limiting-factor assessment ...")
    _, moves_lf = analyze_moves(video_path)
    if len(moves_lf) >= 3:
        moves_lf, counts, overall = classify_limiting_factors(moves_lf)
        report["limiting_factor"] = {"counts": counts, "overall": overall}
        print(f"      overall limiting factor: {overall.upper()} ({counts})")

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        cap.release()
        if ok:
            colors = {"reach": (0, 0, 255), "strength": (255, 0, 0), "flexibility": (0, 220, 0)}
            for i, m in enumerate(moves_lf):
                p1, p2 = tuple(map(int, m["from"])), tuple(map(int, m["to"]))
                c = colors[m["limiting_factor"]]
                cv2.arrowedLine(frame, p1, p2, c, 3, tipLength=0.15)
                cv2.putText(frame, str(i), (p2[0] + 10, p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            cv2.imwrite(os.path.join(out_dir, "limiting_factor.png"), frame)
    else:
        print("      not enough detected moves for a reliable assessment (need >= 3)")
        report["limiting_factor"] = None

    # Write reports
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    with open(os.path.join(out_dir, "report.txt"), "w") as f:
        f.write(f"Climb analysis: {video_path}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Arm span (px, this video's scale): {body['arm_span_px']:.0f}\n")
        f.write(f"Pose detection rate: {report['pose_detection_rate']:.0%}\n")
        if wall:
            f.write(f"Wall trapezoid ratio: {wall['ratio']:.3f}\n")
        f.write(f"Duration: {stats['duration_s']:.1f}s\n")
        f.write(f"Distance climbed: {stats['distance_climbed_armspans']:.2f} arm-spans "
                f"(~{stats['distance_climbed_m_est']:.2f}m est.)\n")
        f.write(f"Exertion index: {stats['exertion_index']:.2f}\n")
        if report.get("limiting_factor"):
            f.write(f"Limiting factor: {report['limiting_factor']['overall'].upper()} "
                    f"{report['limiting_factor']['counts']}\n")

    print("\n" + "=" * 60)
    print(f"Done. All artifacts in: {out_dir}/")
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("--out-dir", default="demo_output")
    args = parser.parse_args()
    run(args.video_path, args.out_dir)
