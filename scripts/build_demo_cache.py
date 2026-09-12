"""(Re)build the demo's cached analysis for the two-input flow (route photo + climb video).

    python3 scripts/build_demo_cache.py [--force] [--video]

For each demo in demo_assets/demos.json:
  1. pose + stabilisation + climber-free background from the video (cached in <cache>/),
  2. lit holds from the route photo (or the saved route.json / holds_edited.json roles),
  3. photo <-> video alignment with ORB + RANSAC (identity fallback when the photo IS the background frame),
  4. apply_route(): pose, morphology and the observed hand sequence in the route photo's frame,
  5. skeleton overlay video, precomputed hands+feet plans (fourlimb.json), preview PNGs and summary.json.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sendit import viz, kilter, register  # noqa: E402
from sendit import beta as beta_mod  # noqa: E402
from sendit import stabilize  # noqa: E402
from sendit.optimizer import Feasibility  # noqa: E402
from sendit.pipeline import (analyze_video, apply_route, run_optimization, feet_for_result, fourlimb_cache_key,  # noqa: E402
                             holds_for_video_frame)
from sendit.pose import load_pose  # noqa: E402


def start_and_rest(holds, placements):
    """Same rule as app.py: start = the green holds; keep the observed hand assignment when it agrees."""
    setup = kilter.kilter_setup(holds)
    start_set = set(setup["start_ids"])
    placements = list(placements)
    st_obs, consumed = beta_mod.initial_state(placements)
    if st_obs and (not start_set or set(st_obs) <= start_set):
        return st_obs, placements[consumed:], setup
    if setup["start_state"]:
        while placements and placements[0]["hold_id"] in start_set:
            placements.pop(0)
        return tuple(setup["start_state"]), placements, setup
    return None, placements, setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--video", action="store_true", help="re-render overlay videos")
    args = ap.parse_args()
    demos = json.load(open(os.path.join(ROOT, "demo_assets", "demos.json")))["demos"]
    for d in demos:
        if args.only and d["key"] != args.only:
            continue
        video = os.path.join(ROOT, d["video"])
        cache = os.path.join(ROOT, d["cache"])
        route_image = os.path.join(ROOT, d["route_image"])
        print(f"== {d['key']}")
        a = analyze_video(video, cache, force=args.force, wall_type="led", detect_holds=False, progress=lambda s, f: None)
        print(f"   pose {a['timing']['pose_s']:.1f}s  stabilize {a['timing']['stabilize_s']:.1f}s")

        # ---- holds from the route photo, with saved roles when available
        img = cv2.imread(route_image)
        rf_path = os.path.join(cache, "route.json")
        cur = os.path.join(cache, "holds_edited.json")
        angle, body = d.get("angle", 40), None
        if os.path.exists(rf_path):
            rf = json.load(open(rf_path))
            holds, angle, body = rf["holds"], rf.get("angle", angle), rf.get("body")
            print(f"   holds from route.json ({len(holds)})")
        elif os.path.exists(cur):
            holds = json.load(open(cur))
            print(f"   holds from holds_edited.json ({len(holds)})")
        else:
            holds = kilter.detect_route_holds(img)
            print(f"   holds detected on the route photo ({len(holds)})")
        if d.get("finish_top"):
            hand = [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]
            top = min(hand, key=lambda h: h["y"])
            for h in holds:
                if h.get("role") == "finish":
                    h["role"] = None
            top["role"] = "finish"
        for w in kilter.role_warnings(holds):
            print("   warning:", w)

        # ---- align the video's background frame to the route photo
        bg = cv2.imread(os.path.join(cache, "background.png"))
        H, n = register.align_images(bg, img)
        if H is None or not register.plausible(H, bg.shape[1], bg.shape[0]):
            H, info = np.eye(3), {"method": "identity", "inliers": 0}
        else:
            info = {"method": "auto", "inliers": int(n)}
        print(f"   alignment: {info}")
        a = apply_route(a, holds, H, route_image, info)
        json.dump({"name": d["name"], "holds": holds, "angle": angle, "body": body}, open(rf_path, "w"), indent=1)
        json.dump(holds, open(cur, "w"), indent=1)
        placements = a["placements"]
        print(f"   placements {len(placements)}  arm span {a['morphology']['arm_span_px']:.0f} px")

        # ---- overlay video with the route holds stuck to the wall
        ov = os.path.join(cache, "overlay.mp4")
        if args.force or not os.path.exists(ov) or args.video:
            pose = load_pose(os.path.join(cache, "pose.json"))
            stab = stabilize.load(os.path.join(cache, "stab.json"))
            if stab["static"]:
                stab = {**stab, "H": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * len(stab["H"])}
            viz.render_pose_overlay_video(video, pose, ov, holds_for_frame=lambda i: holds_for_video_frame(holds, a, stab, i))
            print("   overlay video ->", ov)

        # ---- optimisation with the Kilter setup (start = green, finish = pink, both hands on the finish)
        st_, rest, setup = start_and_rest(holds, placements)
        fin = setup["finish_ids"] or None
        feas = Feasibility(match_finish=True)
        R = run_optimization(holds, a["morphology"], rest, feas=feas, start_state=st_, finish_ids=fin)
        obs, opt, cmp_ = R["observed"], R["optimized"], R["comparison"]
        box = viz.crop_box(holds, img.shape[1], img.shape[0])
        store = {}
        for scale in (1.0, 0.85):
            t0 = time.time()
            R4 = run_optimization(holds, a["morphology"], rest, feas=feas, start_state=st_, finish_ids=fin, morph_scale=scale, fourlimb=True)
            q = R4["optimized_4limb"]
            key = fourlimb_cache_key(holds, R4["weights"], R4["feasibility"], scale, R4["finish_ids"])
            store[key] = q
            if q:
                print(f"   feet plan scale {scale}: {time.time() - t0:.1f}s exact={q['search']['exact']} hand={q['n_hand_moves']} foot={q['n_foot_moves']} feet={q['feet_used']}")
            if scale == 1.0 and q:
                viz.render_comparison(img, holds, obs, q, cmp_, box, feet_obs=feet_for_result(a.get("foot_events", []), obs),
                                      feet_opt=q["feet_by_state"]).save(os.path.join(ROOT, "demo_assets", d["key"], "fourlimb.png"))
        json.dump(store, open(os.path.join(cache, "fourlimb.json"), "w"), indent=1)
        out = os.path.join(ROOT, "demo_assets", d["key"])
        viz.render_comparison(img, holds, obs, opt, cmp_, box).save(os.path.join(out, "comparison.png"))
        Rs = run_optimization(holds, a["morphology"], rest, feas=feas, start_state=st_, finish_ids=fin, morph_scale=0.85)
        l = viz.render_beta_panel(img, holds, opt, "You", viz.C_OPTIMIZED, box, crux=False)
        r = viz.render_beta_panel(img, holds, Rs["optimized"], "15 % shorter", (255, 120, 200), box, crux=False)
        from PIL import Image
        both = Image.new("RGB", (l.width + r.width + 12, max(l.height, r.height)), (18, 18, 22))
        both.paste(l, (0, 0)); both.paste(r, (l.width + 12, 0))
        both.save(os.path.join(out, "personalize.png"))
        summary = {
            "observed_moves": obs["n_moves"] if obs else None, "optimized_moves": opt["n_moves"] if opt else None,
            "observed_cost": obs["total_cost"] if obs else None, "optimized_cost": opt["total_cost"] if opt else None,
            "improvement_frac": cmp_.get("improvement_frac"), "crux": cmp_.get("crux", {}).get("explanation"),
            "simulated_85_moves": Rs["optimized"]["n_moves"] if Rs["optimized"] else None,
            "simulated_85_holds": Rs["optimized"]["holds_used"] if Rs["optimized"] else None,
            "optimized_holds": opt["holds_used"] if opt else None, "observed_holds": obs["holds_used"] if obs else None,
            "alignment": info, "start_state": list(st_) if st_ else None, "finish_ids": fin,
        }
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        print("   ", json.dumps(summary))


if __name__ == "__main__":
    main()
