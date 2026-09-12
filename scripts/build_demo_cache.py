"""(Re)build every demo's cached analysis + presentation assets.

    python3 scripts/build_demo_cache.py [--force]

Writes into demo_assets/<key>/cache/: pose.json, pose_wall.json, stab.json,
background.png, analysis.json, overlay.mp4, and into demo_assets/<key>/:
comparison.png, diff.png, graph.png, personalize.png (emergency fallback).
"""
import argparse
import json
import os
import sys

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sendit import viz  # noqa: E402
from sendit.optimizer import hold_graph_edges  # noqa: E402
from sendit.pipeline import analyze_video, run_optimization, recompute_observed  # noqa: E402
from sendit.pose import load_pose  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    demos = json.load(open(os.path.join(ROOT, "demo_assets", "demos.json")))["demos"]
    for d in demos:
        if args.only and d["key"] != args.only:
            continue
        video = os.path.join(ROOT, d["video"])
        cache = os.path.join(ROOT, d["cache"])
        print(f"== {d['key']}")
        a = analyze_video(video, cache, force=args.force, wall_type=d.get("wall_type", "auto"),
                          progress=lambda s, f: None)
        print(f"   pose {a['timing']['pose_s']:.1f}s  total {a['timing']['total_s']:.1f}s  holds {len(a['holds'])}  placements {len(a['placements'])}")
        pose = load_pose(os.path.join(cache, "pose.json"))
        pose_wall = load_pose(os.path.join(cache, "pose_wall.json"))
        ov = os.path.join(cache, "overlay.mp4")
        if args.force or not os.path.exists(ov):
            viz.render_pose_overlay_video(video, pose, ov)
            print("   overlay video ->", ov)
        holds = a["holds"]
        cur = os.path.join(cache, "holds_edited.json")
        if os.path.exists(cur):
            holds = json.load(open(cur))
            placements = recompute_observed(a, holds, pose_wall)
            print(f"   using curated holds ({len(holds)}), placements {len(placements)}")
        else:
            placements = a["placements"]
        bg = cv2.imread(os.path.join(cache, "background.png"))
        box = viz.crop_box(holds, bg.shape[1], bg.shape[0])
        R = run_optimization(holds, a["morphology"], placements)
        obs, opt, cmp_ = R["observed"], R["optimized"], R["comparison"]
        out = os.path.join(ROOT, "demo_assets", d["key"])
        viz.render_comparison(bg, holds, obs, opt, cmp_, box).save(os.path.join(out, "comparison.png"))
        viz.render_diff(bg, holds, obs, opt, cmp_, box).save(os.path.join(out, "diff.png"))
        viz.render_graph(bg, holds, hold_graph_edges(holds, R["climber"], R["feasibility"]), opt, box).save(os.path.join(out, "graph.png"))
        Rs = run_optimization(holds, a["morphology"], placements, morph_scale=0.85)
        l = viz.render_beta_panel(bg, holds, opt, "MEASURED climber · optimized beta", viz.C_OPTIMIZED, box, crux=False)
        r = viz.render_beta_panel(bg, holds, Rs["optimized"], "SIMULATED 85% reach · optimized beta", (255, 120, 200), box, crux=False)
        from PIL import Image
        both = Image.new("RGB", (l.width + r.width + 12, max(l.height, r.height)), (18, 18, 22))
        both.paste(l, (0, 0)); both.paste(r, (l.width + 12, 0))
        both.save(os.path.join(out, "personalize.png"))
        summary = {
            "observed_cost": obs["total_cost"] if obs else None, "observed_moves": obs["n_moves"] if obs else None,
            "optimized_cost": opt["total_cost"], "optimized_moves": opt["n_moves"],
            "improvement_frac": cmp_.get("improvement_frac"), "crux": cmp_.get("crux", {}).get("explanation"),
            "simulated_85_cost": Rs["optimized"]["total_cost"], "simulated_85_moves": Rs["optimized"]["n_moves"],
            "simulated_85_holds": Rs["optimized"]["holds_used"], "optimized_holds": opt["holds_used"],
            "observed_holds": obs["holds_used"] if obs else None,
        }
        json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
        print("   ", json.dumps(summary))


if __name__ == "__main__":
    main()
