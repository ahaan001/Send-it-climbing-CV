"""Synthetic benchmark: does morphology change the optimal beta?

Generates N random routes (hold grid with jitter and random grip ratings),
solves the hands-only problem exactly for reach scales {0.8, 0.85, 0.9, 1.0,
1.1} and reports how often the optimal hold set / sequence changes relative
to the measured (1.0) climber, the cost ratio, and how often the reach
envelope had to be relaxed (those routes are excluded from the change stats).

    python3 scripts/benchmark_morphology.py --n 200 --seed 0
Writes docs/benchmark_morphology.json and docs/benchmark_morphology.png.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sendit.optimizer import Climber, Weights, Feasibility, optimize_beta  # noqa: E402

ARM = 1000.0
SCALES = [0.8, 0.85, 0.9, 1.0, 1.1]


def random_route(rng: np.random.Generator, rows=7, cols=5, drop=0.25, jitter=0.25):
    """Grid of holds (0.32 arm horizontal, 0.28 arm vertical spacing), some cells
    removed, jittered positions, grips weighted toward 2-4. Start = matched
    hands on the bottom-centre hold, finish = the top-most hold."""
    dx, dy = 0.32 * ARM, 0.28 * ARM
    holds = []
    hid = 0
    for r in range(rows):
        for c in range(cols):
            if r not in (0, rows - 1) and rng.random() < drop:
                continue
            x = 500 + (c - cols // 2) * dx + rng.uniform(-jitter, jitter) * dx
            y = 1000 - r * dy + rng.uniform(-jitter, jitter) * dy
            grip = int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.25, 0.35, 0.2, 0.1]))
            holds.append({"id": hid, "x": float(x), "y": float(y), "grip": grip, "on_route": True, "role": None,
                          "source": "synthetic", "label": f"H{hid}"})
            hid += 1
    bottom = [h for h in holds if h["y"] > 1000 - 0.5 * dy]
    start = min(bottom, key=lambda h: abs(h["x"] - 500))
    finish = min(holds, key=lambda h: h["y"])
    start["role"], finish["role"] = "start", "finish"
    return holds, start["id"], finish["id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    W, F = Weights(), Feasibility()
    base = Climber(arm_span_px=ARM, leg_len_px=450, torso_px=300)
    per_scale = {s: {"n": 0, "holds_changed": 0, "seq_changed": 0, "relaxed": 0, "cost_ratio": [], "runtime": [], "expanded": []}
                 for s in SCALES}
    t_all = time.time()
    for i in range(args.n):
        holds, s_id, f_id = random_route(rng)
        ref = optimize_beta(holds, base, W, F, (s_id, s_id), {f_id}, relax=False)
        if ref is None:
            for s in SCALES:
                per_scale[s]["relaxed"] += 1
            continue
        for s in SCALES:
            t0 = time.time()
            res = optimize_beta(holds, base.scaled(s), W, F, (s_id, s_id), {f_id}, relax=False)
            st = per_scale[s]
            if res is None:
                st["relaxed"] += 1
                continue
            st["n"] += 1
            st["holds_changed"] += int(set(res["holds_used"]) != set(ref["holds_used"]))
            st["seq_changed"] += int(res["states"] != ref["states"])
            st["cost_ratio"].append(res["total_cost"] / ref["total_cost"])
            st["runtime"].append(time.time() - t0)
            st["expanded"].append(res["search"]["n_expanded"])
    summary = {}
    for s in SCALES:
        st = per_scale[s]
        n = max(1, st["n"])
        cr = np.array(st["cost_ratio"]) if st["cost_ratio"] else np.array([1.0])
        summary[str(s)] = {
            "routes_compared": st["n"], "pct_hold_set_changed": 100.0 * st["holds_changed"] / n,
            "pct_sequence_changed": 100.0 * st["seq_changed"] / n, "pct_needed_relaxation": 100.0 * st["relaxed"] / args.n,
            "cost_ratio_mean": float(cr.mean()), "cost_ratio_median": float(np.median(cr)),
            "cost_ratio_q25": float(np.percentile(cr, 25)), "cost_ratio_q75": float(np.percentile(cr, 75)),
            "runtime_ms_mean": 1000.0 * float(np.mean(st["runtime"])) if st["runtime"] else None,
            "states_expanded_mean": float(np.mean(st["expanded"])) if st["expanded"] else None,
        }
    out = {"n_routes": args.n, "seed": args.seed, "scales": SCALES, "weights": W.__dict__, "feasibility": F.__dict__,
           "total_runtime_s": time.time() - t_all, "per_scale": summary}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "benchmark_morphology.json"), "w"), indent=1)
    for s in SCALES:
        r = summary[str(s)]
        print(f"scale {s:>4}: n={r['routes_compared']:3d}  hold-set changed {r['pct_hold_set_changed']:5.1f}%  "
              f"sequence changed {r['pct_sequence_changed']:5.1f}%  cost ratio median {r['cost_ratio_median']:.2f} "
              f"(IQR {r['cost_ratio_q25']:.2f}-{r['cost_ratio_q75']:.2f})  relaxed {r['pct_needed_relaxation']:4.1f}%  "
              f"{(r['runtime_ms_mean'] or 0):.0f} ms")
    # ---- chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    fig.patch.set_facecolor("#111318")
    for ax in (a1, a2):
        ax.set_facecolor("#181b22")
        for sp in ax.spines.values():
            sp.set_color("#3a4050")
        ax.tick_params(colors="#dfe5ee")
        ax.xaxis.label.set_color("#dfe5ee"); ax.yaxis.label.set_color("#dfe5ee"); ax.title.set_color("#f5f7fa")
    xs = [f"{int(s * 100)} %" for s in SCALES]
    a1.bar(xs, [summary[str(s)]["pct_hold_set_changed"] for s in SCALES], color="#00c8ff", width=0.55, label="hold set changed")
    a1.bar(xs, [summary[str(s)]["pct_sequence_changed"] for s in SCALES], color="#ff8c00", width=0.3, label="sequence changed")
    a1.set_ylabel("% of routes vs measured climber"); a1.set_title("Optimal beta changes with morphology"); a1.legend(facecolor="#181b22", labelcolor="#dfe5ee")
    med = [summary[str(s)]["cost_ratio_median"] for s in SCALES]
    q25 = [summary[str(s)]["cost_ratio_q25"] for s in SCALES]
    q75 = [summary[str(s)]["cost_ratio_q75"] for s in SCALES]
    a2.fill_between(xs, q25, q75, color="#00c8ff", alpha=0.25, label="IQR")
    a2.plot(xs, med, "o-", color="#00c8ff", label="median")
    a2.axhline(1.0, color="#9aa3b2", lw=1, ls="--")
    a2.set_ylabel("optimal cost ÷ measured climber's"); a2.set_title("Same route costs more for a smaller climber"); a2.legend(facecolor="#181b22", labelcolor="#dfe5ee")
    fig.suptitle(f"{args.n} synthetic routes · hands-only exact A* · routes needing envelope relaxation excluded", color="#9aa3b2", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, "docs", "benchmark_morphology.png"), facecolor=fig.get_facecolor())
    print("saved docs/benchmark_morphology.png/json in %.0fs" % out["total_runtime_s"])


if __name__ == "__main__":
    main()
