"""Concise coaching insights derived from the structured optimization output.

The rule-based generator needs no network and is what the demo uses. An
optional LLM rewrite (xAI Grok or Google Gemini, whichever key is present)
only paraphrases the same structured facts -- it never chooses the beta.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .optimizer import LEFT


def _lab(hid, i):
    return hid.get(i, {}).get("label", f"H{i}")


def _hand(h):
    return "left" if h == LEFT else "right"


def rule_based(R: dict, hid: dict, partial: bool = False) -> list[str]:
    obs, opt, cmp_ = R.get("observed"), R.get("optimized"), R.get("comparison", {})
    out = []
    if not opt:
        return ["No feasible beta was found; check start/finish roles and route membership."]
    if obs and obs["moves"]:
        cx = cmp_.get("crux")
        if cx:
            drivers = ", ".join(f"{k} {v:.2f}" for k, v in cx["drivers"])
            out.append(f"Your highest-cost move under our model was the {_hand(cx['hand'])} hand {_lab(hid, cx['from'])} → "
                       f"{_lab(hid, cx['to'])}: a {cx['reach_frac']:.0%}-of-arm-span reach"
                       + (f" to a hold you rated {cx['grip']}/5" if cx['grip'] >= 4 else "") + f" (cost drivers: {drivers}).")
        if partial:
            out.append("The clip ends before the finish, so the optimizer's route beyond your last hold is a plan, not a comparison.")
        elif cmp_.get("same_sequence"):
            out.append("Your sequence already matches the minimum-cost beta for your morphology under this objective. Nice.")
        elif cmp_.get("same_holds"):
            out.append("You used the optimizer's holds, but with a different hand order; the ordering change lowers crossed-hand and travel cost.")
        else:
            drop = cmp_.get("only_observed", [])
            add = cmp_.get("only_optimized", [])
            imp = cmp_.get("improvement_frac", 0.0)
            s = f"The optimizer lowers total movement cost by {imp:.0%} ({obs['total_cost']:.1f} → {opt['total_cost']:.1f}) with {opt['n_moves']} hand moves instead of {obs['n_moves']}"
            if drop:
                s += f", skipping {', '.join(_lab(hid, i) for i in drop)}"
            if add:
                s += f" and using {', '.join(_lab(hid, i) for i in add)} instead"
            out.append(s + ".")
            if obs["max_reach_frac"] > opt["max_reach_frac"] + 0.05:
                out.append(f"Peak reach drops from {obs['max_reach_frac']:.0%} to {opt['max_reach_frac']:.0%} of your arm span: "
                           f"the recommended line trades one long lock-off for moderate, repeatable moves.")
            if obs.get("mean_grip") and opt.get("mean_grip") and opt["mean_grip"] < obs["mean_grip"] - 0.2:
                out.append(f"Average grip rating improves from {obs['mean_grip']:.1f} to {opt['mean_grip']:.1f} (lower is better).")
    else:
        out.append(f"Recommended beta: {opt['n_moves']} hand moves, max reach {opt['max_reach_frac']:.0%} of arm span.")
    # biggest single move in the optimized beta
    if opt["moves"]:
        big = max(opt["moves"], key=lambda m: m["reach_frac"])
        out.append(f"Hardest move in the recommended beta: {_hand(big['hand'])} hand {_lab(hid, big['from'])} → {_lab(hid, big['to'])} "
                   f"at {big['reach_frac']:.0%} of arm span; set your feet before it.")
    return out


def summary_for_llm(R: dict, hid: dict, partial: bool) -> dict:
    obs, opt, cmp_ = R.get("observed"), R.get("optimized"), R.get("comparison", {})

    def seq(res):
        if not res:
            return None
        return [{"hand": _hand(m["hand"]), "from": _lab(hid, m["from"]), "to": _lab(hid, m["to"]),
                 "reach_frac_of_arm_span": round(m["reach_frac"], 2), "grip_rating_of_target": m["grip"],
                 "cost": round(m["cost"], 2), "cost_terms": {k: round(v, 2) for k, v in m["terms"].items()}} for m in res["moves"]]
    return {
        "climber": {"label": R["climber"].label, "scale_vs_measured": R["climber"].scale},
        "observed": {"total_cost": round(obs["total_cost"], 2), "moves": seq(obs)} if obs else None,
        "optimized": {"total_cost": round(opt["total_cost"], 2), "moves": seq(opt)} if opt else None,
        "improvement_frac": cmp_.get("improvement_frac"), "partial_clip": partial,
        "crux": cmp_.get("crux", {}).get("explanation"),
        "objective": "reach (quadratic, normalized by arm span) + grip rating + per-move + travel + direction + crossed hands + foot support",
    }


def llm_rewrite(summary: dict, timeout: float = 20.0) -> Optional[str]:
    """Optional. Uses XAI_API_KEY (Grok) or GEMINI_API_KEY if set. Returns None otherwise or on any error."""
    prompt = ("You are a climbing coach. Using ONLY the structured analysis below (do not invent holds or numbers), "
              "write 3 short coaching sentences for the climber: what their costliest move was and why, what the optimizer "
              "recommends instead and why it is cheaper for THEIR body, and one practical cue. Plain text.\n\n"
              + json.dumps(summary, indent=1))
    try:
        import requests
        if os.environ.get("XAI_API_KEY"):
            r = requests.post("https://api.x.ai/v1/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"},
                              json={"model": os.environ.get("XAI_MODEL", "grok-4-fast"), "messages": [{"role": "user", "content": prompt}], "temperature": 0.3},
                              timeout=timeout)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        if os.environ.get("GEMINI_API_KEY"):
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
            r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                              params={"key": os.environ["GEMINI_API_KEY"]},
                              json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=timeout)
            r.raise_for_status()
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None
    return None
