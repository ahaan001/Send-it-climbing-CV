"""End-to-end analysis with on-disk caching.

analyze_video(video, cache_dir) produces, in cache_dir:
  pose.json        per-frame landmarks (MediaPipe)
  background.png   climber-free wall image (temporal median)
  first_frame.png
  analysis.json    morphology, auto-detected holds, observed placements
The optimization itself is milliseconds and is run on demand by the UI.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

import cv2
import numpy as np

from . import beta as beta_mod
from . import holds as holds_mod
from . import pose as pose_mod
from . import stabilize
from .optimizer import Climber, Weights, Feasibility, optimize_beta, score_sequence, compare, calibrate_envelope


def _np(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


def analyze_video(video_path: str, cache_dir: str, force: bool = False,
                  progress: Optional[Callable[[str, float], None]] = None,
                  contact_radius_frac: float = 0.12, min_contact_s: float = 0.25,
                  wall_type: str = "auto") -> dict:
    """wall_type: 'auto' | 'led' (lit Kilter/Moon board) | 'color' (normal wall)."""
    os.makedirs(cache_dir, exist_ok=True)
    pose_path = os.path.join(cache_dir, "pose.json")
    pose_wall_path = os.path.join(cache_dir, "pose_wall.json")
    stab_path = os.path.join(cache_dir, "stab.json")
    bg_path = os.path.join(cache_dir, "background.png")
    ff_path = os.path.join(cache_dir, "first_frame.png")
    an_path = os.path.join(cache_dir, "analysis.json")

    def report(stage, frac):
        if progress:
            progress(stage, frac)

    t0 = time.time()
    # 1. pose (original pixel coordinates)
    if not force and os.path.exists(pose_path):
        pose = pose_mod.load_pose(pose_path)
        report("pose (cached)", 1.0)
    else:
        report("pose", 0.0)
        pose = pose_mod.extract_pose(video_path, progress=lambda f: report("pose", f))
        pose_mod.save_pose(pose, pose_path)
    t_pose = time.time() - t0

    # 2. camera-motion compensation -> wall coordinates
    t1 = time.time()
    if not force and os.path.exists(stab_path):
        stab = stabilize.load(stab_path)
        report("stabilize (cached)", 1.0)
    else:
        report("stabilize", 0.0)
        stab = stabilize.compute_homographies(video_path, progress=lambda f: report("stabilize", f))
        stabilize.save(stab, stab_path)
    if stab["static"]:
        canvas_w, canvas_h, T = pose["w"], pose["h"], np.eye(3)
        stab_used = {**stab, "H": [np.eye(3).tolist()] * len(stab["H"])}
    else:
        canvas_w, canvas_h, T = stabilize.canvas_geometry(stab)
        stab_used = stab
    pose_wall = stabilize.stabilize_pose(pose, stab_used, T)
    pose_mod.save_pose(pose_wall, pose_wall_path)
    t_stab = time.time() - t1

    # 3. climber-free background in wall coordinates
    report("background", 0.0)
    if not force and os.path.exists(bg_path):
        bg = cv2.imread(bg_path)
    else:
        if stab["static"]:
            bg = holds_mod.background_frame(video_path)
        else:
            bg = stabilize.stabilized_background(video_path, stab_used, T, canvas_w, canvas_h)
        cv2.imwrite(bg_path, bg)
        cv2.imwrite(ff_path, holds_mod.first_frame(video_path))
    report("background", 1.0)

    morph = pose_mod.morphology_from_pose(pose_wall)

    # 4. holds
    report("holds", 0.0)
    led = holds_mod.detect_led_holds(video_path) if wall_type in ("auto", "led") else []
    if wall_type == "auto" and len(led) >= 5:
        # A lit board has a few small glowing rings on a dark panel: lit pixels are a tiny fraction of the
        # board mask (~0.4 % on our Kilter clip). A normal wall with colourful holds is several times higher.
        lit_frac = holds_mod.lit_fraction_in_board(video_path)
        auto_led = lit_frac is not None and lit_frac < 0.007
    else:
        auto_led = False
    if wall_type == "led" or auto_led:
        method = "led"
        H0 = T @ np.array(stab_used["H"][0], np.float64)
        for h in led:
            x, y = stabilize.warp_points(H0, [(h["x"], h["y"])])[0]
            h["x"], h["y"] = float(x), float(y)
        holds = led
    else:
        method = "color"
        holds = holds_mod.detect_color_holds(bg)
    holds = holds_mod.merge_close_holds(holds, 0.07 * morph["arm_span_px"])
    inferred = holds_mod.dwell_inferred_holds(pose_wall, morph["arm_span_px"], holds)
    if method == "led":
        for h in inferred:  # on a lit board the route IS the lit holds; keep suggestions off-route
            h["on_route"] = False
    holds = holds + inferred
    report("holds", 1.0)

    # 5. observed beta
    report("observed beta", 0.0)
    min_contact_frames = max(4, int(round(min_contact_s * pose["fps"])))
    events = beta_mod.contact_events(pose_wall, holds, morph["arm_span_px"], contact_radius_frac, min_contact_frames)
    placements = beta_mod.observed_placements(events)
    beta_mod.default_roles(holds, placements)
    hid = {h["id"]: h for h in holds}
    placements = beta_mod.truncate_at_finish(placements, [h["id"] for h in holds if h.get("role") == "finish"],
                                             morph["arm_span_px"], hid=hid)
    context = beta_mod.measured_move_context(pose_wall, placements)
    foot_events = beta_mod.observed_feet(pose_wall, holds, morph["arm_span_px"], contact_radius_frac, min_contact_frames)
    report("observed beta", 1.0)

    analysis = {
        "video": os.path.abspath(video_path),
        "w": pose["w"], "h": pose["h"], "fps": pose["fps"], "n_frames": pose["n_frames"],
        "canvas_w": int(canvas_w), "canvas_h": int(canvas_h), "T": np.asarray(T).tolist(),
        "camera": {"static": bool(stab["static"]), "max_shift_px": stab["max_shift_px"], "ref_idx": stab["ref_idx"]},
        "timing": {"pose_s": t_pose, "stabilize_s": t_stab, "total_s": time.time() - t0},
        "morphology": morph,
        "hold_method": method,
        "wall_type": wall_type,
        "holds": holds,
        "events": events,
        "foot_events": foot_events,
        "placements": placements,
        "move_context": context,
        "settings": {"contact_radius_frac": contact_radius_frac, "min_contact_frames": min_contact_frames},
        "cache_dir": os.path.abspath(cache_dir),
    }
    with open(an_path, "w") as f:
        json.dump(analysis, f, indent=1, default=_np)
    return analysis


def load_analysis(cache_dir: str) -> dict:
    with open(os.path.join(cache_dir, "analysis.json")) as f:
        return json.load(f)


def recompute_observed(analysis: dict, holds: list, pose: dict | None = None) -> list:
    """Re-run contact detection against an edited hold set (after manual correction)."""
    if pose is None:
        pose = pose_mod.load_pose(os.path.join(analysis["cache_dir"], "pose_wall.json"))
    s = analysis.get("settings", {})
    events = beta_mod.contact_events(pose, holds, analysis["morphology"]["arm_span_px"],
                                     s.get("contact_radius_frac", 0.12), s.get("min_contact_frames", 4))
    placements = beta_mod.observed_placements(events)
    hid = {h["id"]: h for h in holds}
    return beta_mod.truncate_at_finish(placements, [h["id"] for h in holds if h.get("role") == "finish"],
                                       analysis["morphology"]["arm_span_px"], hid=hid)


def recompute_feet(analysis: dict, holds: list, pose: dict | None = None) -> list:
    """Foot contact events against an edited hold set (measured, not suggested)."""
    if pose is None:
        pose = pose_mod.load_pose(os.path.join(analysis["cache_dir"], "pose_wall.json"))
    s = analysis.get("settings", {})
    return beta_mod.observed_feet(pose, holds, analysis["morphology"]["arm_span_px"],
                                  s.get("contact_radius_frac", 0.12), s.get("min_contact_frames", 4))


def run_optimization(holds: list, morphology: dict, placements: list,
                     weights: Weights | None = None, feas: Feasibility | None = None,
                     morph_scale: float = 1.0, calibrate: bool = True,
                     start_state=None, finish_ids=None, fourlimb: bool = False,
                     fourlimb_exact_threshold: int = 60_000, fourlimb_max_expansions: int = 150_000,
                     fourlimb_beam: int = 200) -> dict:
    """Score the observed beta and compute the optimized beta under the same
    objective for (optionally scaled) morphology. fourlimb=True additionally
    computes the hands+feet plan (result['optimized_4limb']); the hands-only
    comparison stays the headline."""
    W = weights or Weights()
    F = feas or Feasibility()
    hid = {h["id"]: h for h in holds}
    measured = Climber.from_morphology(morphology)
    climber = measured if morph_scale == 1.0 else measured.scaled(morph_scale)

    if start_state is None:
        st, consumed = beta_mod.initial_state(placements)
        rest = placements[consumed:] if st else []
        if st is None:
            starts = [h["id"] for h in holds if h.get("role") == "start" and h.get("on_route", True)]
            if not starts:
                route = [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]
                starts = [max(route, key=lambda h: h["y"])["id"]] if route else []
            st = (starts[0], starts[-1]) if starts else None
            rest = placements
    else:
        st, rest = tuple(start_state), placements
    if finish_ids is None:
        finish_ids = [h["id"] for h in holds if h.get("role") == "finish" and h.get("on_route", True)]
        if not finish_ids:
            route = [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]
            finish_ids = [min(route, key=lambda h: h["y"])["id"]] if route else []
    if st is None or not finish_ids:
        return {"error": "Need a start state and a finish hold."}

    # observed beta is always scored with the MEASURED climber (that's who climbed it)
    observed = score_sequence(holds, measured, W, F, st, rest) if rest else None
    F_used = calibrate_envelope(F, observed) if calibrate else F
    optimized = optimize_beta(holds, climber, W, F_used, st, finish_ids)
    geometric = optimize_beta(holds, climber, Weights.geometric(), F_used, st, finish_ids, label="Shortest geometric path")
    if geometric:  # re-score the geometric path with the real objective so costs are comparable
        geometric = score_sequence(holds, climber, W, F_used, st,
                                   [{"hand": m["hand"], "hold_id": m["to"]} for m in geometric["moves"]],
                                   label="Shortest geometric path")
    cmp_ = compare(observed, optimized, hid)
    out = {
        "climber": climber, "measured": measured, "weights": W, "feasibility": F_used,
        "start_state": list(st), "finish_ids": list(finish_ids),
        "observed": observed, "optimized": optimized, "geometric": geometric, "comparison": cmp_,
        "optimized_4limb": None,
    }
    if fourlimb:
        out["optimized_4limb"] = optimize_beta(holds, climber, W, F_used, st, finish_ids, label="Full-body plan",
                                               limbs="all", exact_threshold=fourlimb_exact_threshold,
                                               max_expansions=fourlimb_max_expansions, beam=fourlimb_beam)
    return out


def feet_for_result(foot_events: list, result: dict | None):
    """Observed feet aligned with an observed result's states: state 0 uses the
    first move's frame, state i uses move i's frame. None where no foot hold
    was detected (smearing, on the mat, not visible)."""
    if not result or not result.get("moves") or not foot_events:
        return None
    frames = [result["moves"][0].get("frame")] + [m.get("frame") for m in result["moves"]]
    out = []
    for f in frames:
        feet = [None, None]
        if f is not None:
            for e in foot_events:
                if e["start"] <= f <= e["end"] + 2:
                    feet[0 if e["limb"] == "LEFT_FOOT" else 1] = e["hold_id"]
        out.append(feet)
    return out


def fourlimb_cache_key(holds: list, weights: Weights, feas: Feasibility, morph_scale: float, finish_ids) -> str:
    """Stable key for precomputed four-limb results (same inputs -> same key)."""
    import hashlib
    from dataclasses import astuple
    payload = json.dumps({"holds": holds, "w": astuple(weights), "f": astuple(feas), "scale": morph_scale,
                          "finish": sorted(finish_ids)}, sort_keys=True, default=_np)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]
