"""Coaching tips built from the optimizer's structured output.

rule_based() and why_block() need no network and are what the demo shows.
summary_for_llm() packs the same facts, plus what the video shows about hips,
feet and timing, into one dict; llm_insights() asks Grok (xAI) or Gemini to
write per-move coaching from it. The LLM never chooses the sequence.

Every user-facing string here is written for a recreational climber: short
cues, hold labels and move numbers, no model vocabulary.
"""
from __future__ import annotations

import json
import math
import os
import re
from typing import Optional

from .holds import GRIP_LABELS
from .optimizer import HANDS, LEFT, RIGHT, limb_name
from .pose import hand_point

CONTACT_RADIUS_FRAC = 0.12   # same default as beta.contact_events
CROSS_FLAG = 0.3             # move["cross"] above this counts as crossed hands
_DRIVER_WORDS = {"reach": "a long reach", "grip": "a poor hold", "cross": "crossed hands",
                 "foot": "no foot support", "travel": "a long move", "direction": "a long move",
                 "hang": "feet off the wall"}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


# --------------------------------------------------------------------------- small helpers
def _lab(hid: dict, i) -> str:
    if i is None:
        return "no hold"
    return hid.get(i, {}).get("label", f"H{i}")


def _hand(h) -> str:
    return "left" if h == LEFT else "right"


def _num(n: int) -> str:
    """Spell a count so tips carry no digits except hold labels and move numbers."""
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    return str(n)


def _join(labels: list) -> str:
    labels = list(labels)
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _reach_word(frac: float) -> str:
    if frac < 0.45:
        return "a short reach"
    if frac < 0.6:
        return "a medium reach"
    if frac < 0.75:
        return "a long reach"
    return "a very long reach"


def _ease_word(imp: float) -> str:
    if imp >= 0.5:
        return "much easier"
    if imp >= 0.2:
        return "easier"
    if imp > 0.05:
        return "a little easier"
    return "about as hard"


def _words(s: str) -> int:
    return len(s.split())


def _fit(build, ids: list, hid: dict, limit: int = 15) -> str:
    """build(label_text) -> sentence; shrink the hold list until the sentence fits."""
    labels = [_lab(hid, i) for i in ids]
    for n in range(len(labels), 0, -1):
        part = labels[:n] + (["others"] if n < len(labels) else [])
        s = build(_join(part))
        if _words(s) <= limit:
            return s
    return build(labels[0] if labels else "")


def _crux_terms(cx: dict) -> list:
    """Driver names of the crux that really explain it, biggest first. A
    driver only counts when it is big in climbing terms (same gates as
    optimizer.crux_explanation), so a 0.3-span reach is never 'long'."""
    gates = {"reach": lambda v: float(cx.get("reach_frac", 0.0)) >= 0.45,
             "grip": lambda v: int(cx.get("grip", 3)) >= 4,
             "cross": lambda v: v > 0.1, "foot": lambda v: v > 0.1, "hang": lambda v: v > 0.1,
             "direction": lambda v: v > 0.15, "travel": lambda v: v > 0.15}
    return [t for t, v in cx.get("drivers", []) if t in gates and gates[t](float(v))]


def _crux_reason(cx: dict) -> str:
    """Single biggest reason for the crux, from its drivers."""
    terms = _crux_terms(cx)
    return _DRIVER_WORDS[terms[0]] if terms else "a mix of small things"


def _crux_phrase(cx: dict) -> str:
    """Up to two reasons, for the tips: 'a long reach to a poor hold'."""
    terms = _crux_terms(cx)
    if "reach" in terms and "grip" in terms:
        return "a long reach to a poor hold"
    seen = []
    for t in terms:
        if _DRIVER_WORDS[t] not in seen:
            seen.append(_DRIVER_WORDS[t])
    if not seen:
        return "a mix of small things"
    return " with ".join(seen[:2])


def _hand_moves(res: Optional[dict]) -> list:
    if not res:
        return []
    return [(i, m) for i, m in enumerate(res.get("moves", [])) if (m.get("limb") or m.get("hand")) in HANDS]


# --------------------------------------------------------------------------- rule-based tips
def rule_based(R: dict, hid: dict, partial: bool = False) -> list[str]:
    """Short cue-style tips, one per line. Numbers appear only as move numbers
    and hold labels."""
    obs, opt, cmp_ = R.get("observed"), R.get("optimized"), R.get("comparison") or {}
    out: list[str] = []
    if not opt:
        return ["We could not find a line to the top. Check the start and finish holds, "
                "and which holds are on your climb."]
    opt_holds = set(opt.get("holds_used", []))
    have_obs = bool(obs and obs.get("moves"))
    if have_obs:
        cx = cmp_.get("crux")
        if cx:
            who = limb_name(cx.get("limb") or cx.get("hand"))
            out.append(f"Move {cx['move_index'] + 1} was your hardest: {who} {_lab(hid, cx['from'])} to "
                       f"{_lab(hid, cx['to'])}, {_crux_phrase(cx)}.")
            same = [m for _i, m in _hand_moves(opt) if m["from"] == cx["from"] and m["to"] == cx["to"]]
            alt = [m for _i, m in _hand_moves(opt) if m["to"] == cx["to"] and m["from"] != cx["from"]]
            if same:
                out.append("The suggested line keeps this move. Set your feet before it.")
            elif alt:
                a = alt[0]
                if a.get("reach_frac", 1.0) < cx.get("reach_frac", 0.0) - 0.02:
                    out.append(f"Try {_lab(hid, a['from'])} first to shorten it.")
                else:
                    out.append(f"Try {_lab(hid, a['from'])} first.")
            elif cx["to"] not in opt_holds:
                out.append(f"The suggested line skips {_lab(hid, cx['to'])}.")
        if partial:
            out.append("The clip ends before the top. Past your last hold, the suggested line is a plan, "
                       "not a comparison.")
        elif cmp_.get("same_sequence"):
            out.append("Your moves already match the suggested line. Nice.")
        elif cmp_.get("same_holds"):
            out.append("Same holds as the suggested line, but a different hand order. "
                       "The suggested order means shorter moves and fewer crossed hands.")
        else:
            n_o, n_p = int(obs["n_moves"]), int(opt["n_moves"])
            if n_o != n_p:
                s = f"You used {_num(n_o)} moves. We suggest {_num(n_p)}."
            else:
                s = "Same number of moves, but different holds."
            drop = [_lab(hid, i) for i in cmp_.get("only_observed", [])]
            add = [_lab(hid, i) for i in cmp_.get("only_optimized", [])]
            if drop and add:
                s += f" Skip {_join(drop)} and use {_join(add)} instead."
            elif drop:
                s += f" Skip {_join(drop)}."
            elif add:
                s += f" Add {_join(add)}."
            out.append(s)
            out.append(f"Overall the suggested line is {_ease_word(cmp_.get('improvement_frac', 0.0))}.")
            if obs["max_reach_frac"] > opt["max_reach_frac"] + 0.05:
                i, big = max(_hand_moves(obs), key=lambda im: im[1]["reach_frac"])
                out.append(f"Your longest reach was move {i + 1}, {_hand(big['hand'])} hand "
                           f"{_lab(hid, big['from'])} to {_lab(hid, big['to'])}. "
                           "The suggested line keeps every reach shorter.")
            if obs.get("mean_grip") and opt.get("mean_grip") and opt["mean_grip"] < obs["mean_grip"] - 0.2:
                poor = [h for h in obs.get("holds_used", []) if h in hid and hid[h].get("grip", 3) >= 4
                        and h not in opt_holds]
                if len(poor) == 1:
                    out.append(f"{_lab(hid, poor[0])} is a {GRIP_LABELS[hid[poor[0]]['grip']]} hold. "
                               "The suggested line avoids it.")
                elif poor:
                    out.append(f"{_join([_lab(hid, h) for h in poor])} are poor holds. "
                               "The suggested line avoids them.")
                else:
                    out.append("The suggested line uses better holds.")
    else:
        out.append(f"Suggested line: {_num(opt['n_moves'])} moves.")
    hm = _hand_moves(opt)
    if hm:
        i, big = max(hm, key=lambda im: im[1]["reach_frac"])
        out.append(f"Hardest move on the suggested line: move {i + 1}, {_hand(big['hand'])} hand "
                   f"{_lab(hid, big['from'])} to {_lab(hid, big['to'])}, {_reach_word(big['reach_frac'])}. "
                   f"Set your feet before move {i + 1}.")
    return out


def why_block(R: dict, hid: dict, partial: bool = False) -> list[str]:
    """At most three sentences, each at most fifteen words: which holds the
    suggestion skips, which it uses instead, and the biggest reason for the crux."""
    obs, opt, cmp_ = R.get("observed"), R.get("optimized"), R.get("comparison") or {}
    if not obs or not obs.get("moves"):
        return ["We could not see your moves, so there is nothing to compare yet."]
    if not opt:
        return ["We could not find a suggested line to compare with."]
    out: list[str] = []
    if cmp_.get("same_sequence"):
        out.append("Your moves already match the suggested line.")
    elif cmp_.get("same_holds"):
        out.append("You used the same holds as the suggested line, in a different order.")
    else:
        drop, add = cmp_.get("only_observed", []), cmp_.get("only_optimized", [])
        if drop:
            out.append(_fit(lambda L: f"The suggested line skips {L}, which you used.", drop, hid))
        if add and not partial:
            out.append(_fit(lambda L: f"It uses {L} instead.", add, hid))
    cx = cmp_.get("crux")
    if cx:
        who = limb_name(cx.get("limb") or cx.get("hand"))
        s = (f"Move {cx['move_index'] + 1}, {who} {_lab(hid, cx['from'])} to {_lab(hid, cx['to'])}, "
             f"was hardest because of {_crux_reason(cx)}.")
        if _words(s) > 15:
            s = f"Move {cx['move_index'] + 1} was hardest because of {_crux_reason(cx)}."
        out.append(s)
    if partial:
        out.append("The clip ends early, so past your last hold this is a plan.")
    return out[:3]


# --------------------------------------------------------------------------- video-derived facts
def _frame(frames: dict, f):
    fr = frames.get(f)
    if fr is None and f is not None:
        fr = frames.get(str(f))
    return fr


def _hips_at(pose: Optional[dict], frame, min_vis: float = 0.5, search: int = 3):
    """Hip midpoint (x, y) at `frame`, or the nearest frame within +-search
    where both hips are visible. Returns (point|None, frame_used|None)."""
    if not pose or frame is None:
        return None, None
    frames = pose.get("frames", {})
    for d in range(0, search + 1):
        for f in ((frame - d, frame + d) if d else (frame,)):
            fr = _frame(frames, f)
            if not fr:
                continue
            lh, rh = fr.get("LEFT_HIP"), fr.get("RIGHT_HIP")
            if lh and rh and min(lh[2], rh[2]) >= min_vis:
                return ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0), f
    return None, None


def _move_start_frame(pose: Optional[dict], m: dict, hid: dict, arm_span: Optional[float],
                      prev_frame, min_vis: float = 0.5, lookback: int = 90):
    """Last frame before the hand lands where it was still on its previous
    hold: the moment the move starts. Falls back to the landing frame."""
    f_arr = m.get("frame")
    if f_arr is None:
        return None
    if not pose or arm_span is None or m.get("from") not in hid or (m.get("hand") not in HANDS):
        return f_arr
    frames = pose.get("frames", {})
    src = hid[m["from"]]
    radius = CONTACT_RADIUS_FRAC * arm_span
    lo = prev_frame if prev_frame is not None else max(0, f_arr - lookback)
    for f in range(f_arr - 1, lo - 1, -1):
        fr = _frame(frames, f)
        if not fr:
            continue
        hp = hand_point(fr, m["hand"], min_vis)
        if hp is not None and math.hypot(hp[0] - src["x"], hp[1] - src["y"]) <= radius:
            return f
    return f_arr


def _direction(hf: Optional[dict], ht: Optional[dict], arm_span: Optional[float]):
    if not hf or not ht:
        return None
    tol = 0.05 * arm_span if arm_span else 20.0
    dx, dy = ht["x"] - hf["x"], ht["y"] - hf["y"]   # image y grows downward
    vertical = "up" if dy < -tol else ("down" if dy > tol else "sideways")
    lateral = "right" if dx > tol else ("left" if dx < -tol else "straight")
    return {"vertical": vertical, "lateral": lateral}


def _side_vs_hips(hold: Optional[dict], hips, arm_span: Optional[float]):
    if not hold or hips is None:
        return None
    tol = 0.08 * arm_span if arm_span else 30.0
    dx = hold["x"] - hips[0]
    return "under" if abs(dx) <= tol else ("right" if dx > 0 else "left")


def _arm_span_px(R: dict) -> Optional[float]:
    for key in ("measured", "climber"):
        c = R.get(key)
        if c is None:
            continue
        a = c.get("arm_span_px") if isinstance(c, dict) else getattr(c, "arm_span_px", None)
        if a:
            return float(a)
    for res in (R.get("observed"), R.get("optimized")):
        if res and res.get("climber") and res["climber"].get("arm_span_px"):
            return float(res["climber"]["arm_span_px"])
    return None


def _align_context(move_context: Optional[list], n: int) -> list:
    """beta.measured_move_context() has one entry per placement after the
    first; the scored line drops the placements that formed the start pair,
    so the contexts line up with the moves from the tail."""
    mc = list(move_context or [])
    if len(mc) >= n:
        return mc[len(mc) - n:]
    return [None] * (n - len(mc)) + mc


def tracking_gaps(pose: dict, fps: float, n_frames: int, min_vis: float = 0.5,
                  min_gap_s: float = 0.5) -> list[tuple[float, float]]:
    """Time ranges (seconds) where the climber was not tracked: the frame is
    missing from pose["frames"] or both wrists are below min_vis. Runs split
    by a single tracked frame are merged; gaps shorter than min_gap_s are dropped."""
    frames = (pose or {}).get("frames", {})
    fps = float(fps or 30.0)

    def tracked(f):
        fr = _frame(frames, f)
        if not fr:
            return False
        return any(p is not None and len(p) > 2 and p[2] >= min_vis
                   for p in (fr.get("LEFT_WRIST"), fr.get("RIGHT_WRIST")))

    runs = []
    cur = None
    for f in range(int(n_frames or 0)):
        if tracked(f):
            if cur is not None:
                runs.append(cur)
                cur = None
        elif cur is None:
            cur = [f, f + 1]
        else:
            cur[1] = f + 1
    if cur is not None:
        runs.append(cur)
    merged: list = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 1:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(round(s / fps, 2), round(e / fps, 2)) for s, e in merged if (e - s) / fps >= min_gap_s]


def gaps_text(gaps) -> str:
    return ", ".join(f"{s:.1f}\u2013{e:.1f} s" for s, e in (gaps or []))


# --------------------------------------------------------------------------- LLM summary
HOW_TO_READ = (
    "Holds are labelled H<id>. Moves are numbered from 1 inside each line. 'hand' is the hand that moves, "
    "'support_hand' stays on the wall. reach_frac_of_arm_span is the hand-to-hand span after the move as a "
    "fraction of this climber's arm span (0.7 and up is a big reach). hips_vs_support_hand is where the hips "
    "were just before the move, in arm spans, relative to the support hand (+ = to the right). feet says "
    "which hold each foot was on, if the video showed it, and whether that hold was left of, right of, or "
    "under the hips. support_elbow_min_deg is the support arm's most bent angle during the move (a straight "
    "arm rests, a bent arm tires). pose_missing means the video did not show the body at that moment: say "
    "'we could not see'. Directions are already up/down/left/right as the climber sees the wall."
)


def _move_entry(i: int, m: dict, hid: dict, arm_span: Optional[float]) -> dict:
    limb = m.get("limb") or m.get("hand")
    hf, ht = hid.get(m.get("from")), hid.get(m.get("to"))
    d = {"move": i + 1, "limb": limb_name(limb)}
    if limb in HANDS:
        d["hand"] = _hand(limb)
        d["support_hand"] = _hand(RIGHT if limb == LEFT else LEFT)
        d["support_hand_on"] = _lab(hid, m.get("other"))
    d["from"] = _lab(hid, m.get("from"))
    d["to"] = _lab(hid, m.get("to"))
    d["from_role"] = hf.get("role") if hf else None
    d["to_role"] = ht.get("role") if ht else None
    d["direction"] = _direction(hf, ht, arm_span)
    d["reach_frac_of_arm_span"] = round(float(m.get("reach_frac", 0.0)), 2)
    d["crossed_hands"] = bool(float(m.get("cross", 0.0)) > CROSS_FLAG)
    g = m.get("grip") if m.get("grip") else (ht.get("grip", 3) if ht else None)
    d["target_hold_rating"] = int(g) if g else None
    d["target_hold_quality"] = GRIP_LABELS.get(int(g)) if g else None
    return d


def summary_for_llm(R: dict, hid: dict, partial: bool, *, move_context=None, foot_events=None, pose=None,
                    fps=None, board_angle_deg=None, arm_span_m=None, height_m=None, injury=None,
                    gaps=None) -> dict:
    """Everything the coach model may use: both lines move by move, what the
    video showed about hips, feet and timing before each of the climber's
    moves, the comparison, and the tracking gaps."""
    from .beta import feet_at_placements

    obs, opt, cmp_ = R.get("observed"), R.get("optimized"), R.get("comparison") or {}
    A = _arm_span_px(R)
    fps = float(fps) if fps else (float(pose.get("fps")) if pose and pose.get("fps") else None)
    if gaps is None and pose and fps:
        n_frames = pose.get("n_frames") or (max(int(k) for k in pose.get("frames", {})) + 1 if pose.get("frames") else 0)
        gaps = tracking_gaps(pose, fps, n_frames)
    gaps = [(float(s), float(e)) for s, e in (gaps or [])]

    your = None
    if obs and obs.get("moves"):
        moves = obs["moves"]
        ctxs = _align_context(move_context, len(moves))
        entries = []
        prev_frame = None
        for i, m in enumerate(moves):
            d = _move_entry(i, m, hid, A)
            ctx = ctxs[i] or {}
            f_arr = m.get("frame")
            start_f = _move_start_frame(pose, m, hid, A, prev_frame)
            hips, hip_f = _hips_at(pose, start_f)
            d["pose_missing"] = hips is None
            if (m.get("hand") in HANDS):
                sup = hid.get(m.get("other"))
                if hips is not None and sup is not None and A:
                    off = (hips[0] - sup["x"]) / A
                    side = "under" if abs(off) < 0.03 else ("right" if off > 0 else "left")
                    d["hips_offset_frac"] = round(off, 2)
                    d["hips_vs_support_hand"] = f"{off:+.2f} {side}"
                else:
                    d["hips_offset_frac"] = None
                    d["hips_vs_support_hand"] = None
                feet = {}
                if foot_events and start_f is not None:
                    fa = feet_at_placements(foot_events, [{"frame": start_f}])[0]
                    for foot, key in (("LEFT_FOOT", "left"), ("RIGHT_FOOT", "right")):
                        fid = fa.get(foot)
                        feet[key] = ({"hold": _lab(hid, fid), "side_vs_hips": _side_vs_hips(hid.get(fid), hips, A)}
                                     if fid is not None else None)
                else:
                    feet = {"left": None, "right": None}
                d["feet"] = feet
                e = ctx.get("support_elbow_min_deg")
                d["support_elbow_min_deg"] = int(round(e)) if e is not None else None
                dur = None
                if ctx.get("duration_frames") is not None and fps:
                    dur = ctx["duration_frames"] / fps
                elif f_arr is not None and prev_frame is not None and fps:
                    dur = (f_arr - prev_frame) / fps
                d["duration_s"] = round(dur, 1) if dur is not None else None
                d["time_s"] = round(f_arr / fps, 1) if (f_arr is not None and fps) else None
            entries.append(d)
            if f_arr is not None:
                prev_frame = f_arr
        frames = [m.get("frame") for m in moves if m.get("frame") is not None]
        your = {"n_moves": int(obs["n_moves"]), "moves": entries,
                "holds_used": [_lab(hid, h) for h in obs.get("holds_used", [])],
                "total_time_s": round((max(frames) - min(frames)) / fps, 1) if (len(frames) > 1 and fps) else None}

    sugg = None
    if opt and opt.get("moves"):
        sugg = {"n_moves": int(opt["n_moves"]), "moves": [_move_entry(i, m, hid, A) for i, m in enumerate(opt["moves"])],
                "holds_used": [_lab(hid, h) for h in opt.get("holds_used", [])]}

    comparison = None
    if cmp_:
        cx = cmp_.get("crux")
        crux = None
        if cx:
            who = limb_name(cx.get("limb") or cx.get("hand"))
            crux = {"move": cx["move_index"] + 1, "limb": who, "from": _lab(hid, cx["from"]), "to": _lab(hid, cx["to"]),
                    "reach_frac_of_arm_span": round(float(cx.get("reach_frac", 0.0)), 2),
                    "reason": _crux_reason(cx),
                    "explanation": f"Move {cx['move_index'] + 1} ({who} {_lab(hid, cx['from'])} to "
                                   f"{_lab(hid, cx['to'])}) was the hardest move: {_crux_phrase(cx)}."}
        comparison = {
            "holds_only_in_your_climb": [_lab(hid, i) for i in cmp_.get("only_observed", [])],
            "holds_only_in_suggested": [_lab(hid, i) for i in cmp_.get("only_optimized", [])],
            "shared_holds": [_lab(hid, i) for i in cmp_.get("shared_holds", [])],
            "improvement_frac": (round(float(cmp_["improvement_frac"]), 2)
                                 if cmp_.get("improvement_frac") is not None else None),
            "same_sequence": bool(cmp_.get("same_sequence", False)),
            "same_holds": bool(cmp_.get("same_holds", False)),
            "crux": crux,
        }

    climber = {}
    if board_angle_deg is not None:
        climber["board_angle_deg"] = float(board_angle_deg)
    if arm_span_m is not None:
        climber["arm_span_m"] = float(arm_span_m)
    if height_m is not None:
        climber["height_m"] = float(height_m)
    c = R.get("climber")
    scale = c.get("scale") if isinstance(c, dict) else getattr(c, "scale", None)
    if scale is not None and abs(float(scale) - 1.0) > 1e-6:
        climber["reach_scale_vs_measured"] = round(float(scale), 2)

    return {
        "how_to_read": HOW_TO_READ,
        "climber": climber,
        "injury": injury,
        "partial_clip": bool(partial),
        "tracking_gaps": [[s, e] for s, e in gaps],
        "tracking_gaps_text": gaps_text(gaps),
        "your_climb": your,
        "suggested_line": sugg,
        "comparison": comparison,
    }


# --------------------------------------------------------------------------- LLM plumbing
_LLM: dict = {"provider": None, "model": None}
_XAI_MODELS = "https://api.x.ai/v1/models"
_XAI_CHAT = "https://api.x.ai/v1/chat/completions"
_GEMINI_MODELS = "https://generativelanguage.googleapis.com/v1beta/models"

SYSTEM_PROMPT = ("You are an experienced climbing coach talking to a recreational climber. Plain words, short "
                 "sentences. You never invent holds, moves or numbers; you only use what the analysis gives you.")


class LLMStatus(dict):
    """Result of llm_available(). A plain dict, except that its truth value is
    the 'ok' flag so `if llm_available():` keeps working."""

    def __bool__(self):
        return bool(self.get("ok"))


def load_env(path: str) -> None:
    """Tiny .env loader (KEY=VALUE lines, # comments). Never overrides a variable already set."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k and v and k not in os.environ:
            os.environ[k] = v


def _scrub(text: str) -> str:
    """Make sure no key value ever reaches the terminal."""
    for name in ("XAI_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(name)
        if v and v in text:
            text = text.replace(v, "***")
    return text


def _fail(provider, reason) -> LLMStatus:
    return LLMStatus(ok=False, provider=provider, model=None, reason=_scrub(str(reason)))


def _probe(timeout: float) -> LLMStatus:
    key = os.environ.get("XAI_API_KEY")
    if key:
        try:
            import requests
            r = requests.get(_XAI_MODELS, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
            if not r.ok:
                return _fail("xai", f"HTTP {r.status_code}: {r.text[:200]}")
            ids = [m.get("id") for m in (r.json().get("data") or []) if m.get("id")]
            want = os.environ.get("XAI_MODEL")
            model = want if want in ids else next(
                (i for i in ids if "grok" in i.lower() and "image" not in i.lower() and "vision" not in i.lower()), None)
            if not model:
                return _fail("xai", f"no grok chat model in the list: {ids[:12]}")
            return LLMStatus(ok=True, provider="xai", model=model, reason="")
        except Exception as e:  # noqa: BLE001
            return _fail("xai", f"{type(e).__name__}: {e}")
    gkey = os.environ.get("GEMINI_API_KEY")
    if gkey:
        try:
            import requests
            r = requests.get(_GEMINI_MODELS, params={"key": gkey}, timeout=timeout)
            if not r.ok:
                return _fail("gemini", f"HTTP {r.status_code}: {r.text[:200]}")
            models = r.json().get("models") or []
            names = [(m.get("name") or "").split("/")[-1] for m in models
                     if "generateContent" in (m.get("supportedGenerationMethods") or [])]
            want = (os.environ.get("GEMINI_MODEL") or "").split("/")[-1]
            model = want if want in names else next((n for n in names if "gemini" in n.lower()), None)
            if not model:
                return _fail("gemini", f"no gemini model with generateContent in the list: {names[:12]}")
            return LLMStatus(ok=True, provider="gemini", model=model, reason="")
        except Exception as e:  # noqa: BLE001
            return _fail("gemini", f"{type(e).__name__}: {e}")
    return _fail(None, "no API key set (XAI_API_KEY or GEMINI_API_KEY)")


def llm_available(timeout: float = 8.0) -> LLMStatus:
    """One cheap probe: is a configured key accepted, and which model will we use?
    {"ok", "provider": "xai"|"gemini"|None, "model", "reason"}. The chosen model is cached."""
    res = _probe(timeout)
    if res["ok"]:
        _LLM.update(provider=res["provider"], model=res["model"])
    else:
        _LLM.update(provider=None, model=None)
        print(f"[coach] LLM unavailable: {res['reason']}")
    return res


def _build_prompt(summary: dict) -> str:
    return (
        "Below is the analysis of one climb, as JSON. It has two lines of moves: \"your_climb\" (what the climber "
        "did on video) and \"suggested_line\" (the sequence we suggest for this climber's body).\n\n"
        "Write 5 to 8 short coaching insights in total. Each insight is one bullet tied to a move number. Cover:\n"
        "- centre of mass: where the hips should be before the move and why (use hips_vs_support_hand and feet);\n"
        "- hand choice: which hand leads and why. Warn about a barn door when the hand and the foot on the same "
        "side are the only contacts and the hips sit outside that line;\n"
        "- hips and feet: flag, drop knee, high foot, and when to bring the feet up;\n"
        "- pacing and rests: use duration_s and support_elbow_min_deg (a straight arm rests, a bent arm tires);\n"
        "- one bullet on what the suggested line changes and why it suits this body (arm span, height, board "
        "angle, injury flags when given).\n\n"
        "Rules:\n"
        "- Use only holds, moves and numbers from the data. Never invent holds. Name holds by their labels (H4).\n"
        "- Plain language. Short sentences. No costs, scores or model talk.\n"
        "- Where pose_missing is true, say \"we could not see\" instead of guessing.\n"
        "- Do not change the suggested sequence; explain it.\n"
        "- Output exactly two headed sections, in this order, with these exact headings: \"Your climb\" and "
        "\"Suggested line\". Under each heading write \"- \" bullets, and every bullet starts with \"Move N:\". "
        "No other text before, between or after the sections.\n\n"
        "DATA:\n" + json.dumps(summary, indent=1)
    )


def llm_insights(summary: dict, timeout: float = 45.0) -> Optional[str]:
    """Per-move coaching from Grok or Gemini. Returns the reply text, or None
    (with the reason printed to the terminal) when no model is usable."""
    if not _LLM.get("model"):
        if not llm_available():
            return None
    provider, model = _LLM["provider"], _LLM["model"]
    prompt = _build_prompt(summary)
    try:
        import requests
        if provider == "xai":
            r = requests.post(_XAI_CHAT, headers={"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"},
                              json={"model": model, "temperature": 0.3, "max_tokens": 1500,
                                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                 {"role": "user", "content": prompt}]},
                              timeout=timeout)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            txt = r.json()["choices"][0]["message"]["content"]
        elif provider == "gemini":
            r = requests.post(f"{_GEMINI_MODELS}/{model}:generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                              json={"system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                                    "contents": [{"parts": [{"text": prompt}]}],
                                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500}},
                              timeout=timeout)
            if not r.ok:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            txt = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            raise RuntimeError("no provider selected")
        txt = (txt or "").strip()
        if not txt:
            raise RuntimeError("empty reply")
        return txt
    except Exception as e:  # noqa: BLE001
        print(f"[coach] LLM call failed ({provider} {model}): {_scrub(f'{type(e).__name__}: {e}')[:300]}")
        return None


llm_rewrite = llm_insights   # older name


_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+(.*\S)\s*$")
_DECOR_RE = re.compile(r"[#*_:>`\-\s]+")


def _clean(text: str) -> str:
    return text.replace("**", "").replace("__", "").strip()


def parse_insights(text: Optional[str]) -> dict:
    """Split a reply into {"your_climb": [...], "suggested": [...]}; tolerant
    of markdown headers, bold and numbered bullets."""
    out = {"your_climb": [], "suggested": []}
    if not text:
        return out
    section = "your_climb"
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = _DECOR_RE.sub(" ", line).strip().lower()
        if head and len(head.split()) <= 4 and not head.startswith("move "):
            if head.startswith("your climb"):
                section = "your_climb"
                continue
            if head.startswith("suggested"):
                section = "suggested"
                continue
        m = _BULLET_RE.match(line)
        item = _clean(m.group(1) if m else line)
        if item:
            out[section].append(item)
    return out
