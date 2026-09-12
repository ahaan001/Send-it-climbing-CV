"""Rule-based injury-risk flags for a hand sequence.

HEURISTIC, NOT MEDICAL ADVICE. Six simple rules, run on both the climber's
own line (the scored observed result) and the suggested line (the optimizer
result). Each rule looks at one hand move and says: this move pattern is
commonly associated with a strain, here is a plain-language alternative.

Rules (one function each, thresholds are the module constants below):
  overextension  reach after the move >= REACH_OVEREXTENSION of arm span
  impingement    hands crossed (> CROSS_AMOUNT) while reaching >= CROSS_REACH
  pulley         poor target hold (grip >= PULLEY_GRIP) at a long reach (>= PULLEY_REACH)
  drop           the moving hand goes DOWN by more than DROP_FRAC of arm span
  lockoff        support-arm elbow bent below LOCKOFF_ELBOW_DEG for >= LOCKOFF_LONG_S  (video only)
  dynamic        hips travel > DYNAMIC_HIP_FRAC of arm span in < DYNAMIC_MAX_S         (video only)

Board angle is deliberately NOT an input: severity depends only on the move
geometry and, for the two video rules, on what the body actually did.

Move-context alignment (observed line only). ``beta.measured_move_context``
returns ``len(placements) - 1`` entries; entry ``i`` describes the transition
``placements[i] -> placements[i+1]`` (support arm = the hand that is NOT
placed at ``placements[i+1]``). ``pipeline.run_optimization`` scores
``placements[consumed:]`` -- the first ``consumed`` placements form the start
pair -- so observed hand moves are a SUFFIX of the placement list and the last
move corresponds to the last context entry. ``align_move_context`` therefore
(1) matches on ``frame`` when the context entries carry one, else (2)
reconstructs placement frames from the cumulative ``duration_frames`` and
matches the moves' ``frame`` values (robust to a skipped placement anywhere),
else (3) tail-aligns by index.
"""
from __future__ import annotations

from typing import Optional

from .optimizer import HANDS, LEFT

# --------------------------------------------------------------------------- thresholds
REACH_OVEREXTENSION = 0.85   # reach after the move, fraction of arm span
CROSS_AMOUNT = 0.5           # crossed-hands amount from optimizer.move_cost (0..1)
CROSS_REACH = 0.6            # reach that makes a crossed position risky
PULLEY_GRIP = 4              # grip rating 4 = poor, 5 = terrible
PULLEY_REACH = 0.7
DROP_FRAC = 0.15             # downward hand travel, fraction of arm span
LOCKOFF_ELBOW_DEG = 60.0     # support-arm minimum elbow angle
LOCKOFF_LONG_S = 1.0
DYNAMIC_HIP_FRAC = 0.5       # hip travel, fraction of arm span
DYNAMIC_MAX_S = 0.5
DISCLAIMER = "Heuristic, not medical advice."

# --------------------------------------------------------------------------- severity table
OVEREXTENSION_HIGH_REACH = 0.95   # overextension escalates to high at this reach
PULLEY_HIGH_GRIP = 5              # pulley escalates to high on a terrible hold
DYNAMIC_MEDIUM_HIP_FRAC = 0.8     # dynamic escalates to medium at this hip travel

SEVERITY = {
    "overextension": "medium",   # high when reach_frac >= OVEREXTENSION_HIGH_REACH
    "impingement": "medium",
    "pulley": "medium",          # high when grip >= PULLEY_HIGH_GRIP
    "drop": "medium",
    "lockoff": "medium",
    "dynamic": "low",            # medium when hip travel > DYNAMIC_MEDIUM_HIP_FRAC of arm span
}
SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2}
RULES = ("overextension", "impingement", "pulley", "drop", "lockoff", "dynamic")

# Plain climber language. Placeholders: {n} move number, {hand}/{other} left|right, {frm}/{to} hold ids.
RULE_COPY = {
    "overextension": (
        "Move {n} is a near-full stretch for your {hand} hand, and a reach that long loads the shoulder.",
        "Bring a foot up or move your {other} hand closer first, so the reach to {to} is shorter."),
    "impingement": (
        "On move {n} your hands are crossed and you are still reaching far, which pinches the shoulder.",
        "Match hands or swap them before you reach for {to}, so your arms stay uncrossed."),
    "pulley": (
        "Move {n} is a long reach to {to}, a poor hold, so your fingers take the load at full stretch.",
        "Get your body closer to {to} before you grab it, and hold it open-handed rather than crimping hard."),
    "drop": (
        "On move {n} your {hand} hand drops a long way down to {to}, which can shock-load your {other} arm.",
        "Lower your body with a foot move first, then take {to} with a short, controlled reach."),
    "lockoff": (
        "Move {n} held a deep lock-off on your {other} arm for a long time, which strains the elbow.",
        "Set a higher foot or move through this reach faster, so the lock-off is short."),
    "dynamic": (
        "Move {n} was a fast, jumpy move to {to}, and a quick catch is hard on the fingers and shoulders.",
        "Try a slower, more controlled reach to {to}, or use a hold in between."),
}


# --------------------------------------------------------------------------- helpers
def _reach(move: dict) -> float:
    return float(move.get("reach_frac") or 0.0)


def _label(hid: dict, i) -> str:
    h = hid.get(i) if hid else None
    return h.get("label", f"H{i}") if h else f"H{i}"


def _target_grip(move: dict, hid: dict) -> int:
    h = hid.get(move.get("to")) if hid else None
    if h is not None and h.get("grip") is not None:
        return int(h["grip"])
    return int(move.get("grip") or 3)


def _seconds(ctx: dict, fps) -> Optional[float]:
    d = ctx.get("duration_frames")
    if d is None or not fps or fps <= 0:
        return None
    return float(d) / float(fps)


def hand_moves(result: Optional[dict]) -> list:
    """Hand moves of an optimizer result, in order (foot moves of a full-body plan are skipped)."""
    if not result:
        return []
    return [m for m in result.get("moves") or [] if (m.get("limb") or m.get("hand")) in HANDS]


# --------------------------------------------------------------------------- rules (severity or None)
def check_overextension(move: dict) -> Optional[str]:
    """(a) Shoulder over-extension: hand-to-hand span after the move >= REACH_OVEREXTENSION."""
    r = _reach(move)
    if r < REACH_OVEREXTENSION:
        return None
    return "high" if r >= OVEREXTENSION_HIGH_REACH else SEVERITY["overextension"]


def check_impingement(move: dict) -> Optional[str]:
    """(b) Shoulder impingement: crossed hands (> CROSS_AMOUNT) while reaching >= CROSS_REACH."""
    cross = float(move.get("cross") or 0.0)
    if cross > CROSS_AMOUNT and _reach(move) >= CROSS_REACH:
        return SEVERITY["impingement"]
    return None


def check_pulley(move: dict, hid: dict) -> Optional[str]:
    """(c) Finger pulley load: poor target hold (grip >= PULLEY_GRIP) at a reach >= PULLEY_REACH."""
    grip = _target_grip(move, hid)
    if grip >= PULLEY_GRIP and _reach(move) >= PULLEY_REACH:
        return "high" if grip >= PULLEY_HIGH_GRIP else SEVERITY["pulley"]
    return None


def check_drop(move: dict, hid: dict, arm_span_px: float) -> Optional[str]:
    """(d) Downward move: target hold lower than the origin by more than DROP_FRAC of arm span
    (image y grows downward, so dy = to.y - from.y > 0 is a drop)."""
    if not hid or not arm_span_px:
        return None
    a, b = hid.get(move.get("from")), hid.get(move.get("to"))
    if a is None or b is None:
        return None
    dy = float(b["y"]) - float(a["y"])
    if dy > DROP_FRAC * float(arm_span_px):
        return SEVERITY["drop"]
    return None


def check_lockoff(ctx: Optional[dict], fps) -> Optional[str]:
    """(e) Long deep lock-off: support-arm minimum elbow angle < LOCKOFF_ELBOW_DEG during a move
    that lasted >= LOCKOFF_LONG_S. Needs a measured move context and the video fps."""
    if not ctx:
        return None
    elbow, secs = ctx.get("support_elbow_min_deg"), _seconds(ctx, fps)
    if elbow is None or secs is None:
        return None
    if float(elbow) < LOCKOFF_ELBOW_DEG and secs >= LOCKOFF_LONG_S:
        return SEVERITY["lockoff"]
    return None


def check_dynamic(ctx: Optional[dict], arm_span_px: float, fps) -> Optional[str]:
    """(f) Dynamic move: hips travel > DYNAMIC_HIP_FRAC of arm span in < DYNAMIC_MAX_S.
    Needs a measured move context and the video fps."""
    if not ctx or not arm_span_px:
        return None
    hip, secs = ctx.get("hip_travel_px"), _seconds(ctx, fps)
    if hip is None or secs is None:
        return None
    hip = float(hip)
    if hip > DYNAMIC_HIP_FRAC * float(arm_span_px) and secs < DYNAMIC_MAX_S:
        return "medium" if hip > DYNAMIC_MEDIUM_HIP_FRAC * float(arm_span_px) else SEVERITY["dynamic"]
    return None


# --------------------------------------------------------------------------- context alignment
def _align_by_reconstructed_frames(moves: list, move_context: list) -> Optional[list]:
    """Placement frames relative to placements[0] follow from the cumulative durations
    (rel[i] = frame of placements[i] - frame of placements[0]). A move ending at
    placements[i] has frame == rel[i] + shift for one unknown shift; try each anchor
    and accept the first that explains EVERY move. Returns None if none does."""
    frames = [m.get("frame") for m in moves]
    durs = [c.get("duration_frames") if isinstance(c, dict) else None for c in move_context]
    if not frames or any(f is None for f in frames) or any(d is None for d in durs):
        return None
    frames = [int(f) for f in frames]
    rel = [0]
    for d in durs:
        rel.append(rel[-1] + int(d))
    for anchor in range(len(rel) - 1, 0, -1):          # try "last move ends at placements[anchor]"
        shift = frames[-1] - rel[anchor]
        mapping, i = [], 1
        for f in frames:                                # monotone greedy match, handles equal frames
            while i < len(rel) and rel[i] + shift < f:
                i += 1
            if i < len(rel) and rel[i] + shift == f:
                mapping.append(i - 1)
                i += 1
            else:
                mapping.append(None)
        if all(k is not None for k in mapping):
            return [move_context[k] for k in mapping]
    return None


def align_move_context(moves: list, move_context: Optional[list]) -> list:
    """One context entry (or None) per hand move. See the module docstring for why the
    observed moves are the tail of the context list; frame matching is preferred when
    possible so a skipped placement in the middle does not shift every later entry."""
    n = len(moves)
    if not move_context or n == 0:
        return [None] * n
    ctx = list(move_context)
    if all(isinstance(c, dict) and c.get("frame") is not None for c in ctx) \
            and all(m.get("frame") is not None for m in moves):
        by_frame = {int(c["frame"]): c for c in ctx}
        return [by_frame.get(int(m["frame"])) for m in moves]
    by_frames = _align_by_reconstructed_frames(moves, ctx)
    if by_frames is not None:
        return by_frames
    offset = len(ctx) - n
    return [ctx[offset + j] if 0 <= offset + j < len(ctx) else None for j in range(n)]


# --------------------------------------------------------------------------- public API
def _row(rule: str, severity: str, n: int, move: dict, hid: dict) -> dict:
    hand = "left" if move.get("hand") == LEFT else "right"
    other = "right" if hand == "left" else "left"
    frm, to = _label(hid, move.get("from")), _label(hid, move.get("to"))
    why, instead = RULE_COPY[rule]
    fmt = dict(n=n, hand=hand, other=other, frm=frm, to=to)
    return {"severity": severity, "move": n, "hand": hand, "from": frm, "to": to, "rule": rule,
            "why": why.format(**fmt), "instead": instead.format(**fmt)}


def flag_line(result: Optional[dict], hid: dict, arm_span_px: float,
              move_context: Optional[list] = None, fps: Optional[float] = None) -> list:
    """Injury-risk rows for one scored line (an optimizer result dict, or None).
    Rows: {severity, move (1-based hand-move number), hand, from, to, rule, why, instead},
    in move order, then rule order. The two video rules (lockoff, dynamic) only run when
    move_context AND fps are given, i.e. on the climber's own line."""
    moves = hand_moves(result)
    if not moves:
        return []
    ctxs = align_move_context(moves, move_context)
    rows = []
    for j, (m, ctx) in enumerate(zip(moves, ctxs)):
        n = j + 1
        checks = (
            ("overextension", check_overextension(m)),
            ("impingement", check_impingement(m)),
            ("pulley", check_pulley(m, hid)),
            ("drop", check_drop(m, hid, arm_span_px)),
            ("lockoff", check_lockoff(ctx, fps)),
            ("dynamic", check_dynamic(ctx, arm_span_px, fps)),
        )
        for rule, sev in checks:
            if sev:
                rows.append(_row(rule, sev, n, m, hid))
    return rows


def _arm_span_px(R: dict) -> float:
    """Measured climber's arm span. R['measured'] is an optimizer.Climber (or a dict once cached)."""
    for key in ("measured", "climber"):
        c = R.get(key)
        if c is None:
            continue
        v = getattr(c, "arm_span_px", None)
        if v is None and isinstance(c, dict):
            v = c.get("arm_span_px")
        if v:
            return float(v)
    for key in ("observed", "optimized"):
        res = R.get(key)
        if res and isinstance(res.get("climber"), dict) and res["climber"].get("arm_span_px"):
            return float(res["climber"]["arm_span_px"])
    return 0.0


def injury_report(R: dict, hid: dict, move_context: Optional[list] = None, fps: Optional[float] = None) -> dict:
    """{"observed": rows for the climber's own line, "suggested": rows for the suggested line,
    "disclaimer": DISCLAIMER}. R is pipeline.run_optimization's result; move_context is
    analysis['move_context'] and fps analysis['fps'] (both optional, observed line only).
    The measured arm span is used for both lines so a drop is judged against the real body."""
    out = {"observed": [], "suggested": [], "disclaimer": DISCLAIMER}
    if not R or R.get("error"):
        return out
    span = _arm_span_px(R)
    out["observed"] = flag_line(R.get("observed"), hid, span, move_context, fps)
    out["suggested"] = flag_line(R.get("optimized"), hid, span)
    return out


def worst_severity(rows: list) -> Optional[str]:
    """Highest severity in a list of rows, or None when there is nothing to flag."""
    if not rows:
        return None
    return max((r["severity"] for r in rows), key=lambda s: SEVERITY_ORDER.get(s, -1))
