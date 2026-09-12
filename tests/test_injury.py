"""Injury-risk rules: each rule fires on exactly one hand-written move and stays quiet on a benign one.
Run: venv/bin/python -m pytest tests -q
"""
import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.holds import make_hold
from sendit.optimizer import Climber, Weights, Feasibility, score_sequence, optimize_beta, LEFT, RIGHT
from sendit import injury
from sendit.injury import (flag_line, injury_report, align_move_context, hand_moves, worst_severity,
                           check_overextension, check_impingement, check_pulley, check_drop,
                           check_lockoff, check_dynamic,
                           REACH_OVEREXTENSION, CROSS_AMOUNT, CROSS_REACH, PULLEY_GRIP, PULLEY_REACH,
                           DROP_FRAC, LOCKOFF_ELBOW_DEG, LOCKOFF_LONG_S, DYNAMIC_HIP_FRAC, DYNAMIC_MAX_S,
                           DISCLAIMER, RULES, RULE_COPY)

A = 1000.0   # arm span in px


def wall():
    holds = [
        make_hold(0, 500, 1000, role="start", label="S"),
        make_hold(1, 500, 600, label="H1"),            # 0.40 above S: benign
        make_hold(2, 500, 100, label="H2"),            # 0.90 above S
        make_hold(3, 500, 40, label="H3"),             # 0.96 above S
        make_hold(4, 500, 400, grip=4, label="H4"),    # poor hold
        make_hold(5, 500, 300, grip=5, label="H5"),    # terrible hold
        make_hold(6, 500, 800, label="H6"),            # 0.20 BELOW H1 (drop)
        make_hold(7, 500, 700, label="H7"),            # 0.10 below H1 (not a drop)
    ]
    return holds, {h["id"]: h for h in holds}


def mv(hand, frm, to, reach, cross=0.0, other=0, frame=None, grip=3):
    """A move dict shaped like optimizer.score_sequence / optimize_beta output."""
    return {"limb": hand, "hand": hand, "from": frm, "to": to, "other": other, "cost": 1.0,
            "terms": {}, "feasible": True, "frame": frame, "reach_frac": reach, "travel_frac": 0.3,
            "up_frac": 0.3, "grip": grip, "cross": cross, "foot": 0.0, "reasons": []}


def res(moves):
    return {"label": "test", "moves": moves, "n_moves": len(moves), "states": [], "holds_used": []}


def rules_by_move(rows):
    out = {}
    for r in rows:
        out.setdefault(r["move"], []).append(r["rule"])
    return out


def ctx(elbow=None, hip=0.0, dur=10):
    return {"support_elbow_min_deg": elbow, "support_elbow_mean_deg": elbow, "hip_travel_px": hip,
            "duration_frames": dur}


# --------------------------------------------------------------------------- constants
def test_threshold_constants_and_disclaimer():
    assert (REACH_OVEREXTENSION, CROSS_AMOUNT, CROSS_REACH, PULLEY_GRIP, PULLEY_REACH, DROP_FRAC) == \
        (0.85, 0.5, 0.6, 4, 0.7, 0.15)
    assert (LOCKOFF_ELBOW_DEG, LOCKOFF_LONG_S, DYNAMIC_HIP_FRAC, DYNAMIC_MAX_S) == (60.0, 1.0, 0.5, 0.5)
    assert DISCLAIMER == "Heuristic, not medical advice."


# --------------------------------------------------------------------------- (a) overextension
def test_a_overextension():
    holds, hid = wall()
    moves = [mv(LEFT, 0, 1, 0.40),          # benign
             mv(RIGHT, 0, 2, 0.90),         # medium
             mv(LEFT, 1, 3, 0.96, other=2)]  # high
    assert check_overextension(moves[0]) is None
    assert check_overextension(moves[1]) == "medium"
    assert check_overextension(moves[2]) == "high"
    rows = flag_line(res(moves), hid, A)
    assert rules_by_move(rows) == {2: ["overextension"], 3: ["overextension"]}
    assert [r["severity"] for r in rows] == ["medium", "high"]
    assert rows[0]["hand"] == "right" and rows[0]["from"] == "S" and rows[0]["to"] == "H2"
    assert rows[1]["hand"] == "left" and rows[1]["from"] == "H1" and rows[1]["to"] == "H3"


# --------------------------------------------------------------------------- (b) impingement
def test_b_impingement():
    holds, hid = wall()
    moves = [mv(LEFT, 0, 1, 0.40),                  # benign
             mv(RIGHT, 0, 1, 0.65, cross=0.8),      # crossed and reaching: medium
             mv(LEFT, 1, 7, 0.40, cross=0.8),       # crossed but short: nothing
             mv(RIGHT, 1, 4, 0.65, cross=0.3)]      # reaching but barely crossed: nothing
    assert check_impingement(moves[0]) is None
    assert check_impingement(moves[1]) == "medium"
    assert check_impingement(moves[2]) is None and check_impingement(moves[3]) is None
    rows = flag_line(res(moves), hid, A)
    assert rules_by_move(rows) == {2: ["impingement"]}
    assert rows[0]["severity"] == "medium"


# --------------------------------------------------------------------------- (c) pulley
def test_c_pulley():
    holds, hid = wall()
    moves = [mv(LEFT, 0, 1, 0.40),          # benign, average hold
             mv(RIGHT, 0, 4, 0.75),         # poor hold, long reach: medium
             mv(LEFT, 1, 5, 0.75, other=4),  # terrible hold, long reach: high
             mv(RIGHT, 4, 5, 0.30, other=5),  # terrible hold but close: nothing
             mv(LEFT, 5, 1, 0.75, other=5)]  # long reach to an average hold: nothing
    assert check_pulley(moves[0], hid) is None
    assert check_pulley(moves[1], hid) == "medium"
    assert check_pulley(moves[2], hid) == "high"
    assert check_pulley(moves[3], hid) is None and check_pulley(moves[4], hid) is None
    rows = flag_line(res(moves), hid, A)
    # move 5 is a 0.75 reach downwards 300 px -> the drop rule fires there, not pulley
    assert rules_by_move(rows) == {2: ["pulley"], 3: ["pulley"], 5: ["drop"]}
    assert [r["severity"] for r in rows if r["rule"] == "pulley"] == ["medium", "high"]
    # grip is read from the hold set, not the move dict
    assert check_pulley(mv(LEFT, 0, 4, 0.75, grip=1), hid) == "medium"


# --------------------------------------------------------------------------- (d) drop
def test_d_drop():
    holds, hid = wall()
    moves = [mv(LEFT, 0, 1, 0.40),                  # up: benign
             mv(RIGHT, 1, 6, 0.20, other=1),        # down 200 px = 0.20 span: medium
             mv(LEFT, 1, 7, 0.10, other=6)]         # down 100 px = 0.10 span: nothing
    assert check_drop(moves[0], hid, A) is None
    assert check_drop(moves[1], hid, A) == "medium"
    assert check_drop(moves[2], hid, A) is None
    rows = flag_line(res(moves), hid, A)
    assert rules_by_move(rows) == {2: ["drop"]}
    # threshold scales with the climber: for a huge arm span the same drop is nothing
    assert check_drop(moves[1], hid, 5000.0) is None
    assert check_drop(moves[1], hid, 0.0) is None   # unknown span: rule skipped, never crashes


# --------------------------------------------------------------------------- (e) lock-off
def test_e_lockoff_needs_context_and_fps():
    holds, hid = wall()
    fps = 30.0
    moves = [mv(LEFT, 0, 1, 0.40, frame=10), mv(RIGHT, 0, 7, 0.40, frame=50, other=1),
             mv(LEFT, 1, 4, 0.40, frame=90, other=7)]
    context = [ctx(elbow=90.0, dur=10),            # open elbow, short: nothing
               ctx(elbow=45.0, dur=40),            # deep lock-off for 1.33 s: medium
               ctx(elbow=45.0, dur=12)]            # deep but only 0.4 s: nothing
    assert check_lockoff(context[0], fps) is None
    assert check_lockoff(context[1], fps) == "medium"
    assert check_lockoff(context[2], fps) is None
    assert check_lockoff(context[1], None) is None and check_lockoff(None, fps) is None
    rows = flag_line(res(moves), hid, A, move_context=context, fps=fps)
    assert rules_by_move(rows) == {2: ["lockoff"]}
    assert rows[0]["severity"] == "medium" and rows[0]["hand"] == "right" and "left arm" in rows[0]["why"]
    # the video rules never run without a context or without fps
    assert flag_line(res(moves), hid, A) == []
    assert flag_line(res(moves), hid, A, move_context=context) == []


# --------------------------------------------------------------------------- (f) dynamic
def test_f_dynamic_hip_travel():
    holds, hid = wall()
    fps = 30.0
    moves = [mv(LEFT, 0, 1, 0.40, frame=10), mv(RIGHT, 0, 7, 0.40, frame=20, other=1),
             mv(LEFT, 1, 4, 0.40, frame=30, other=7), mv(RIGHT, 7, 1, 0.40, frame=60, other=4)]
    context = [ctx(elbow=120.0, hip=100.0, dur=10),    # little hip travel: nothing
               ctx(elbow=120.0, hip=600.0, dur=10),    # 0.6 span in 0.33 s: low
               ctx(elbow=120.0, hip=850.0, dur=10),    # 0.85 span in 0.33 s: medium
               ctx(elbow=120.0, hip=600.0, dur=30)]    # 0.6 span but a slow 1.0 s: nothing
    assert check_dynamic(context[0], A, fps) is None
    assert check_dynamic(context[1], A, fps) == "low"
    assert check_dynamic(context[2], A, fps) == "medium"
    assert check_dynamic(context[3], A, fps) is None
    assert check_dynamic(context[1], A, None) is None and check_dynamic(context[1], 0.0, fps) is None
    rows = flag_line(res(moves), hid, A, move_context=context, fps=fps)
    assert rules_by_move(rows) == {2: ["dynamic"], 3: ["dynamic"]}
    assert [r["severity"] for r in rows] == ["low", "medium"]


# --------------------------------------------------------------------------- context alignment
def test_context_alignment_tail_and_frames():
    # placements at frames [0, 10, 40, 50]; the first two form the start pair, so the
    # observed moves are the placements at 40 and 50 -> context entries 1 and 2 of 3.
    context = [ctx(elbow=100.0, dur=10), ctx(elbow=40.0, dur=30), ctx(elbow=100.0, dur=10)]
    moves_with_frames = [mv(LEFT, 0, 1, 0.4, frame=40), mv(RIGHT, 0, 7, 0.4, frame=50)]
    moves_no_frames = [mv(LEFT, 0, 1, 0.4), mv(RIGHT, 0, 7, 0.4)]
    assert align_move_context(moves_with_frames, context) == [context[1], context[2]]
    assert align_move_context(moves_no_frames, context) == [context[1], context[2]]
    # a placement skipped in the middle (frame 50) is handled by frame matching, where
    # index alignment alone would shift the entries
    context5 = context + [ctx(elbow=100.0, dur=20)]     # placements [0, 10, 40, 50, 70]
    skipped = [mv(LEFT, 0, 1, 0.4, frame=40), mv(RIGHT, 0, 7, 0.4, frame=70)]
    assert align_move_context(skipped, context5) == [context5[1], context5[3]]
    # explicit frames on the entries win outright
    tagged = [dict(c, frame=f) for c, f in zip(context, (10, 40, 50))]
    assert align_move_context(moves_with_frames, tagged) == [tagged[1], tagged[2]]
    # degenerate inputs
    assert align_move_context(moves_no_frames, None) == [None, None]
    assert align_move_context([], context) == []
    assert align_move_context(moves_no_frames, [context[0]]) == [None, context[0]]
    # and the lock-off lands on hand move 1 (frame 40), not hand move 2
    holds, hid = wall()
    rows = flag_line(res(moves_with_frames), hid, A, move_context=context, fps=30.0)
    assert rules_by_move(rows) == {1: ["lockoff"]}


# --------------------------------------------------------------------------- real optimizer moves
def test_real_score_sequence_fields_flow_through():
    """Field names come from optimizer.move_cost: reach_frac, cross, frame. A left hand
    crossing far right of the right hand while reaching 0.65 spans fires impingement only."""
    holds = [make_hold(0, 300, 1000, role="start", label="A"),
             make_hold(1, 650, 450, label="B"),
             make_hold(2, 700, 300, role="finish", label="T")]
    hid = {h["id"]: h for h in holds}
    obs = score_sequence(holds, Climber(arm_span_px=A), Weights(), Feasibility(), (0, 0),
                         [{"hand": LEFT, "hold_id": 1, "frame": 12}, {"hand": RIGHT, "hold_id": 2, "frame": 40}])
    assert [m["frame"] for m in obs["moves"]] == [12, 40]
    assert obs["moves"][0]["cross"] > CROSS_AMOUNT and obs["moves"][0]["reach_frac"] >= CROSS_REACH
    rows = flag_line(obs, hid, A)
    assert rules_by_move(rows) == {1: ["impingement"]}
    assert rows[0]["hand"] == "left" and rows[0]["from"] == "A" and rows[0]["to"] == "B"
    assert hand_moves(None) == [] and hand_moves({}) == []


# --------------------------------------------------------------------------- report
def test_injury_report_both_lines_and_disclaimer():
    holds = [make_hold(0, 500, 1000, role="start", label="S"),
             make_hold(1, 500, 550, label="M"),
             make_hold(2, 500, 100, role="finish", label="T")]
    hid = {h["id"]: h for h in holds}
    measured = Climber(arm_span_px=A)
    W, F = Weights(), Feasibility()
    # the climber skipped M and went straight to T: a 0.90 reach
    placements = [{"hand": LEFT, "hold_id": 2, "frame": 30}]
    observed = score_sequence(holds, measured, W, F, (0, 0), placements)
    optimized = optimize_beta(holds, measured, W, F, (0, 0), {2})
    assert optimized is not None and 1 in optimized["holds_used"]
    R = {"measured": measured, "climber": measured, "observed": observed, "optimized": optimized}
    rep = injury_report(R, hid)
    assert set(rep) == {"observed", "suggested", "disclaimer"}
    assert rep["disclaimer"] == DISCLAIMER == "Heuristic, not medical advice."
    assert [r["rule"] for r in rep["observed"]] == ["overextension"]
    assert rep["observed"][0]["move"] == 1 and rep["observed"][0]["to"] == "T"
    assert rep["suggested"] == []
    # video context reaches the observed line only: placements were [L S f0, R S f2, L T f30]
    context = [ctx(elbow=100.0, dur=2), ctx(elbow=40.0, dur=28)]
    rep2 = injury_report(R, hid, move_context=context, fps=20.0)
    assert sorted(r["rule"] for r in rep2["observed"]) == ["lockoff", "overextension"]
    assert rep2["suggested"] == []
    # cached results carry the climber as a dict; missing pieces never crash
    R_dict = dict(R, measured={"arm_span_px": A})
    assert injury_report(R_dict, hid)["observed"] == rep["observed"]
    empty = injury_report({"error": "Need a start state and a finish hold."}, hid)
    assert empty == {"observed": [], "suggested": [], "disclaimer": DISCLAIMER}
    assert injury_report({"measured": measured, "observed": None, "optimized": None}, hid)["observed"] == []
    assert worst_severity(rep["observed"]) == "medium" and worst_severity([]) is None


# --------------------------------------------------------------------------- copy + invariants
BANNED = ["state", "a*", "dijkstra", "beam", "envelope", "objective", "px", "symmetric difference",
          "normalised", "proxy", "feasible edge", "joint states", "homography", "observed", "optimized",
          "beta", "cost"]


def test_copy_is_plain_climber_language():
    holds, hid = wall()
    move = mv(LEFT, 4, 5, 0.9, other=4)
    assert set(RULE_COPY) == set(RULES)
    for rule in RULES:
        row = injury._row(rule, "medium", 3, move, hid)
        for key in ("why", "instead"):
            text = row[key]
            assert text.endswith("."), text
            assert text.count(".") == 1, f"{rule} {key}: one short sentence, got {text!r}"
            low = text.lower()
            for w in BANNED:
                assert w not in low, f"{rule} {key} uses banned word {w!r}: {text}"
            # no numbers except the move number and hold ids
            stripped = re.sub(r"\b[Mm]ove 3\b", "", text).replace(row["from"], "").replace(row["to"], "")
            assert not re.search(r"\d", stripped), f"{rule} {key} leaks a number: {text}"
        assert set(row) == {"severity", "move", "hand", "from", "to", "rule", "why", "instead"}


def test_board_angle_cannot_change_severity():
    """Severity is a function of the move and the body only: there is no angle input anywhere."""
    for fn in (flag_line, injury_report, check_overextension, check_impingement, check_pulley,
               check_drop, check_lockoff, check_dynamic):
        assert not any("angle" in p for p in inspect.signature(fn).parameters), fn.__name__
    holds, hid = wall()
    moves = [mv(RIGHT, 0, 2, 0.90), mv(LEFT, 0, 4, 0.75, other=2)]
    base = flag_line(res(moves), hid, A)
    tilted = flag_line(dict(res(moves), board_angle_deg=45.0), hid, A)
    assert base == tilted and [r["rule"] for r in base] == ["overextension", "pulley"]
