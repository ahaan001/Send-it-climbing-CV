"""Deterministic sanity tests: small graphs where the right answer is obvious.
Run: python3 -m pytest tests -q   (or python3 tests/test_optimizer.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.holds import make_hold
from sendit.optimizer import (Climber, Weights, Feasibility, optimize_beta, score_sequence,
                              compare, hold_graph_edges, move_cost, foot_support_costs)

A = 1000.0  # arm span in px


def ladder(grips_left=(3, 3), grips_right=(3, 3)):
    """Start hold S at the bottom, finish T at the top, two candidate
    intermediate lines: LEFT column (holds L1, L2) and RIGHT column (R1, R2).
    Both columns are the same geometry; only grip ratings differ."""
    holds = [
        make_hold(0, 500, 1000, role="start", label="S"),
        make_hold(1, 420, 700, grip=grips_left[0], label="L1"),
        make_hold(2, 420, 400, grip=grips_left[1], label="L2"),
        make_hold(3, 580, 700, grip=grips_right[0], label="R1"),
        make_hold(4, 580, 400, grip=grips_right[1], label="R2"),
        make_hold(5, 500, 100, role="finish", label="T"),
    ]
    return holds


def test_A_grip_changes_beta():
    """Path through terrible holds (LEFT column) vs good holds (RIGHT column).
    With grip weight on, optimizer must prefer the RIGHT column; with grip
    weight zero the two columns tie and either is fine."""
    holds = ladder(grips_left=(5, 5), grips_right=(1, 1))
    climber = Climber(arm_span_px=A)
    res = optimize_beta(holds, climber, Weights(w_grip=1.0), Feasibility(), (0, 0), {5})
    assert res is not None
    used = set(res["holds_used"])
    assert used & {3, 4}, f"expected the good RIGHT column, got {res['holds_used']}"
    assert not (used & {1, 2}), f"should avoid terrible LEFT column, got {res['holds_used']}"
    # flipping the ratings flips the answer
    holds2 = ladder(grips_left=(1, 1), grips_right=(5, 5))
    res2 = optimize_beta(holds2, climber, Weights(w_grip=1.0), Feasibility(), (0, 0), {5})
    assert set(res2["holds_used"]) & {1, 2} and not (set(res2["holds_used"]) & {3, 4})


def test_A2_short_reach_to_terrible_vs_longer_to_jug():
    """A very short reach to a terrible sloper vs a longer reach to a secure jug."""
    holds = [
        make_hold(0, 500, 1000, role="start", label="S"),
        make_hold(1, 500, 800, grip=5, label="sloper"),   # 0.20 span, terrible
        make_hold(2, 500, 600, grip=1, label="jug"),      # 0.40 span, excellent
        make_hold(3, 500, 300, role="finish", label="T"),
    ]
    climber = Climber(arm_span_px=A)
    res = optimize_beta(holds, climber, Weights(w_grip=1.0, w_move=0.6), Feasibility(), (0, 0), {3})
    assert 1 not in res["holds_used"], res["holds_used"]
    res0 = optimize_beta(holds, climber, Weights(w_grip=0.0, w_move=0.6), Feasibility(), (0, 0), {3})
    # with no grip penalty, cost is dominated by reach^2 + per-move; still fine either way, but must run
    assert res0 is not None


def test_B_morphology_changes_feasibility_and_path():
    """Direct move S -> T needs a span of 0.80 arm spans. Long-reach climber
    can do it; a climber at 85 % reach can't and must use the intermediate."""
    holds = [
        make_hold(0, 500, 1000, role="start", label="S"),
        make_hold(1, 500, 600, grip=4, label="mid"),      # 0.40 (poor hold, still needed by short climber)
        make_hold(2, 500, 200, role="finish", label="T"), # 0.80 direct from S
    ]
    tall = Climber(arm_span_px=A)
    short = tall.scaled(0.85)
    # per-move cost high enough that a climber who CAN skip the poor hold prefers to
    W = Weights(w_move=1.2, r_ref=0.5, w_grip=1.0)
    F = Feasibility(max_reach_frac=0.85)
    e_tall = {(i, j) for i, j, _ in hold_graph_edges(holds, tall, F)}
    e_short = {(i, j) for i, j, _ in hold_graph_edges(holds, short, F)}
    assert (0, 2) in e_tall and (0, 2) not in e_short
    r_tall = optimize_beta(holds, tall, W, F, (0, 0), {2})
    r_short = optimize_beta(holds, short, W, F, (0, 0), {2})
    assert 1 not in r_tall["holds_used"], r_tall["holds_used"]
    assert 1 in r_short["holds_used"], r_short["holds_used"]
    assert r_short["relaxed_to"] is None
    # same geometry costs more for the shorter climber
    m_tall = move_cost(holds[0], holds[1], holds[0], "LEFT", tall, W, F)[0]
    m_short = move_cost(holds[0], holds[1], holds[0], "LEFT", short, W, F)[0]
    assert m_short > m_tall


def test_C_observed_vs_optimized_improvement():
    """Observed sequence takes the terrible column; optimized takes the good
    one. Improvement must be positive and computed from the same objective."""
    holds = ladder(grips_left=(5, 5), grips_right=(1, 1))
    climber = Climber(arm_span_px=A)
    W, F = Weights(), Feasibility()
    observed_placements = [{"hand": "LEFT", "hold_id": 1}, {"hand": "RIGHT", "hold_id": 2},
                           {"hand": "LEFT", "hold_id": 5}]
    obs = score_sequence(holds, climber, W, F, (0, 0), observed_placements)
    opt = optimize_beta(holds, climber, W, F, (0, 0), {5})
    assert opt["total_cost"] <= obs["total_cost"]
    cmp_ = compare(obs, opt, {h["id"]: h for h in holds})
    expected = (obs["total_cost"] - opt["total_cost"]) / obs["total_cost"]
    assert abs(cmp_["improvement_frac"] - expected) < 1e-9
    assert cmp_["improvement_frac"] > 0
    assert cmp_["crux"]["move_index"] == obs["crux_index"]
    # scoring the optimizer's own moves reproduces its cost exactly
    re_scored = score_sequence(holds, climber, W, F, (0, 0),
                               [{"hand": m["hand"], "hold_id": m["to"]} for m in opt["moves"]])
    assert abs(re_scored["total_cost"] - opt["total_cost"]) < 1e-9


def test_D_disconnected_graph_relaxes_gracefully():
    holds = [make_hold(0, 500, 1000, role="start"), make_hold(1, 500, 0, role="finish")]  # 1.0 span apart
    res = optimize_beta(holds, Climber(arm_span_px=A), Weights(), Feasibility(max_reach_frac=0.85), (0, 0), {1})
    assert res is not None and res["relaxed_to"] is not None and res["relaxed_to"] >= 1.0
    res2 = optimize_beta(holds, Climber(arm_span_px=A), Weights(), Feasibility(max_reach_frac=0.85), (0, 0), {1}, relax=False)
    assert res2 is None


def test_E_off_route_holds_are_never_used():
    holds = ladder(grips_left=(1, 1), grips_right=(1, 1))
    for h in holds:
        if h["id"] in (1, 2):
            h["on_route"] = False
    res = optimize_beta(holds, Climber(arm_span_px=A), Weights(), Feasibility(), (0, 0), {5})
    assert not (set(res["holds_used"]) & {1, 2})


def test_F_foot_support():
    climber = Climber(arm_span_px=A, leg_len_px=450, torso_px=300)   # H = 750
    holds = [make_hold(0, 500, 1000), make_hold(1, 500, 500), make_hold(2, 500, 0)]
    fc = foot_support_costs(holds, climber)
    assert fc[1] == 0.0          # hold 0 is 500px (0.67 H) below hold 1 -> supported
    assert fc[2] == 0.0          # hold 1 is 0.67H below hold 2 -> supported
    assert fc[0] == 1.0          # nothing below the bottom hold


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
