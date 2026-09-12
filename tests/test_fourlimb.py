"""Four-limb (hands + feet) optimizer tests and hands-only regression."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.holds import make_hold
from sendit.optimizer import (Climber, Weights, Feasibility, optimize_beta, score_sequence, estimate_states,
                              feet_ok, foot_in_window, hands_mid, _astar, _beam, _make_expand, _make_heuristic,
                              _FootCandidates, _is_goal, LEFT, RIGHT, LEFT_FOOT, RIGHT_FOOT, FEET, HANDS)
from test_optimizer import ladder, A

CLIMBER = Climber(arm_span_px=A, leg_len_px=450, torso_px=300)   # H = 750


def wall_with_feet(foot_on_route=True):
    """Tall ladder: start S near the ground, hand rungs every ~350 px, finish T
    five moves up; foot holds every ~300 px so feet can follow. Ground is
    assumed just below the lowest hold, so the upper moves are cut loose
    unless feet are placed."""
    holds = [
        make_hold(0, 500, 1000, role="start", label="S"),
        make_hold(1, 450, 650, label="M1"), make_hold(2, 560, 640, label="M2"),
        make_hold(3, 480, 300, label="M3"), make_hold(8, 540, 280, label="M4"),
        make_hold(9, 470, -60, label="M5"), make_hold(10, 550, -80, label="M6"),
        make_hold(11, 500, -420, role="finish", label="T"),
        make_hold(4, 420, 1500, role="foot", on_route=foot_on_route, label="F1"),
        make_hold(5, 600, 1450, role="foot", on_route=foot_on_route, label="F2"),
        make_hold(6, 480, 1150, role="foot", on_route=foot_on_route, label="F3"),
        make_hold(7, 560, 900, role="foot", on_route=foot_on_route, label="F4"),
        make_hold(12, 440, 600, role="foot", on_route=foot_on_route, label="F5"),
        make_hold(13, 580, 250, role="foot", on_route=foot_on_route, label="F6"),
        make_hold(14, 460, -100, role="foot", on_route=foot_on_route, label="F7"),
    ]
    return holds


def test_hands_only_regression_matches_previous_behaviour():
    holds = ladder(grips_left=(5, 5), grips_right=(1, 1))
    res = optimize_beta(holds, Climber(arm_span_px=A), Weights(), Feasibility(), (0, 0), {5})
    assert res["limbs"] == "hands" and len(res["states"][0]) == 2
    assert set(res["holds_used"]) & {3, 4} and not (set(res["holds_used"]) & {1, 2})
    assert res["search"]["method"] == "astar" and res["search"]["exact"]
    # A* with the admissible bound must equal plain Dijkstra (heuristic 0)
    hid = {h["id"]: h for h in holds}
    from sendit.optimizer import hand_holds, foot_support_costs
    route = hand_holds(holds)
    W, F = Weights(), Feasibility()
    expand = _make_expand(hid, route, Climber(arm_span_px=A), W, F, foot_support_costs(holds, Climber(arm_span_px=A), F), "hands", None)
    goal = lambda s: _is_goal(s, {5}, False)  # noqa: E731
    m0, c0, _ = _astar((0, 0), goal, expand, lambda s: 0.0)
    assert abs(c0 - res["total_cost"]) < 1e-9


def test_fourlimb_picks_feet_inside_window_and_prefers_support():
    holds = wall_with_feet(True)
    gy = 1612.0   # fix the assumed ground so both variants below share it
    res = optimize_beta(holds, CLIMBER, Weights(), Feasibility(), (0, 0), {11}, limbs="all", ground_y=gy)
    assert res is not None and res["limbs"] == "all"
    hid = {h["id"]: h for h in holds}
    for st in res["states"]:
        assert feet_ok(hid, tuple(st), CLIMBER, Feasibility())
    assert res["n_foot_moves"] >= 1, "feet should be placed at some point"
    assert all(f is None or hid[f].get("role") == "foot" or hid[f].get("on_route") for st in res["states"] for f in st[2:])
    # with the foot-only holds off-route, feet may still use hand holds (as in real climbing) but never
    # the foot-only ones, and the climb cannot get cheaper than with them available
    foot_only = {h["id"] for h in holds if h.get("role") == "foot"}
    res_off = optimize_beta(wall_with_feet(False), CLIMBER, Weights(), Feasibility(), (0, 0), {11}, limbs="all", ground_y=gy)
    assert res_off is not None
    assert not any(f in foot_only for st in res_off["states"] for f in st[2:])
    assert res["total_cost"] <= res_off["total_cost"] + 1e-9


def test_astar_equals_dijkstra_on_fourlimb_fixture():
    holds = wall_with_feet(True)
    hid = {h["id"]: h for h in holds}
    from sendit.optimizer import hand_holds
    W, F = Weights(), Feasibility()
    fc = _FootCandidates(holds, hid, CLIMBER, F)
    expand = _make_expand(hid, hand_holds(holds), CLIMBER, W, F, None, "all", fc)
    goal = lambda s: _is_goal(s, {11}, False)  # noqa: E731
    m_d, c_d, _ = _astar((0, 0, None, None), goal, expand, lambda s: 0.0)
    m_a, c_a, st = _astar((0, 0, None, None), goal, expand, _make_heuristic(hid, {11}, CLIMBER, W, F))
    assert abs(c_d - c_a) < 1e-9
    m_b, c_b, _ = _beam((0, 0, None, None), goal, expand, _make_heuristic(hid, {11}, CLIMBER, W, F), beam=50)
    assert m_b is not None and c_b >= c_d - 1e-9
    # beam path is valid: replay it with the same expand
    s = (0, 0, None, None)
    for mv in m_b:
        nxt = {ns: c for ns, c, _ in expand(s)}
        ns = list(s)
        from sendit.optimizer import LIMB_INDEX
        ns[LIMB_INDEX[mv["limb"]]] = mv["to"]
        assert tuple(ns) in nxt
        s = tuple(ns)


def test_budget_exhaustion_falls_back_to_beam():
    holds = wall_with_feet(True)
    res = optimize_beta(holds, CLIMBER, Weights(), Feasibility(), (0, 0), {11}, limbs="all",
                        exact_threshold=0, max_expansions=2, beam=30)
    assert res is not None and res["search"]["method"] == "beam" and res["search"]["exact"] is False
    exact = optimize_beta(holds, CLIMBER, Weights(), Feasibility(), (0, 0), {11}, limbs="all")
    assert res["relaxed_to"] is None and exact["relaxed_to"] is None, "beam failure must not silently relax the envelope"
    assert res["total_cost"] >= exact["total_cost"] - 1e-9


def test_score_sequence_fourlimb_and_infeasible_flag():
    holds = wall_with_feet(True)
    placements = [{"limb": LEFT_FOOT, "hold_id": 4}, {"limb": RIGHT_FOOT, "hold_id": 5},
                  {"hand": LEFT, "hold_id": 1}, {"hand": RIGHT, "hold_id": 2},
                  {"limb": LEFT_FOOT, "hold_id": 6}, {"hand": LEFT, "hold_id": 3}]
    res = score_sequence(holds, CLIMBER, Weights(), Feasibility(), (0, 0), placements)
    assert res["limbs"] == "all" and res["n_foot_moves"] == 3 and res["n_hand_moves"] == 3
    # a foot placed far outside the window is flagged, not dropped
    bad = score_sequence(holds, CLIMBER, Weights(), Feasibility(), (0, 0), [{"limb": LEFT_FOOT, "hold_id": 11}])
    assert bad["n_infeasible"] == 1


def test_estimate_states_grows_with_feet():
    holds = wall_with_feet(True)
    e_h = estimate_states(holds, CLIMBER, Feasibility(), "hands")
    e_a = estimate_states(holds, CLIMBER, Feasibility(), "all")
    assert e_a["joint_states"] > e_h["joint_states"] >= 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
