"""Coaching text, LLM summary and plumbing tests. No network is used."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit import coach
from sendit.optimizer import Climber, Weights, Feasibility, optimize_beta, score_sequence, compare
from test_optimizer import ladder, A

BANNED = ["state", "A*", "Dijkstra", "beam", "envelope", "objective", "px", "symmetric difference",
          "normalised", "proxy", "feasible edge", "joint states", "homography", "observed", "optimized",
          "beta", "cost"]


def fixture(placements=None):
    """Ladder from test_optimizer with H<id> labels: the climber takes the terrible
    LEFT column (H1, H2); the suggested line takes the good RIGHT column (H3, H4)."""
    holds = ladder(grips_left=(5, 5), grips_right=(1, 1))
    for h in holds:
        h["label"] = f"H{h['id']}"
    climber = Climber(arm_span_px=A)
    W, F = Weights(), Feasibility()
    if placements is None:
        placements = [{"hand": "LEFT", "hold_id": 1, "frame": 10}, {"hand": "RIGHT", "hold_id": 2, "frame": 20},
                      {"hand": "LEFT", "hold_id": 5, "frame": 30}]
    obs = score_sequence(holds, climber, W, F, (0, 0), placements)
    opt = optimize_beta(holds, climber, W, F, (0, 0), {5})
    hid = {h["id"]: h for h in holds}
    R = {"observed": obs, "optimized": opt, "comparison": compare(obs, opt, hid),
         "climber": climber, "measured": climber}
    return R, hid


def strip_allowed(line: str) -> str:
    return re.sub(r"\bH\d+\b|\b[Mm]ove \d+\b", "", line)


def assert_no_banned(text: str):
    low = text.lower()
    for w in BANNED:
        if w.isalnum() or " " in w:
            assert not re.search(rf"\b{re.escape(w.lower())}\b", low), f"banned word {w!r} in: {text}"
        else:
            assert w.lower() not in low, f"banned word {w!r} in: {text}"


def test_rule_based_lines_are_cues_without_stray_digits():
    R, hid = fixture()
    for partial in (False, True):
        lines = coach.rule_based(R, hid, partial=partial)
        assert lines and all(isinstance(s, str) and s.strip() for s in lines)
        for line in lines:
            assert "\n" not in line
            assert not re.search(r"\d", strip_allowed(line)), f"stray number in tip: {line}"
            assert_no_banned(line)
    lines = coach.rule_based(R, hid)
    text = " ".join(lines)
    assert "hardest" in text
    assert "Move 2" in text or "Move 1" in text or "Move 3" in text
    assert "Same number of moves" in text and "H1" in text and "H2" in text   # 3 moves each; holds by label
    assert "Set your feet before move" in text
    partial_text = " ".join(coach.rule_based(R, hid, partial=True))
    assert "clip ends" in partial_text and "You used" not in partial_text
    # a longer climb: counts are spelled out, never digits
    R4, hid4 = fixture(placements=[{"hand": "LEFT", "hold_id": 1}, {"hand": "RIGHT", "hold_id": 2},
                                   {"hand": "LEFT", "hold_id": 4}, {"hand": "RIGHT", "hold_id": 5}])
    lines4 = coach.rule_based(R4, hid4)
    assert any("You used four moves. We suggest three." in s for s in lines4), lines4
    for line in lines4:
        assert not re.search(r"\d", strip_allowed(line)), f"stray number in tip: {line}"


def test_rule_based_without_a_suggested_line():
    R, hid = fixture()
    R2 = dict(R, optimized=None, comparison={})
    lines = coach.rule_based(R2, hid)
    assert len(lines) == 1 and "top" in lines[0]
    assert_no_banned(lines[0])
    # no moves seen on video: still describes the suggested line
    R3 = dict(R, observed=None, comparison={})
    lines = coach.rule_based(R3, hid)
    assert any("Suggested line" in s for s in lines)
    for line in lines:
        assert not re.search(r"\d", strip_allowed(line)), line


def test_why_block_is_three_short_sentences():
    R, hid = fixture()
    for partial in (False, True):
        block = coach.why_block(R, hid, partial=partial)
        assert 1 <= len(block) <= 3
        for s in block:
            assert len(s.split()) <= 15, s
            assert len(re.findall(r"[.!?](?:\s|$)", s)) <= 1, s
            assert_no_banned(s)
    block = coach.why_block(R, hid)
    assert "skips" in block[0] and "H1" in block[0] and "H2" in block[0]
    assert "instead" in block[1] and "H3" in block[1] and "H4" in block[1]
    assert "hardest because of" in block[2]
    assert "plan" in coach.why_block(R, hid, partial=True)[-1]
    # a long hold list is shortened instead of overflowing
    R2 = dict(R, comparison=dict(R["comparison"], only_observed=[0, 1, 2, 3, 4, 5] * 3))
    for s in coach.why_block(R2, hid):
        assert len(s.split()) <= 15, s


def lm(x, y, v=0.9):
    return [float(x), float(y), v]


def synthetic_pose():
    # frame 10: left hand just landed on H1, right hand still on H0, hips 200 px right of H0
    f10 = {"LEFT_HIP": lm(690, 1200), "RIGHT_HIP": lm(710, 1200),
           "LEFT_WRIST": lm(420, 700), "LEFT_INDEX": lm(420, 700),
           "RIGHT_WRIST": lm(500, 1000), "RIGHT_INDEX": lm(500, 1000)}
    # frame 20: hips not visible, wrists visible (right hand on H2, left on H1)
    f20 = {"LEFT_HIP": lm(410, 900, 0.2), "RIGHT_HIP": lm(430, 900, 0.2),
           "LEFT_WRIST": lm(420, 700), "RIGHT_WRIST": lm(420, 400)}
    return {"w": 1000, "h": 1500, "fps": 10.0, "n_frames": 40, "frames": {10: f10, 20: f20}}


def test_summary_for_llm_has_the_facts_a_coach_needs():
    R, hid = fixture()
    pose = synthetic_pose()
    foot_events = [{"hand": "LEFT_FOOT", "limb": "LEFT_FOOT", "hold_id": 0, "start": 0, "end": 12, "frames": 13}]
    # one extra leading context entry: contexts line up with the moves from the tail
    move_context = [{"support_elbow_min_deg": 999.0, "duration_frames": 99},
                    {"support_elbow_min_deg": 150.4, "duration_frames": 10, "hip_travel_px": 0.0},
                    {"support_elbow_min_deg": 80.2, "duration_frames": 10, "hip_travel_px": 50.0},
                    {"support_elbow_min_deg": None, "duration_frames": 10, "hip_travel_px": 50.0}]
    S = coach.summary_for_llm(R, hid, True, move_context=move_context, foot_events=foot_events, pose=pose,
                              fps=10.0, board_angle_deg=40, arm_span_m=1.8, height_m=1.75,
                              injury={"left_shoulder": True})
    for key in ("how_to_read", "climber", "injury", "partial_clip", "tracking_gaps", "your_climb",
                "suggested_line", "comparison"):
        assert key in S, key
    assert "objective" not in S and "objective" not in S["how_to_read"].lower()
    assert S["partial_clip"] is True and S["injury"] == {"left_shoulder": True}
    assert S["climber"] == {"board_angle_deg": 40.0, "arm_span_m": 1.8, "height_m": 1.75}

    moves = S["your_climb"]["moves"]
    assert [m["move"] for m in moves] == [1, 2, 3]
    m1 = moves[0]
    for key in ("hand", "support_hand", "from", "to", "from_role", "to_role", "direction", "reach_frac_of_arm_span",
                "crossed_hands", "target_hold_rating", "hips_vs_support_hand", "feet", "support_elbow_min_deg",
                "duration_s", "pose_missing"):
        assert key in m1, key
    assert m1["hand"] == "left" and m1["support_hand"] == "right" and m1["from"] == "H0" and m1["to"] == "H1"
    assert m1["from_role"] == "start" and m1["to_role"] is None
    assert m1["direction"]["vertical"] == "up" and m1["direction"]["lateral"] == "left"
    assert m1["target_hold_rating"] == 5 and m1["target_hold_quality"] == "terrible"
    assert m1["crossed_hands"] is False
    # hips 200 px to the right of the support hand (H0 at x=500) with a 1000 px arm span
    assert m1["pose_missing"] is False
    assert abs(m1["hips_offset_frac"] - 0.20) < 1e-9
    assert m1["hips_vs_support_hand"] == "+0.20 right"
    assert m1["feet"]["left"] == {"hold": "H0", "side_vs_hips": "left"} and m1["feet"]["right"] is None
    assert m1["support_elbow_min_deg"] == 150 and m1["duration_s"] == 1.0
    # move 2 starts at frame 10 (right hand still on H0 there): hips right of the left support hand on H1
    m2 = moves[1]
    assert m2["pose_missing"] is False and m2["hips_offset_frac"] > 0 and m2["hips_vs_support_hand"].endswith("right")
    assert m2["support_elbow_min_deg"] == 80
    # move 3 starts at frame 20 where the hips are not visible and no frame nearby has them
    m3 = moves[2]
    assert m3["pose_missing"] is True and m3["hips_vs_support_hand"] is None and m3["support_elbow_min_deg"] is None
    assert m3["direction"]["vertical"] == "up"
    assert m3["to_role"] == "finish"

    sugg = S["suggested_line"]["moves"]
    assert sugg and all("hips_vs_support_hand" not in m for m in sugg)
    assert all(m["direction"]["vertical"] in ("up", "sideways", "down") for m in sugg)
    assert all(m["direction"]["lateral"] in ("left", "right", "straight") for m in sugg)

    cmp_ = S["comparison"]
    assert cmp_["holds_only_in_your_climb"] == ["H1", "H2"] and cmp_["holds_only_in_suggested"] == ["H3", "H4"]
    assert cmp_["shared_holds"] == ["H0", "H5"] and cmp_["same_sequence"] is False and 0 < cmp_["improvement_frac"] <= 1
    assert cmp_["crux"]["move"] in (1, 2, 3) and cmp_["crux"]["reason"] and "explanation" in cmp_["crux"]

    # only frames 10 and 20 are tracked at 10 fps; a lone tracked frame does not split a gap,
    # so the whole 4 s clip counts as one gap
    assert S["tracking_gaps"] == [[0.0, 4.0]]
    assert S["tracking_gaps_text"] == "0.0–4.0 s"

    # without any pose, every move is marked as not seen and nothing crashes
    S2 = coach.summary_for_llm(R, hid, False)
    assert all(m["pose_missing"] for m in S2["your_climb"]["moves"]) and S2["tracking_gaps"] == []
    assert S2["your_climb"]["moves"][0]["duration_s"] is None


def test_tracking_gaps_and_text():
    frames = {}
    for f in range(0, 10):
        frames[f] = {"LEFT_WRIST": lm(0, 0), "RIGHT_WRIST": lm(0, 0)}
    for f in range(30, 40):
        frames[f] = {"LEFT_WRIST": lm(0, 0, 0.1), "RIGHT_WRIST": lm(0, 0, 0.9)}   # one wrist visible: tracked
    frames[19] = {"LEFT_WRIST": lm(0, 0), "RIGHT_WRIST": lm(0, 0)}               # single tracked frame: merged over
    frames[40] = {"LEFT_WRIST": lm(0, 0, 0.1), "RIGHT_WRIST": lm(0, 0, 0.1)}     # both wrists hidden: a 1-frame gap
    frames[41] = {"LEFT_WRIST": lm(0, 0), "RIGHT_WRIST": lm(0, 0)}
    pose = {"frames": frames}
    gaps = coach.tracking_gaps(pose, fps=10.0, n_frames=42)
    assert gaps == [(1.0, 3.0)]
    assert coach.tracking_gaps(pose, fps=10.0, n_frames=42, min_gap_s=0.0) == [(1.0, 3.0), (4.0, 4.1)]
    assert coach.gaps_text(gaps) == "1.0–3.0 s"
    assert coach.gaps_text([(0.0, 1.2), (3.4, 11.0)]) == "0.0–1.2 s, 3.4–11.0 s"
    assert coach.gaps_text([]) == ""
    # string frame keys (raw JSON) are handled too
    assert coach.tracking_gaps({"frames": {str(k): v for k, v in frames.items()}}, 10.0, 42) == [(1.0, 3.0)]


def test_parse_insights_splits_sections():
    reply = """## Your climb
- **Move 1:** Keep your hips under H0 before you go.
* Move 2: Bring your left foot up to H1 first.
1. Move 3: Rest here; your right arm was bent the whole time.

**Suggested line:**
- Move 1: Go to H3 with your right hand.
- Move 2: We could not see your feet here.
"""
    parsed = coach.parse_insights(reply)
    assert parsed["your_climb"] == ["Move 1: Keep your hips under H0 before you go.",
                                    "Move 2: Bring your left foot up to H1 first.",
                                    "Move 3: Rest here; your right arm was bent the whole time."]
    assert parsed["suggested"] == ["Move 1: Go to H3 with your right hand.",
                                   "Move 2: We could not see your feet here."]
    assert coach.parse_insights("") == {"your_climb": [], "suggested": []}
    assert coach.parse_insights(None) == {"your_climb": [], "suggested": []}
    loose = coach.parse_insights("Move 1: no headers at all")
    assert loose["your_climb"] == ["Move 1: no headers at all"] and loose["suggested"] == []


def test_llm_available_without_keys_is_falsy(monkeypatch, capsys):
    for k in ("XAI_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    st = coach.llm_available(timeout=1.0)
    assert isinstance(st, dict) and st["ok"] is False and st["provider"] is None and st["model"] is None
    assert "no API key" in st["reason"]
    assert not st                      # app.py's `if llm_available():` keeps working
    assert "[coach] LLM unavailable" in capsys.readouterr().out
    assert coach.llm_insights({"x": 1}) is None
    assert coach.llm_rewrite is coach.llm_insights


def test_load_env_does_not_override(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# comment\nSENDIT_TEST_A=one\nexport SENDIT_TEST_B='two'\nSENDIT_TEST_C=\nSENDIT_TEST_D=3\n")
    monkeypatch.setenv("SENDIT_TEST_D", "keep")
    for k in ("SENDIT_TEST_A", "SENDIT_TEST_B", "SENDIT_TEST_C"):
        monkeypatch.delenv(k, raising=False)
    coach.load_env(str(p))
    assert os.environ["SENDIT_TEST_A"] == "one" and os.environ["SENDIT_TEST_B"] == "two"
    assert "SENDIT_TEST_C" not in os.environ and os.environ["SENDIT_TEST_D"] == "keep"
    coach.load_env(str(tmp_path / "missing.env"))   # silently ignored
    for k in ("SENDIT_TEST_A", "SENDIT_TEST_B"):
        monkeypatch.delenv(k, raising=False)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
