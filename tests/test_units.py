"""Units, formatting and pixel-scale tests for sendit.units.
Run: python3 -m pytest tests -q
"""
import copy
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.units import (METRIC, IMPERIAL, HEIGHT_PROXY_FRAC, MINUS, DASH,
                          to_metres, from_metres, unit_label, format_length, format_delta,
                          px_per_metre, apply_body_inputs, relative_length, length_text)


def morph(height_proxy_px=850.0, arm_span_px=1000.0, leg_len_px=500.0):
    """Shape of morphology_from_pose() output, trimmed to the fields units.py reads."""
    return {
        "arm_span_px": arm_span_px,
        "shoulder_width_px": 200.0,
        "torso_px": 350.0,
        "leg_len_px": leg_len_px,
        "height_proxy_px": height_proxy_px,
        "ratios": {
            "arm_span_over_height_proxy": (arm_span_px / height_proxy_px) if height_proxy_px else None,
            "forearm_over_upper_arm": 0.9,
            "leg_over_arm_span": leg_len_px / arm_span_px,
        },
    }


# ---------------------------------------------------------------- conversions

def test_constants():
    assert METRIC == "metric" and IMPERIAL == "imperial"
    assert HEIGHT_PROXY_FRAC == 0.85


def test_to_metres_known_values():
    assert to_metres(100, METRIC) == pytest.approx(1.0)
    assert to_metres(172, METRIC) == pytest.approx(1.72)
    assert to_metres(12, IMPERIAL) == pytest.approx(0.3048)
    assert to_metres(72, IMPERIAL) == pytest.approx(1.8288)


def test_from_metres_known_values():
    assert from_metres(1.0, METRIC) == pytest.approx(100.0)
    assert from_metres(1.8288, IMPERIAL) == pytest.approx(72.0)


@pytest.mark.parametrize("units", [METRIC, IMPERIAL])
@pytest.mark.parametrize("value", [0.0, 1.0, 25.0, 63.0, 172.5, 300.0])
def test_round_trip(units, value):
    assert to_metres(from_metres(to_metres(value, units), units), units) == pytest.approx(to_metres(value, units))
    assert from_metres(to_metres(value, units), units) == pytest.approx(value)


def test_unit_label():
    assert unit_label(METRIC) == "cm"
    assert unit_label(IMPERIAL) == "in"


def test_unknown_units_rejected():
    for fn in (lambda: to_metres(1, "furlongs"), lambda: from_metres(1, "furlongs"),
               lambda: unit_label("furlongs"), lambda: format_length(1.0, "furlongs"),
               lambda: format_delta(1.0, "furlongs"), lambda: length_text(1.0, 1.0, 1.0, "furlongs")):
        with pytest.raises(ValueError):
            fn()


# ----------------------------------------------------------------- formatting

def test_format_length_metric():
    assert format_length(1.72, METRIC) == "1.72 m"
    assert format_length(1.0, METRIC) == "1.00 m"
    assert format_length(0.63, METRIC) == "63 cm"
    assert format_length(0.634, METRIC) == "63 cm"
    assert format_length(0.635, METRIC) == "64 cm"   # half rounds up, not to even


def test_format_length_imperial():
    assert format_length(1.727, IMPERIAL) == "5 ft 8 in"   # 68.0 in
    assert format_length(0.635, IMPERIAL) == "25 in"       # short lengths: inches only
    assert format_length(0.3048, IMPERIAL) == "12 in"
    assert format_length(0.9144, IMPERIAL) == "3 ft 0 in"  # 36 in: switch to ft + in
    assert format_length(35.4 * 0.0254, IMPERIAL) == "35 in"


def test_format_length_imperial_twelve_inch_carry():
    # 1.83 m = 72.05 in -> rounds to 72 -> exactly 6 ft, not "5 ft 12 in"
    assert format_length(1.83, IMPERIAL) == "6 ft 0 in"
    # 5 ft 11.6 in -> rounds to 72 in -> 6 ft 0 in
    assert format_length(71.6 * 0.0254, IMPERIAL) == "6 ft 0 in"
    # 2 ft 11.6 in -> rounds to 36 in -> 3 ft 0 in
    assert format_length(35.6 * 0.0254, IMPERIAL) == "3 ft 0 in"
    # 5 ft 11.4 in stays 5 ft 11 in
    assert format_length(71.4 * 0.0254, IMPERIAL) == "5 ft 11 in"


def test_format_length_none_and_nan():
    assert format_length(None, METRIC) == DASH
    assert format_length(None, IMPERIAL) == DASH
    assert format_length(float("nan"), METRIC) == DASH


def test_format_delta_uses_real_minus_sign():
    assert format_delta(-0.26, METRIC) == f"{MINUS}26 cm"
    assert format_delta(-0.254, IMPERIAL) == f"{MINUS}10 in"
    assert MINUS == "−"
    assert "-" not in format_delta(-0.26, METRIC)   # no ASCII hyphen


def test_format_delta_positive_and_zero():
    assert format_delta(0.26, METRIC) == "+26 cm"
    assert format_delta(0.254, IMPERIAL) == "+10 in"
    assert format_delta(0.0, METRIC) == "0 cm"
    assert format_delta(-0.001, METRIC) == "0 cm"   # rounds to zero: no sign


# --------------------------------------------------------------- pixel scale

def test_px_per_metre_from_height():
    m = morph(height_proxy_px=850.0)
    # 850 px proxy = 0.85 * 1.0 m -> 1000 px / m
    assert px_per_metre(m, 1.0, None) == pytest.approx(1000.0)
    assert px_per_metre(m, 1.7, None) == pytest.approx(850.0 / (HEIGHT_PROXY_FRAC * 1.7))


def test_px_per_metre_from_arm_span():
    m = morph(arm_span_px=1000.0)
    assert px_per_metre(m, None, 2.0) == pytest.approx(500.0)


def test_px_per_metre_height_wins_over_arm_span():
    m = morph(height_proxy_px=850.0, arm_span_px=1000.0)
    # arm span alone would give 500; height alone gives 1000; height wins
    assert px_per_metre(m, 1.0, 2.0) == pytest.approx(1000.0)


def test_px_per_metre_none_when_no_inputs():
    assert px_per_metre(morph(), None, None) is None
    assert px_per_metre(morph(), 0.0, None) is None
    assert px_per_metre(morph(), None, 0.0) is None


def test_px_per_metre_falls_back_when_height_proxy_missing():
    m = morph(height_proxy_px=None)
    assert px_per_metre(m, 1.8, None) is None
    assert px_per_metre(m, 1.8, 2.0) == pytest.approx(500.0)


# ----------------------------------------------------------- apply_body_inputs

def test_apply_body_inputs_no_inputs_is_a_copy():
    m = morph()
    out, scale = apply_body_inputs(m, None, None)
    assert scale is None
    assert out == m
    assert out is not m and out["ratios"] is not m["ratios"]


def test_apply_body_inputs_only_height_changes_nothing():
    m = morph()
    before = copy.deepcopy(m)
    out, scale = apply_body_inputs(m, 1.0, None)
    assert scale == pytest.approx(1000.0)
    assert out == before
    assert m == before


def test_apply_body_inputs_only_arm_span_changes_nothing():
    m = morph()
    before = copy.deepcopy(m)
    out, scale = apply_body_inputs(m, None, 1.9)
    assert scale == pytest.approx(1000.0 / 1.9)
    assert out == before
    assert m == before


def test_apply_body_inputs_both_overrides_arm_span():
    m = morph(height_proxy_px=850.0, arm_span_px=1000.0, leg_len_px=500.0)
    before = copy.deepcopy(m)
    out, scale = apply_body_inputs(m, 1.0, 1.8)
    assert scale == pytest.approx(1000.0)             # from height
    assert out["arm_span_px"] == pytest.approx(1800.0)  # user's arm span wins
    assert m == before                                 # input untouched
    # dependent ratios follow the new arm span
    assert out["ratios"]["arm_span_over_height_proxy"] == pytest.approx(1800.0 / 850.0)
    assert out["ratios"]["leg_over_arm_span"] == pytest.approx(500.0 / 1800.0)
    assert out["ratios"]["forearm_over_upper_arm"] == before["ratios"]["forearm_over_upper_arm"]
    # everything else is untouched
    for k in ("shoulder_width_px", "torso_px", "leg_len_px", "height_proxy_px"):
        assert out[k] == before[k]


def test_apply_body_inputs_both_but_no_height_proxy_keeps_video_arm_span():
    m = morph(height_proxy_px=None, arm_span_px=1000.0)
    out, scale = apply_body_inputs(m, 1.8, 2.0)
    assert scale == pytest.approx(500.0)
    assert out["arm_span_px"] == pytest.approx(1000.0)


# ------------------------------------------------------------------ UI text

def test_relative_length():
    assert relative_length(490.0, 1000.0) == "0.49 × arm span"
    assert relative_length(490.0, 0.0) == DASH
    assert relative_length(490.0, None) == DASH


def test_length_text_uses_real_units_when_scaled():
    assert length_text(490.0, 1000.0, 1000.0, METRIC) == "49 cm"
    assert length_text(1720.0, 1000.0, 1000.0, METRIC) == "1.72 m"
    assert length_text(1727.0, 1000.0, 1000.0, IMPERIAL) == "5 ft 8 in"


def test_length_text_falls_back_to_relative():
    assert length_text(490.0, None, 1000.0, METRIC) == "0.49 × arm span"
    assert length_text(490.0, 0.0, 1000.0, IMPERIAL) == "0.49 × arm span"


def test_length_text_none():
    assert length_text(None, 1000.0, 1000.0, METRIC) == DASH
    assert length_text(None, None, 1000.0, IMPERIAL) == DASH


def test_user_facing_strings_avoid_banned_words():
    banned = ["px", "state", "beam", "objective", "cost", "observed", "optimized", "proxy"]
    samples = [format_length(1.72, METRIC), format_length(1.83, IMPERIAL), format_length(None, METRIC),
               format_delta(-0.26, METRIC), format_delta(0.1, IMPERIAL),
               relative_length(490.0, 1000.0), length_text(490.0, None, 1000.0, METRIC),
               length_text(490.0, 1000.0, 1000.0, IMPERIAL), unit_label(METRIC), unit_label(IMPERIAL)]
    for s in samples:
        low = s.lower()
        for w in banned:
            assert w not in low, f"{w!r} appears in {s!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
