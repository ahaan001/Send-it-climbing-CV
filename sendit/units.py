"""Units and pixel scale for the analysis.

Every length inside the analysis is a video pixel. The climber can
optionally tell us their height or arm span; either one turns pixels into
real-world lengths. This module owns that conversion plus the metric /
imperial formatting used by the UI.

Precedence for the pixel scale (px_per_metre):
    1. height   (video height proxy / (HEIGHT_PROXY_FRAC * height))
    2. arm span (video arm span / arm span)
    3. no scale -> lengths are shown relative to the arm span instead.
"""
from __future__ import annotations

import copy
import math

METRIC = "metric"
IMPERIAL = "imperial"

# morphology_from_pose measures "height" as torso + legs (shoulder midpoint to
# hip midpoint, plus thigh and shin). That leaves out the head, neck and feet,
# which together are roughly 15% of standing height, so the proxy is about
# 0.85 of the height a climber would tell us.
HEIGHT_PROXY_FRAC = 0.85

_M_PER_INCH = 0.0254
_CM_PER_M = 100.0
_IN_PER_FT = 12
# Imperial switches from plain inches to feet + inches at 3 ft, the
# counterpart of metric switching from cm to m at 1 m.
_FT_IN_SWITCH = 3 * _IN_PER_FT

MINUS = "−"   # proper minus sign for deltas
DASH = "—"    # em dash for "no value"


def _check_units(units: str) -> str:
    if units not in (METRIC, IMPERIAL):
        raise ValueError(f"units must be {METRIC!r} or {IMPERIAL!r}, got {units!r}")
    return units


def _round_half_up(x: float) -> int:
    """Round to the nearest integer, halves away from zero (Python's round()
    uses banker's rounding, which surprises people in a UI)."""
    return int(math.copysign(math.floor(abs(x) + 0.5), x))


# --------------------------------------------------------------------------
# conversions
# --------------------------------------------------------------------------

def to_metres(value: float, units: str) -> float:
    """User input -> metres. Metric input is centimetres, imperial is inches."""
    _check_units(units)
    if units == METRIC:
        return float(value) / _CM_PER_M
    return float(value) * _M_PER_INCH


def from_metres(m: float, units: str) -> float:
    """Metres -> centimetres (metric) or inches (imperial)."""
    _check_units(units)
    if units == METRIC:
        return float(m) * _CM_PER_M
    return float(m) / _M_PER_INCH


def unit_label(units: str) -> str:
    """Short label for input boxes: 'cm' or 'in'."""
    _check_units(units)
    return "cm" if units == METRIC else "in"


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def format_length(m: float | None, units: str) -> str:
    """Human-friendly length.

    metric:   '1.72 m' when at least 1 m, else '63 cm'
    imperial: '5 ft 8 in' when at least 3 ft (inches rounded, 12 in carries
              into feet), else inches only, e.g. '25 in' (the imperial
              twin of '63 cm')
    None:     an em dash
    """
    _check_units(units)
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return DASH
    m = float(m)
    if units == METRIC:
        if m >= 1.0:
            return f"{m:.2f} m"
        return f"{_round_half_up(m * _CM_PER_M)} cm"
    total_in = _round_half_up(m / _M_PER_INCH)
    if total_in >= _FT_IN_SWITCH:
        ft, inches = divmod(total_in, _IN_PER_FT)
        return f"{ft} ft {inches} in"
    return f"{total_in} in"


def format_delta(m: float, units: str) -> str:
    """Signed difference in cm or inches, e.g. '−26 cm', '+3 in', '0 cm'."""
    _check_units(units)
    n = _round_half_up(from_metres(m, units))
    label = unit_label(units)
    if n < 0:
        return f"{MINUS}{-n} {label}"
    if n > 0:
        return f"+{n} {label}"
    return f"0 {label}"


# --------------------------------------------------------------------------
# pixel scale from the climber's own numbers
# --------------------------------------------------------------------------

def px_per_metre(morph: dict, height_m: float | None, arm_span_m: float | None) -> float | None:
    """Pixels per metre from the climber's height (preferred) or arm span.

    Height wins when both are given. If height is given but the video did
    not yield a height proxy (legs never visible), arm span is used instead.
    Returns None when neither input can be matched to a video measurement.
    """
    if height_m is not None and height_m > 0:
        proxy = morph.get("height_proxy_px")
        if proxy:
            return float(proxy) / (HEIGHT_PROXY_FRAC * float(height_m))
    if arm_span_m is not None and arm_span_m > 0:
        span = morph.get("arm_span_px")
        if span:
            return float(span) / float(arm_span_m)
    return None


def apply_body_inputs(morph: dict, height_m: float | None,
                      arm_span_m: float | None) -> tuple[dict, float | None]:
    """Deep-copy morph and return (morph, px_per_metre).

    When BOTH height and arm span are given, the scale comes from height and
    the climber's stated arm span overrides the video measurement
    (arm_span_px = arm_span_m * scale). Ratios that depend on arm span are
    recomputed so the copy stays self-consistent. When only one input is
    given the copy is identical to the input; only the scale is returned.
    """
    out = copy.deepcopy(morph)
    scale = px_per_metre(morph, height_m, arm_span_m)
    both = (height_m is not None and height_m > 0
            and arm_span_m is not None and arm_span_m > 0)
    if both and scale is not None:
        span_px = float(arm_span_m) * scale
        out["arm_span_px"] = span_px
        ratios = out.get("ratios")
        if isinstance(ratios, dict):
            proxy = out.get("height_proxy_px")
            leg = out.get("leg_len_px")
            if "arm_span_over_height_proxy" in ratios:
                ratios["arm_span_over_height_proxy"] = (span_px / float(proxy)) if proxy else None
            if "leg_over_arm_span" in ratios:
                ratios["leg_over_arm_span"] = (float(leg) / span_px) if leg else None
    return out, scale


# --------------------------------------------------------------------------
# text for the UI
# --------------------------------------------------------------------------

def relative_length(px: float, arm_span_px: float) -> str:
    """Length as a fraction of the climber's arm span, e.g. '0.49 × arm span'."""
    if px is None or not arm_span_px or arm_span_px <= 0:
        return DASH
    return f"{float(px) / float(arm_span_px):.2f} × arm span"


def length_text(px: float | None, px_per_m: float | None, arm_span_px: float, units: str) -> str:
    """Real units when a pixel scale exists, else relative to arm span."""
    _check_units(units)
    if px is None:
        return DASH
    if px_per_m:
        return format_length(float(px) / float(px_per_m), units)
    return relative_length(px, arm_span_px)
