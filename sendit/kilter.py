"""Kilter Board light rules -> hold roles.

On a Kilter Board the route is shown by lit LEDs, and the LED colour tells
you what each hold is for:

  green          start holds (both hands begin on them; matched if only one)
  blue           hands and feet
  orange/yellow  feet only (never grabbed)
  pink/purple    finish (both must be held when two are lit)
  unlit          off-route

Green holds may also be used as feet later in the climb; the optimizer
already lets feet use any on-route hold, so nothing special is needed here.

This module maps LED hues (OpenCV 0-179) to those roles, finds the lit holds
on a single climber-free image, and packages the start / finish / foot ids the
optimizer needs. Hold dicts follow sendit.holds.make_hold; this module adds
``"colour"`` and overwrites ``"role"`` for holds that carry a ``"hue"``.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

from .holds import make_hold

# src/ holds the original hackathon detector (hold_detection.py); same path
# handling as sendit/holds.py so `from hold_detection import ...` works.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ----------------------------------------------------------------------------
# Wall angle -> display-only difficulty multiplier
# ----------------------------------------------------------------------------

ANGLE_SCALE_DEG = 30.0  # every 30 degrees of overhang multiplies the difficulty by e


def difficulty_multiplier(angle_deg: float) -> float:
    """Display-only factor for how much harder the wall angle makes a climb.
    0 degrees -> 1.0; ANGLE_SCALE_DEG degrees -> e; strictly increasing."""
    return math.exp(float(angle_deg) / ANGLE_SCALE_DEG)


# ----------------------------------------------------------------------------
# LED colour bands (OpenCV hue, 0-179, inclusive ranges)
# ----------------------------------------------------------------------------

HUE_MIN = 0
HUE_MAX = 179

# Red/pink wraps around the top of the hue circle, so it has two ranges.
HUE_BANDS: dict[str, list[tuple[int, int]]] = {
    "pink": [(0, 7), (150, 179)],
    "orange": [(8, 22)],
    "yellow": [(23, 37)],
    "green": [(38, 85)],
    "blue": [(86, 130)],
    "purple": [(131, 149)],
}

# None means a plain hand hold (hands and feet), matching how the rest of the
# code stores roles: "start" | "finish" | "foot" | None.
ROLE_BY_COLOUR: dict[str, str | None] = {
    "green": "start",
    "blue": None,
    "orange": "foot",
    "yellow": "foot",
    "pink": "finish",
    "purple": "finish",
}

# A Kilter route has one or two start holds and one or two finish holds.
MIN_ROLE_HOLDS = 1
MAX_ROLE_HOLDS = 2


def _wrap_hue(hue) -> int:
    """Round to an int and wrap into 0..HUE_MAX (hue is circular)."""
    if hue is None:
        raise ValueError("hue is None; use assign_roles() to skip holds without a hue")
    return int(round(float(hue))) % (HUE_MAX + 1)


def colour_from_hue(hue) -> str:
    """Name of the LED colour band that contains this OpenCV hue."""
    h = _wrap_hue(hue)
    for colour, ranges in HUE_BANDS.items():
        for lo, hi in ranges:
            if lo <= h <= hi:
                return colour
    raise ValueError(f"hue {h} is not covered by HUE_BANDS")  # bands cover 0..179


def role_from_hue(hue) -> str | None:
    """Kilter role for this hue: 'start' | 'finish' | 'foot' | None (hand hold)."""
    return ROLE_BY_COLOUR[colour_from_hue(hue)]


def assign_roles(holds: list) -> list:
    """Set h["colour"] and h["role"] in place from h["hue"]. Holds without a
    hue (manual holds, for example) keep whatever role they already have.
    Returns the same list."""
    for h in holds:
        hue = h.get("hue")
        if hue is None:
            continue
        colour = colour_from_hue(hue)
        h["colour"] = colour
        h["role"] = ROLE_BY_COLOUR[colour]
    return holds


# ----------------------------------------------------------------------------
# Route sanity checks (plain language, shown to the climber)
# ----------------------------------------------------------------------------

def _live(holds: list) -> list:
    """Holds still on the route (the editor can switch one off)."""
    return [h for h in holds if h.get("on_route", True)]


def _with_role(holds: list, role: str) -> list:
    """On-route holds with this role, sorted left to right."""
    return sorted((h for h in _live(holds) if h.get("role") == role), key=lambda h: h["x"])


def role_warnings(holds: list) -> list[str]:
    """Plain-language warnings when the start or finish holds do not look like
    a Kilter route (zero, or more than two, of either)."""
    out = []
    n_start = len(_with_role(holds, "start"))
    n_finish = len(_with_role(holds, "finish"))
    if n_start < MIN_ROLE_HOLDS:
        out.append("We did not find a start hold. A Kilter route has one or two. Mark them below.")
    elif n_start > MAX_ROLE_HOLDS:
        out.append(f"We found {n_start} start holds. A Kilter route has one or two. Fix the roles below.")
    if n_finish < MIN_ROLE_HOLDS:
        out.append("We did not find a finish hold. A Kilter route has one or two. Mark it below.")
    elif n_finish > MAX_ROLE_HOLDS:
        out.append(f"We found {n_finish} finish holds. A Kilter route has one or two. Fix the roles below.")
    return out


# ----------------------------------------------------------------------------
# Lit-hold detection on one image
# ----------------------------------------------------------------------------

MERGE_FRAC = 0.02          # default merge distance: 2 % of the image's larger side
MIN_BOARD_FRAC = 0.10      # board mask must cover at least this much of the image to be trusted
MIN_BLOBS_IN_BOARD = 3     # ...and contain at least this many lit blobs


def _merge_blobs(blobs: list, merge_dist: float) -> list:
    """Merge lit blobs closer than merge_dist (one LED often splits into a big
    glow plus a small fragment). Largest blob first; a merged blob keeps the
    largest member's hue, the area-weighted centre and the summed area.
    Blobs are (cx, cy, hue, area) as returned by detect_lit_blobs."""
    merged = []  # each: [cx, cy, hue, area, area_of_largest_member]
    for cx, cy, hue, area in sorted(blobs, key=lambda b: -b[3]):
        for m in merged:
            if np.hypot(cx - m[0], cy - m[1]) < merge_dist:
                tot = m[3] + area
                m[0] = (m[0] * m[3] + cx * area) / tot
                m[1] = (m[1] * m[3] + cy * area) / tot
                m[3] = tot
                break
        else:
            merged.append([float(cx), float(cy), int(hue), float(area), float(area)])
    return [(m[0], m[1], m[2], m[3]) for m in merged]


def detect_route_holds(image_bgr: np.ndarray, merge_dist_px: float | None = None) -> list[dict]:
    """Lit holds on ONE climber-free image of a Kilter-style board.

    Uses the board mask only when it is trustworthy (covers >= 10 % of the
    image and holds >= 3 lit blobs); otherwise detects on the whole image.
    Nearby blobs are merged, ids run bottom-to-top like holds.detect_led_holds,
    and roles come from the LED colour (assign_roles)."""
    from hold_detection import detect_lit_blobs, detect_board_bbox  # src/ module

    h_img, w_img = image_bgr.shape[:2]
    if merge_dist_px is None:
        merge_dist_px = MERGE_FRAC * max(h_img, w_img)

    blobs = None
    board_mask = detect_board_bbox(image_bgr)
    if board_mask is not None and (board_mask > 0).mean() >= MIN_BOARD_FRAC:
        inside = detect_lit_blobs(image_bgr, board_mask=board_mask)
        if len(inside) >= MIN_BLOBS_IN_BOARD:
            blobs = inside
    if blobs is None:
        blobs = detect_lit_blobs(image_bgr)

    merged = _merge_blobs(blobs, merge_dist_px)
    merged.sort(key=lambda b: -b[1])  # bottom of the wall first
    holds = [make_hold(i, cx, cy, source="led", hue=int(hue), area=float(area))
             for i, (cx, cy, hue, area) in enumerate(merged)]
    return assign_roles(holds)


# ----------------------------------------------------------------------------
# What the optimizer needs
# ----------------------------------------------------------------------------

def kilter_setup(holds: list) -> dict:
    """Start / finish / foot ids from the roles, plus the two-hand start pair.

    start_ids, finish_ids, foot_ids: on-route holds with that role, sorted
    left to right by x. start_state: (left_id, right_id) = the leftmost and
    rightmost start holds, or the same id twice when only one is lit; None
    when there are no start holds. match_finish: True when any finish hold is
    lit (the optimizer should end with both hands on the finish)."""
    start_ids = [h["id"] for h in _with_role(holds, "start")]
    finish_ids = [h["id"] for h in _with_role(holds, "finish")]
    foot_ids = [h["id"] for h in _with_role(holds, "foot")]
    if not start_ids:
        start_state = None
    else:
        start_state = (start_ids[0], start_ids[-1])
    return {
        "start_ids": start_ids,
        "finish_ids": finish_ids,
        "foot_ids": foot_ids,
        "start_state": start_state,
        "match_finish": bool(finish_ids),
        "warnings": role_warnings(holds),
    }
