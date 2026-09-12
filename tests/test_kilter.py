"""Kilter light rules: hue -> colour/role, route warnings, single-image
lit-hold detection, and the start/finish/foot setup the optimizer needs."""
import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sendit.holds import make_hold
from sendit import kilter
from sendit.kilter import (ANGLE_SCALE_DEG, HUE_BANDS, ROLE_BY_COLOUR, difficulty_multiplier,
                           colour_from_hue, role_from_hue, assign_roles, role_warnings,
                           detect_route_holds, kilter_setup)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKGROUND = os.path.join(ROOT, "demo_assets", "kilter", "cache", "background.png")

BANNED = ["state", "A*", "Dijkstra", "beam", "envelope", "objective", "px", "symmetric difference",
          "normalised", "proxy", "feasible edge", "joint states", "homography", "observed",
          "optimized", "beta", "cost"]


# ---------------------------------------------------------------- hue -> colour / role

@pytest.mark.parametrize("hue,colour,role", [
    (175, "pink", "finish"),   # wrap-around, top of the circle
    (3, "pink", "finish"),     # wrap-around, bottom of the circle
    (15, "orange", "foot"),
    (30, "yellow", "foot"),
    (60, "green", "start"),
    (100, "blue", None),
    (140, "purple", "finish"),
])
def test_hue_to_colour_and_role(hue, colour, role):
    assert colour_from_hue(hue) == colour
    assert role_from_hue(hue) == role
    assert ROLE_BY_COLOUR[colour] == role


def test_hue_bands_cover_every_hue_exactly_once():
    for h in range(180):
        hits = [c for c, ranges in HUE_BANDS.items() for lo, hi in ranges if lo <= h <= hi]
        assert len(hits) == 1, f"hue {h} matched {hits}"
    assert set(HUE_BANDS) == set(ROLE_BY_COLOUR)


def test_hue_accepts_floats_numpy_and_wraps():
    assert colour_from_hue(np.int64(60)) == "green"
    assert colour_from_hue(59.6) == "green"
    assert colour_from_hue(180) == colour_from_hue(0)   # 180 wraps to 0
    with pytest.raises(ValueError):
        colour_from_hue(None)


def test_assign_roles_sets_colour_and_role_but_keeps_holds_without_hue():
    holds = [make_hold(0, 10, 900, hue=60), make_hold(1, 50, 500, hue=100),
             make_hold(2, 90, 100, hue=140), make_hold(3, 70, 700, role="foot", grip=4),
             make_hold(4, 30, 300, hue=15, role="finish")]
    out = assign_roles(holds)
    assert out is holds
    assert (holds[0]["colour"], holds[0]["role"]) == ("green", "start")
    assert (holds[1]["colour"], holds[1]["role"]) == ("blue", None)
    assert (holds[2]["colour"], holds[2]["role"]) == ("purple", "finish")
    assert "colour" not in holds[3] and holds[3]["role"] == "foot"   # no hue: untouched
    assert (holds[4]["colour"], holds[4]["role"]) == ("orange", "foot")  # hue wins over old role


# ---------------------------------------------------------------- warnings

def _route(n_start, n_finish, n_hand=2):
    holds, i = [], 0
    for _ in range(n_start):
        holds.append(make_hold(i, 100 + 50 * i, 900, role="start")); i += 1
    for _ in range(n_hand):
        holds.append(make_hold(i, 100 + 50 * i, 500)); i += 1
    for _ in range(n_finish):
        holds.append(make_hold(i, 100 + 50 * i, 100, role="finish")); i += 1
    return holds


@pytest.mark.parametrize("n_start,n_finish,expect_start,expect_finish", [
    (0, 1, True, False),
    (1, 1, False, False),
    (2, 2, False, False),
    (3, 1, True, False),
    (1, 0, False, True),
    (1, 2, False, False),
    (1, 3, False, True),
    (0, 0, True, True),
    (3, 3, True, True),
])
def test_role_warnings(n_start, n_finish, expect_start, expect_finish):
    w = role_warnings(_route(n_start, n_finish))
    starts = [m for m in w if "start" in m]
    finishes = [m for m in w if "finish" in m]
    assert bool(starts) == expect_start
    assert bool(finishes) == expect_finish
    assert len(w) == int(expect_start) + int(expect_finish)
    if n_start > 2:
        assert f"{n_start} start holds" in starts[0]
    if n_finish > 2:
        assert f"{n_finish} finish holds" in finishes[0]


def test_role_warnings_plain_language():
    for msg in role_warnings(_route(0, 0)) + role_warnings(_route(3, 3)):
        low = msg.lower()
        for word in BANNED:
            assert word.lower() not in low, f"{word!r} in {msg!r}"


def test_role_warnings_ignore_off_route_holds():
    holds = _route(3, 1)
    holds[0]["on_route"] = False
    assert role_warnings(holds) == []


# ---------------------------------------------------------------- angle multiplier

def test_difficulty_multiplier():
    assert difficulty_multiplier(0) == 1.0
    assert abs(difficulty_multiplier(ANGLE_SCALE_DEG) - math.e) < 1e-9
    assert abs(difficulty_multiplier(30) - math.e) < 1e-9
    vals = [difficulty_multiplier(a) for a in range(0, 71, 5)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    assert difficulty_multiplier(-30) < 1.0


# ---------------------------------------------------------------- detection on a synthetic board

GREEN = (0, 255, 0)      # BGR; hue 60
BLUE = (255, 0, 0)       # hue 120
ORANGE = (0, 128, 255)   # hue 15
PINK = (200, 0, 255)     # hue 156

DISCS = [  # (x, y, bgr, expected role), bottom-to-top y decreasing
    (250, 1050, GREEN, "start"),
    (550, 1040, GREEN, "start"),
    (300, 750, BLUE, None),
    (500, 700, BLUE, None),
    (150, 900, ORANGE, "foot"),
    (400, 200, PINK, "finish"),
]


def _hue_of(bgr):
    return int(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def test_synthetic_disc_colours_land_in_the_right_bands():
    assert colour_from_hue(_hue_of(GREEN)) == "green"
    assert colour_from_hue(_hue_of(BLUE)) == "blue"
    assert colour_from_hue(_hue_of(ORANGE)) == "orange"
    assert colour_from_hue(_hue_of(PINK)) == "pink"


def _synthetic_board(radius=18):
    img = np.zeros((1200, 800, 3), np.uint8)   # 800 wide x 1200 tall, matte black
    for x, y, bgr, _ in DISCS:
        cv2.circle(img, (x, y), radius, bgr, -1)
    return img


def test_detect_route_holds_synthetic():
    holds = detect_route_holds(_synthetic_board())
    assert len(holds) == 6
    assert [h["id"] for h in holds] == list(range(6))
    ys = [h["y"] for h in holds]
    assert ys == sorted(ys, reverse=True)           # bottom of the wall first
    for h in holds:
        assert h["source"] == "led" and h["hue"] is not None and "colour" in h
        assert h["label"] == f"H{h['id']}" and h["on_route"]
    # each drawn disc is recovered within a couple of pixels, with the right role
    for x, y, bgr, role in DISCS:
        near = [h for h in holds if np.hypot(h["x"] - x, h["y"] - y) < 3]
        assert len(near) == 1, (x, y)
        assert near[0]["role"] == role
        assert near[0]["colour"] == colour_from_hue(_hue_of(bgr))
    roles = [h["role"] for h in holds]
    assert roles.count("start") == 2 and roles.count(None) == 2
    assert roles.count("foot") == 1 and roles.count("finish") == 1


def test_detect_route_holds_merges_close_blobs():
    img = np.zeros((1200, 800, 3), np.uint8)
    cv2.circle(img, (300, 900), 8, GREEN, -1)
    cv2.circle(img, (330, 900), 8, GREEN, -1)   # 30 px away: two blobs
    cv2.circle(img, (400, 300), 12, PINK, -1)
    assert len(detect_route_holds(img)) == 3                     # default 2 % of 1200 = 24 px
    merged = detect_route_holds(img, merge_dist_px=40)
    assert len(merged) == 2
    g = [h for h in merged if h["role"] == "start"]
    assert len(g) == 1 and abs(g[0]["x"] - 315) < 1.5


def test_detect_route_holds_without_a_dark_board():
    """No board mask worth trusting (bright grey wall): fall back to the whole image."""
    img = np.full((1200, 800, 3), 200, np.uint8)
    for x, y, bgr, _ in DISCS:
        cv2.circle(img, (x, y), 18, bgr, -1)
    holds = detect_route_holds(img)
    assert len(holds) == 6


def test_kilter_setup_from_synthetic_board():
    holds = detect_route_holds(_synthetic_board())
    setup = kilter_setup(holds)
    by_id = {h["id"]: h for h in holds}
    assert len(setup["start_ids"]) == 2
    xs = [by_id[i]["x"] for i in setup["start_ids"]]
    assert xs == sorted(xs)
    assert setup["start_state"] == (setup["start_ids"][0], setup["start_ids"][1])
    assert by_id[setup["start_state"][0]]["x"] < by_id[setup["start_state"][1]]["x"]
    assert len(setup["finish_ids"]) == 1 and by_id[setup["finish_ids"][0]]["role"] == "finish"
    assert len(setup["foot_ids"]) == 1 and by_id[setup["foot_ids"][0]]["role"] == "foot"
    assert setup["match_finish"] is True
    assert setup["warnings"] == []


def test_kilter_setup_edge_cases():
    holds = _route(1, 0)
    s = kilter_setup(holds)
    assert s["start_state"] == (0, 0)          # one start hold: matched
    assert s["finish_ids"] == [] and s["match_finish"] is False
    assert s["foot_ids"] == []
    assert any("finish" in w for w in s["warnings"])

    s = kilter_setup(_route(0, 2))
    assert s["start_state"] is None and s["start_ids"] == []
    assert len(s["finish_ids"]) == 2 and s["match_finish"] is True

    holds = _route(3, 1)                       # too many starts: leftmost and rightmost
    s = kilter_setup(holds)
    assert s["start_state"] == (0, 2)
    holds[2]["on_route"] = False               # editor switched one off
    assert kilter_setup(holds)["start_state"] == (0, 1)

    holds = [make_hold(0, 300, 900, role="start"), make_hold(1, 100, 880, role="start")]
    assert kilter_setup(holds)["start_ids"] == [1, 0]   # sorted by x, not id


# ---------------------------------------------------------------- real CruxCam frame

@pytest.mark.skipif(not os.path.exists(BACKGROUND), reason="demo cache not present")
def test_detect_route_holds_on_demo_background():
    img = cv2.imread(BACKGROUND)
    assert img is not None
    holds = detect_route_holds(img)
    # On this camera the true start holds read as hue ~14 (orange): known
    # miscalibration, so only the count and the schema are asserted here.
    assert len(holds) >= 10
    for h in holds:
        assert h.get("hue") is not None
        assert h["colour"] in HUE_BANDS
        assert h["source"] == "led"
    assert [h["id"] for h in holds] == list(range(len(holds)))
    ys = [h["y"] for h in holds]
    assert ys == sorted(ys, reverse=True)
