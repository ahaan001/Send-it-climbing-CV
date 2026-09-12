"""Personalized beta optimization: limb-generic state graph + exact / budgeted search.

State  s = (hold under LEFT hand, hold under RIGHT hand[, LEFT foot, RIGHT foot])
       Hands-only mode uses the first two limbs; four-limb mode adds the feet,
       where a foot may be None (cut loose / smearing).
Action = move ONE limb to another on-route hold (a foot may also cut loose)
Edge   feasible iff: the hand-to-hand span after the move fits inside the
       climber's reach envelope (a fraction of THEIR measured arm span), the
       hand doesn't drop too far, the target is on the route (and not
       foot-only for hands); every placed foot sits inside the leg window
       below the hands (0.35..1.25 x body-height proxy, +-0.6 laterally) and
       both feet are within 1.3 x leg length of each other.
Cost   dimensionless per move; hand moves:
         w_r * ReachCost + w_g * GripCost + w_m + w_t * Travel + w_d * Direction
         + w_x * Cross + w_f * FootSupport + w_hang * CutLoose
       foot moves: w_fmove + w_ftravel * FootTravel + 0.5 * w_g * Grip + w_fcross * FeetCrossed
Search exact A* (admissible bound -> same optimum as Dijkstra) when the joint
       state estimate is small; on dense walls a budgeted A* that falls back to
       beam search, reported as approximate. The same cost function scores the
       climber's observed sequence, so comparisons are apples to apples.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, replace, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

LEFT, RIGHT = "LEFT", "RIGHT"
LEFT_FOOT, RIGHT_FOOT = "LEFT_FOOT", "RIGHT_FOOT"
HANDS = (LEFT, RIGHT)
FEET = (LEFT_FOOT, RIGHT_FOOT)
LIMBS = HANDS + FEET
LIMB_INDEX = {LEFT: 0, RIGHT: 1, LEFT_FOOT: 2, RIGHT_FOOT: 3}
TERM_KEYS = ["reach", "grip", "move", "travel", "direction", "cross", "foot", "hang", "fmove", "ftravel", "fcross"]


@dataclass
class Climber:
    arm_span_px: float
    leg_len_px: Optional[float] = None
    torso_px: Optional[float] = None
    label: str = "Measured climber"
    scale: float = 1.0  # 1.0 = as measured; 0.85 = simulated 85 % morphology

    @classmethod
    def from_morphology(cls, m: dict, label="Measured climber"):
        return cls(arm_span_px=m["arm_span_px"], leg_len_px=m.get("leg_len_px"),
                   torso_px=m.get("torso_px"), label=label)

    def scaled(self, factor: float, label: Optional[str] = None) -> "Climber":
        return Climber(arm_span_px=self.arm_span_px * factor,
                       leg_len_px=self.leg_len_px * factor if self.leg_len_px else None,
                       torso_px=self.torso_px * factor if self.torso_px else None,
                       label=label or f"Simulated climber ({factor:.0%} reach)", scale=self.scale * factor)

    @property
    def body_height_proxy_px(self) -> float:
        if self.leg_len_px and self.torso_px:
            return self.leg_len_px + self.torso_px
        return 0.75 * self.arm_span_px  # torso+legs ~ 3/4 of height ~ 3/4 of span

    @property
    def leg_px(self) -> float:
        return self.leg_len_px or 0.45 * self.arm_span_px


@dataclass
class Weights:
    w_reach: float = 1.0     # (normalized reach / r_ref)^2 : longer reaches cost super-linearly more
    r_ref: float = 0.4       # normalized reach (hand-to-hand span / arm span) that costs exactly w_reach
    w_grip: float = 0.8      # (grip rating - 1) / 4 of the target hold: 1 -> 0, 5 -> 1
    w_move: float = 0.35     # fixed cost of every hand movement (re-grip / instability moment)
    w_travel: float = 0.25   # moving-hand travel distance / arm span
    w_dir: float = 0.4       # sideways or downward travel (0 for a straight-up move)
    w_cross: float = 0.5     # hands crossed after the move
    w_foot: float = 0.4      # hands-only: no hold in the leg window below the target; four-limb: support deficit after the move
    # four-limb terms (no effect in hands-only mode)
    w_hang: float = 1.5      # moving a hand while feet are cut loose (campus move), per unplaced foot / 2
    w_fmove: float = 0.25    # fixed cost of every foot movement
    w_ftravel: float = 0.3   # foot travel / leg length
    w_fcross: float = 0.3    # feet crossed

    @classmethod
    def geometric(cls):
        """Pure shortest-geometric-path objective (for the 'why not just
        the shortest path?' comparison)."""
        return cls(w_reach=0, w_grip=0, w_move=0, w_travel=1.0, w_dir=0, w_cross=0, w_foot=0,
                   w_hang=0, w_fmove=0, w_ftravel=1.0, w_fcross=0)


@dataclass
class Feasibility:
    max_reach_frac: float = 0.85   # hand-to-hand span after the move, as a fraction of arm span
    max_down_frac: float = 0.20    # a hand may drop at most this fraction of arm span
    max_travel_frac: float = 1.00  # moving hand travel cap (fraction of arm span); beyond this is a dyno
    match_finish: bool = False     # require both hands on the finish hold
    foot_window: Tuple[float, float] = (0.35, 1.25)  # foot below the hands' midpoint, x body-height proxy
    foot_lateral: float = 0.6      # |dx| from the hands' midpoint, x body-height proxy
    feet_spread_frac: float = 1.3  # max distance between feet, x leg length
    ground_margin: float = 0.15    # ground assumed at lowest hold + this x body-height proxy (feet may rest on it)


def dist(a, b) -> float:
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def grip_penalty(grip: int) -> float:
    return (float(np.clip(grip, 1, 5)) - 1.0) / 4.0


def hand_holds(holds: list) -> list:
    """Holds a hand may use: on-route and not marked foot-only."""
    return [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]


def foot_holds(holds: list) -> list:
    """Holds a foot may use: any on-route hold, foot-only ones included."""
    return [h for h in holds if h.get("on_route", True)]


def foot_support_costs(holds: list, climber: Climber, F: Optional[Feasibility] = None) -> Dict[int, float]:
    """Hands-only lower-body proxy: for each hold, is there ANY hold inside
    the window where this climber's feet could be while a hand is on it?
    0 = supported, 1 = nothing usable (graded in between)."""
    F = F or Feasibility()
    H = climber.body_height_proxy_px
    lo, hi = F.foot_window
    out = {}
    usable = foot_holds(holds)
    for h in holds:
        best = 1.0
        for k in usable:
            if k["id"] == h["id"]:
                continue
            dy = k["y"] - h["y"]  # image y grows downward, so dy>0 means k is below h
            dx = abs(k["x"] - h["x"])
            v = max(0.0, (lo * H - dy) / (lo * H), (dy - hi * H) / (0.5 * H), (dx - F.foot_lateral * H) / (0.4 * H))
            best = min(best, float(np.clip(v, 0.0, 1.0)))
            if best == 0.0:
                break
        out[h["id"]] = best
    return out


def _empty_terms():
    return {k: 0.0 for k in TERM_KEYS}


def move_cost(h_from: dict, h_to: dict, h_other: dict, hand: str, climber: Climber,
              W: Weights, F: Feasibility, foot_costs: Optional[Dict[int, float]] = None,
              foot_deficit_after: Optional[float] = None, hang_deficit_before: float = 0.0):
    """Cost of moving `hand` from h_from to h_to while the other hand stays
    on h_other. Returns (total, terms, feasible, info). In four-limb mode
    pass the support deficit after / before the move instead of foot_costs."""
    A = climber.arm_span_px
    d_reach = dist(h_other, h_to)      # span the climber must hold after the move
    r = d_reach / A
    d_travel = dist(h_from, h_to)
    t = d_travel / A
    up_px = h_from["y"] - h_to["y"]     # positive = the hand goes up
    up_frac = up_px / A
    if d_travel > 1e-6:
        up_cos = up_px / d_travel       # 1 straight up, 0 sideways, -1 straight down
        dir_pen = (1.0 - up_cos) / 2.0 * t
    else:
        dir_pen = 0.0
    if hand == LEFT:
        lx, rx = h_to["x"], h_other["x"]
    else:
        lx, rx = h_other["x"], h_to["x"]
    cross = float(np.clip(((lx - rx) / A - 0.05) / 0.3, 0.0, 1.0))  # left hand right of right hand
    if foot_deficit_after is not None:
        foot = float(foot_deficit_after)
    else:
        foot = foot_costs.get(h_to["id"], 0.0) if foot_costs else 0.0
    g = grip_penalty(h_to.get("grip", 3))

    terms = _empty_terms()
    terms.update({
        "reach": W.w_reach * (r / W.r_ref) ** 2,
        "grip": W.w_grip * g,
        "move": W.w_move,
        "travel": W.w_travel * t,
        "direction": W.w_dir * dir_pen,
        "cross": W.w_cross * cross,
        "foot": W.w_foot * foot,
        "hang": W.w_hang * float(hang_deficit_before),
    })
    total = float(sum(terms.values()))
    reasons = []
    if r > F.max_reach_frac:
        reasons.append(f"span {r:.2f} > envelope {F.max_reach_frac:.2f}")
    if up_frac < -F.max_down_frac:
        reasons.append(f"drops {-up_frac:.2f} of arm span")
    if t > F.max_travel_frac:
        reasons.append(f"travel {t:.2f} > cap {F.max_travel_frac:.2f}")
    if not h_to.get("on_route", True):
        reasons.append("target hold is off-route")
    if h_to.get("role") == "foot":
        reasons.append("target hold is foot-only")
    info = {"reach_frac": r, "travel_frac": t, "up_frac": up_frac, "grip": int(h_to.get("grip", 3)),
            "cross": cross, "foot": foot, "reasons": reasons}
    return total, terms, (len(reasons) == 0), info


# --------------------------------------------------------------------------- feet geometry
def hands_mid(hid: dict, L, R):
    a, b = hid[L], hid[R]
    return (a["x"] + b["x"]) / 2.0, (a["y"] + b["y"]) / 2.0


def foot_in_window(h_foot: dict, mid, climber: Climber, F: Feasibility) -> bool:
    H = climber.body_height_proxy_px
    dy = h_foot["y"] - mid[1]
    dx = abs(h_foot["x"] - mid[0])
    return (F.foot_window[0] * H <= dy <= F.foot_window[1] * H) and dx <= F.foot_lateral * H


def feet_ok(hid: dict, state: tuple, climber: Climber, F: Feasibility) -> bool:
    """Every placed foot inside the leg window of the current hands; feet not too spread."""
    if len(state) < 4:
        return True
    L, R, fL, fR = state
    mid = hands_mid(hid, L, R)
    for f in (fL, fR):
        if f is not None and (f not in hid or not foot_in_window(hid[f], mid, climber, F)):
            return False
    if fL is not None and fR is not None and fL != fR:
        if dist(hid[fL], hid[fR]) > F.feet_spread_frac * climber.leg_px:
            return False
    return True


def support_deficit(state: tuple, ground_ok: bool = False) -> float:
    """0 = both feet placed, 0.5 = one foot, 1 = cut loose. Hands-only states -> 0.
    ground_ok: the feet can reach the floor from this hand position, so an
    unplaced foot is standing, not hanging (the model cannot see the mat)."""
    if len(state) < 4 or ground_ok:
        return 0.0
    return (float(state[2] is None) + float(state[3] is None)) / 2.0


def ground_y_default(holds: list, climber: Climber, F: Feasibility) -> float:
    ys = [h["y"] for h in holds if h.get("on_route", True)] or [h["y"] for h in holds]
    return (max(ys) if ys else 0.0) + F.ground_margin * climber.body_height_proxy_px


def ground_reachable(hid: dict, state: tuple, climber: Climber, F: Feasibility, ground_y: float) -> bool:
    """Can a foot rest on the floor from this hand position? (hands' midpoint + max leg window reaches ground)"""
    _, my = hands_mid(hid, state[0], state[1])
    return my + F.foot_window[1] * climber.body_height_proxy_px >= ground_y


def foot_move_cost(h_from: Optional[dict], h_to: Optional[dict], foot: str, state_after: tuple, hid: dict,
                   climber: Climber, W: Weights):
    """Cost of moving one foot (h_to=None: cut that foot loose)."""
    terms = _empty_terms()
    terms["fmove"] = W.w_fmove
    leg = climber.leg_px
    if h_to is not None:
        if h_from is not None:
            travel = dist(h_from, h_to)
        else:  # foot was loose: travel from the hip proxy (below the hands' midpoint)
            mx, my = hands_mid(hid, state_after[0], state_after[1])
            travel = float(np.hypot(h_to["x"] - mx, h_to["y"] - (my + 0.85 * climber.body_height_proxy_px)))
        terms["ftravel"] = W.w_ftravel * travel / leg
        terms["grip"] = 0.5 * W.w_grip * grip_penalty(h_to.get("grip", 3))
        fL, fR = state_after[2], state_after[3]
        if fL is not None and fR is not None and fL != fR:
            crossed = (hid[fL]["x"] - hid[fR]["x"]) / leg  # left foot right of right foot
            terms["fcross"] = W.w_fcross * float(np.clip((crossed - 0.05) / 0.3, 0.0, 1.0))
        elif fL is not None and fL == fR:
            terms["fcross"] = 0.5 * W.w_fcross   # both feet on one hold: allowed, mildly discouraged
    total = float(sum(terms.values()))
    info = {"reach_frac": 0.0, "travel_frac": 0.0, "up_frac": 0.0,
            "grip": int(h_to.get("grip", 3)) if h_to else 0, "cross": 0.0, "foot": 0.0, "reasons": []}
    return total, terms, True, info


# --------------------------------------------------------------------------- graph helpers
def hold_graph_edges(holds: list, climber: Climber, F: Feasibility):
    """Undirected hold-level feasibility graph: two holds are connected iff
    this climber can hold both at once (span <= reach envelope). This is
    what the UI draws; the search itself runs over limb states."""
    route = hand_holds(holds)
    edges = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            r = dist(route[i], route[j]) / climber.arm_span_px
            if r <= F.max_reach_frac:
                edges.append((route[i]["id"], route[j]["id"], r))
    return edges


class _FootCandidates:
    """Foot holds inside the leg window for a given pair of hand holds (memoized)."""

    def __init__(self, holds, hid, climber, F):
        self.hid = hid
        self.climber = climber
        self.F = F
        self.usable = foot_holds(holds)
        self.xs = np.array([h["x"] for h in self.usable], float)
        self.ys = np.array([h["y"] for h in self.usable], float)
        self.ids = [h["id"] for h in self.usable]
        self.cache = {}

    def __call__(self, L, R):
        key = (L, R) if L <= R else (R, L)
        if key in self.cache:
            return self.cache[key]
        if len(self.ids) == 0:
            self.cache[key] = []
            return []
        mx, my = hands_mid(self.hid, L, R)
        H = self.climber.body_height_proxy_px
        lo, hi = self.F.foot_window
        dy = self.ys - my
        m = (dy >= lo * H) & (dy <= hi * H) & (np.abs(self.xs - mx) <= self.F.foot_lateral * H)
        out = [self.ids[i] for i in np.nonzero(m)[0]]
        self.cache[key] = out
        return out


def estimate_states(holds: list, climber: Climber, F: Feasibility, limbs: str = "hands", sample: int = 200) -> dict:
    """Cheap size estimate of the state space (used to pick exact vs budgeted search)."""
    route = hand_holds(holds)
    edges = hold_graph_edges(holds, climber, F)
    hand_states = 2 * len(edges) + len(route)
    if limbs == "hands":
        return {"hand_states": hand_states, "foot_pair_options": 1.0, "joint_states": hand_states}
    hid = {h["id"]: h for h in holds}
    fc = _FootCandidates(holds, hid, climber, F)
    counts = [len(fc(i, j)) for i, j, _ in edges[:sample]] or [0]
    nf = float(np.mean(counts))
    options = (nf + 1.0) ** 2   # each foot: any candidate or None
    return {"hand_states": hand_states, "foot_pair_options": options, "joint_states": hand_states * options,
            "mean_foot_candidates": nf}


# --------------------------------------------------------------------------- search
def _is_goal(state, finish_ids, match_finish):
    L, R = state[0], state[1]
    if match_finish:
        return L in finish_ids and R in finish_ids
    return L in finish_ids or R in finish_ids


def _astar(start, is_goal, expand, heuristic, max_expansions: Optional[int] = None):
    """A* with an admissible heuristic (heuristic=0 -> plain Dijkstra). Returns
    (moves, cost, stats); moves is None if no path (stats['exhausted'] says why)."""
    best = {start: 0.0}
    prev = {}
    counter = 0
    pq = [(heuristic(start), 0.0, counter, start)]
    n_expanded = n_pushed = 0
    while pq:
        f, g, _, s = heapq.heappop(pq)
        if g > best.get(s, float("inf")):
            continue
        if is_goal(s):
            moves = []
            cur = s
            while cur in prev:
                p, mv = prev[cur]
                moves.append(mv)
                cur = p
            moves.reverse()
            return moves, g, {"n_expanded": n_expanded, "n_pushed": n_pushed, "exhausted": False}
        n_expanded += 1
        if max_expansions is not None and n_expanded > max_expansions:
            return None, None, {"n_expanded": n_expanded, "n_pushed": n_pushed, "exhausted": True}
        for ns, c, mv in expand(s):
            ng = g + c
            if ng < best.get(ns, float("inf")):
                best[ns] = ng
                prev[ns] = (s, mv)
                counter += 1
                n_pushed += 1
                heapq.heappush(pq, (ng + heuristic(ns), ng, counter, ns))
    return None, None, {"n_expanded": n_expanded, "n_pushed": n_pushed, "exhausted": False}


def _beam(start, is_goal, expand, heuristic, beam: int = 200, max_depth: int = 40, per_hand_pair: int = 4):
    """Level-wise beam search over the same expand(); approximate.
    - every successor is goal-tested BEFORE pruning, so a reached finish is never lost;
    - at most `per_hand_pair` foot configurations survive per hand pair, so cheap
      foot shuffles cannot crowd out hand progress;
    - ranking uses g + heuristic (pass the *typical-cost* heuristic, not the
      admissible bound: an underestimate makes far-from-goal states look best).
    Returns (moves, cost, stats)."""
    frontier = {start: (0.0, None)}   # state -> (g, (prev_state, move))
    history = [frontier]
    best_goal = None                  # (g, level, state)
    n_expanded = n_pushed = 0
    if is_goal(start):
        return [], 0.0, {"n_expanded": 0, "n_pushed": 0, "exhausted": False}
    for depth in range(max_depth):
        nxt = {}
        for s, (g, _) in frontier.items():
            n_expanded += 1
            for ns, c, mv in expand(s):
                ng = g + c
                if ns not in nxt or ng < nxt[ns][0]:
                    nxt[ns] = (ng, (s, mv))
                    n_pushed += 1
        if not nxt:
            break
        level = depth + 1
        goals = {st: val for st, val in nxt.items() if is_goal(st)}
        for st, (g, _) in goals.items():
            if best_goal is None or g < best_goal[0]:
                best_goal = (g, level, st)
        ranked = sorted(((st, val) for st, val in nxt.items() if st not in goals),
                        key=lambda kv: kv[1][0] + heuristic(kv[0]))
        keep, per_pair = [], {}
        for st, val in ranked:
            hp = (st[0], st[1])
            if per_pair.get(hp, 0) >= per_hand_pair:
                continue
            per_pair[hp] = per_pair.get(hp, 0) + 1
            keep.append((st, val))
            if len(keep) >= beam:
                break
        frontier = dict(keep)           # goals are not expanded further
        history.append({**frontier, **goals})   # ...but stay reconstructable at this level
        if best_goal is not None and all(g >= best_goal[0] for st, (g, _) in keep):
            break
        if not keep:
            break
    if best_goal is None:
        return None, None, {"n_expanded": n_expanded, "n_pushed": n_pushed, "exhausted": True}
    g, level, s = best_goal
    moves = []
    cur = s
    for lv in range(level, 0, -1):
        _, (p, mv) = history[lv][cur]
        moves.append(mv)
        cur = p
    moves.reverse()
    return moves, g, {"n_expanded": n_expanded, "n_pushed": n_pushed, "exhausted": False}


def _make_expand(hid, route, climber, W, F, foot_costs, limbs, foot_cands, ground_y: Optional[float] = None):
    """Successor generator: moves one limb at a time."""
    fourlimb = limbs == "all"
    if fourlimb and ground_y is None:
        ground_y = ground_y_default(list(hid.values()), climber, F)

    def deficit(state):
        return support_deficit(state, ground_reachable(hid, state, climber, F, ground_y))

    def expand(s):
        L, R = s[0], s[1]
        deficit_before = deficit(s) if fourlimb else 0.0
        # ---- hand moves
        for hand in HANDS:
            frm, other = (L, R) if hand == LEFT else (R, L)
            for b in route:
                if b["id"] == frm:
                    continue
                ns = (b["id"], R) + tuple(s[2:]) if hand == LEFT else (L, b["id"]) + tuple(s[2:])
                if fourlimb:
                    if not feet_ok(hid, ns, climber, F):
                        continue
                    total, terms, feasible, info = move_cost(hid[frm], b, hid[other], hand, climber, W, F,
                                                             foot_deficit_after=deficit(ns),
                                                             hang_deficit_before=deficit_before)
                else:
                    total, terms, feasible, info = move_cost(hid[frm], b, hid[other], hand, climber, W, F, foot_costs)
                if not feasible:
                    continue
                yield ns, total, {"limb": hand, "hand": hand, "from": frm, "to": b["id"], "other": other,
                                  "cost": total, "terms": terms, "feasible": True, **info}
        if not fourlimb:
            return
        # ---- foot moves (including cutting loose)
        cands = foot_cands(L, R)
        for foot in FEET:
            idx = LIMB_INDEX[foot]
            cur = s[idx]
            for c in list(cands) + [None]:
                if c == cur:
                    continue
                ns = list(s)
                ns[idx] = c
                ns = tuple(ns)
                if not feet_ok(hid, ns, climber, F):
                    continue
                total, terms, feasible, info = foot_move_cost(hid.get(cur) if cur is not None else None,
                                                              hid.get(c) if c is not None else None,
                                                              foot, ns, hid, climber, W)
                yield ns, total, {"limb": foot, "hand": None, "from": cur, "to": c, "other": s[5 - idx],
                                  "cost": total, "terms": terms, "feasible": True, **info}
    return expand


def _make_heuristic(hid, finish_ids, climber, W, F, typical: bool = False):
    """Admissible lower bound on the remaining cost (A*): the nearest hand still
    needs >= ceil(d / max_travel) moves at >= w_move + w_travel*t each.
    typical=True returns a realistic estimate instead (per 0.4-arm-span move:
    w_move + w_reach + w_grip/2 + 0.4*w_travel) for beam ranking only."""
    fin = [hid[f] for f in finish_ids if f in hid]
    A = climber.arm_span_px
    step = F.max_travel_frac * A
    c_typ = W.w_move + W.w_reach + 0.5 * W.w_grip + 0.4 * W.w_travel

    def h(s):
        if not fin:
            return 0.0
        d = min(dist(hid[x], f) for x in (s[0], s[1]) for f in fin)
        if d <= 1e-6:
            return 0.0
        if typical:
            return c_typ * d / (0.4 * A)
        return W.w_move * math.ceil(d / step) + W.w_travel * d / A
    return h


def summarize(moves: list, start_state, label: str, climber: Climber, hid: dict) -> dict:
    start = tuple(start_state)
    states = [start]
    for m in moves:
        ns = list(states[-1])
        limb = m.get("limb") or m.get("hand")
        ns[LIMB_INDEX[limb]] = m["to"]
        states.append(tuple(ns))
    holds_used = []
    for s in states:
        for x in s[:2]:
            if x not in holds_used:
                holds_used.append(x)
    feet_used = []
    for s in states:
        for x in s[2:]:
            if x is not None and x not in feet_used:
                feet_used.append(x)
    costs = [m["cost"] for m in moves]
    crux = int(np.argmax(costs)) if costs else None
    term_totals = {}
    for m in moves:
        for k, v in m["terms"].items():
            term_totals[k] = term_totals.get(k, 0.0) + v
    hand_moves = [m for m in moves if (m.get("limb") or m.get("hand")) in HANDS]
    return {
        "label": label,
        "climber": asdict(climber),
        "limbs": "all" if len(start) == 4 else "hands",
        "start_state": list(start),
        "states": [list(s) for s in states],
        "moves": moves,
        "total_cost": float(sum(costs)),
        "term_totals": term_totals,
        "n_moves": len(moves),
        "n_hand_moves": len(hand_moves),
        "n_foot_moves": len(moves) - len(hand_moves),
        "max_reach_frac": float(max((m["reach_frac"] for m in hand_moves), default=0.0)),
        "mean_reach_frac": float(np.mean([m["reach_frac"] for m in hand_moves])) if hand_moves else 0.0,
        "mean_grip": float(np.mean([hid[m["to"]].get("grip", 3) for m in hand_moves])) if hand_moves else None,
        "holds_used": holds_used,
        "feet_used": feet_used,
        "feet_by_state": [list(s[2:]) if len(s) == 4 else None for s in states],
        "crux_index": crux,
        "crux_cost": float(costs[crux]) if crux is not None else None,
        "n_infeasible": int(sum(1 for m in moves if not m["feasible"])),
    }


def optimize_beta(holds: list, climber: Climber, W: Weights, F: Feasibility,
                  start_state: Tuple, finish_ids, label="Optimized beta",
                  relax: bool = True, limbs: str = "hands", max_expansions: int = 150_000,
                  beam: int = 200, exact_threshold: int = 60_000, ground_y: Optional[float] = None) -> Optional[dict]:
    """Minimum-cost limb sequence. limbs='hands' (state = hand pair) or 'all'
    (hands + feet). Exact A* when the state-space estimate is below
    exact_threshold; otherwise a budgeted A* that falls back to beam search
    (result['search']['exact'] is False). If the graph is disconnected under
    the envelope, it is loosened in 0.05 steps (result['relaxed_to'])."""
    route = hand_holds(holds)
    hid = {h["id"]: h for h in holds}
    finish_ids = set(finish_ids)
    start = tuple(start_state)
    if limbs == "all" and len(start) == 2:
        start = start + (None, None)
    if limbs == "hands" and len(start) == 4:
        start = start[:2]
    if not route or not finish_ids or any(s not in hid for s in start[:2]):
        return None
    foot_costs = foot_support_costs(holds, climber, F) if limbs == "hands" else None
    F_used = replace(F)
    t0 = time.time()
    for _ in range(12):
        est = estimate_states(holds, climber, F_used, limbs)
        foot_cands = _FootCandidates(holds, hid, climber, F_used) if limbs == "all" else None
        if limbs == "all" and not feet_ok(hid, start, climber, F_used):
            start = start[:2] + (None, None)   # observed feet outside the model window: start cut loose
        expand = _make_expand(hid, route, climber, W, F_used, foot_costs, limbs, foot_cands, ground_y)
        heur = _make_heuristic(hid, finish_ids, climber, W, F_used)
        heur_typ = _make_heuristic(hid, finish_ids, climber, W, F_used, typical=True)
        goal = lambda s: _is_goal(s, finish_ids, F_used.match_finish)  # noqa: E731
        exact_ok = est["joint_states"] <= exact_threshold
        moves, cost, stats = _astar(start, goal, expand, heur, None if exact_ok else max_expansions)
        method, exact = "astar", True
        beam_used = beam
        if moves is None and stats.get("exhausted"):
            astar_exp = stats["n_expanded"]
            for beam_used in (beam, beam * 4):
                moves, cost, stats2 = _beam(start, goal, expand, heur_typ, beam=beam_used)
                if moves is not None:
                    break
            stats = {**stats2, "astar_expanded": astar_exp}
            method, exact = "beam", False
        if moves is not None:
            res = summarize(moves, start, label, climber, hid)
            res["feasibility"] = asdict(F_used)
            res["relaxed_to"] = None if F_used.max_reach_frac == F.max_reach_frac else F_used.max_reach_frac
            res["search"] = {"method": method, "exact": exact, "n_expanded": stats["n_expanded"],
                             "n_pushed": stats["n_pushed"], "est_states": float(est["joint_states"]),
                             "hand_states": est["hand_states"], "runtime_s": time.time() - t0,
                             "beam": beam_used if method == "beam" else None, "limbs": limbs}
            return res
        if not relax:
            break
        F_used = replace(F_used, max_reach_frac=F_used.max_reach_frac + 0.05,
                         max_down_frac=F_used.max_down_frac + 0.05)
    return None


def score_sequence(holds: list, climber: Climber, W: Weights, F: Feasibility,
                   start_state: Tuple, placements: list, label="Observed beta",
                   ground_y: Optional[float] = None) -> dict:
    """Score an actual (observed or user-edited) placement sequence with the
    SAME objective. Placements: {hand|limb, hold_id[, frame]}; foot placements
    (limb LEFT_FOOT/RIGHT_FOOT, hold_id may be None = cut loose) switch the
    scoring to four-limb mode. Infeasible-under-model moves are kept and flagged."""
    hid = {h["id"]: h for h in holds}
    fourlimb = len(start_state) == 4 or any((p.get("limb") or p.get("hand")) in FEET for p in placements)
    state = list(start_state) + ([None, None] if fourlimb and len(start_state) == 2 else [])
    foot_costs = foot_support_costs(holds, climber, F) if not fourlimb else None
    gy = ground_y if ground_y is not None else ground_y_default(holds, climber, F)

    def deficit(st):
        return support_deficit(tuple(st), ground_reachable(hid, tuple(st), climber, F, gy))
    moves = []
    for p in placements:
        limb = p.get("limb") or p.get("hand")
        idx = LIMB_INDEX[limb]
        target = p["hold_id"]
        if target is not None and target not in hid:
            continue
        frm = state[idx]
        if frm == target:
            continue
        if limb in HANDS:
            other = state[1 - idx]
            ns = list(state)
            ns[idx] = target
            if fourlimb:
                total, terms, feasible, info = move_cost(hid[frm], hid[target], hid[other], limb, climber, W, F,
                                                         foot_deficit_after=deficit(ns),
                                                         hang_deficit_before=deficit(state))
                if not feet_ok(hid, tuple(ns), climber, F):
                    feasible = False
                    info["reasons"] = info["reasons"] + ["feet outside the leg window after the move"]
            else:
                total, terms, feasible, info = move_cost(hid[frm], hid[target], hid[other], limb, climber, W, F, foot_costs)
            moves.append({"limb": limb, "hand": limb, "from": frm, "to": target, "other": other, "cost": total,
                          "terms": terms, "feasible": feasible, "frame": p.get("frame"), **info})
        else:
            ns = list(state)
            ns[idx] = target
            total, terms, feasible, info = foot_move_cost(hid.get(frm) if frm is not None else None,
                                                          hid.get(target) if target is not None else None,
                                                          limb, tuple(ns), hid, climber, W)
            if not feet_ok(hid, tuple(ns), climber, F):
                feasible = False
                info["reasons"] = ["foot outside the leg window"]
            moves.append({"limb": limb, "hand": None, "from": frm, "to": target, "other": state[5 - idx],
                          "cost": total, "terms": terms, "feasible": feasible, "frame": p.get("frame"), **info})
        state = ns
    return summarize(moves, tuple(start_state) + ((None, None) if fourlimb and len(start_state) == 2 else ()),
                     label, climber, hid)


def calibrate_envelope(F: Feasibility, observed: Optional[dict], margin=0.03, cap=1.0) -> Feasibility:
    """A move the climber actually performed is feasible for them by
    definition: widen the reach envelope to cover the observed maximum."""
    if not observed or not observed["moves"]:
        return F
    obs_max = observed["max_reach_frac"]
    if obs_max + margin > F.max_reach_frac:
        return replace(F, max_reach_frac=float(min(cap, obs_max + margin)))
    return F


def limb_name(limb: Optional[str]) -> str:
    return {LEFT: "left hand", RIGHT: "right hand", LEFT_FOOT: "left foot", RIGHT_FOOT: "right foot"}.get(limb, str(limb))


def compare(observed: Optional[dict], optimized: Optional[dict], hid: dict) -> dict:
    out = {}
    if observed and optimized:
        oc, pc = observed["total_cost"], optimized["total_cost"]
        out["improvement_frac"] = (oc - pc) / oc if oc > 0 else 0.0
        so, sp = set(observed["holds_used"]), set(optimized["holds_used"])
        out["shared_holds"] = sorted(so & sp)
        out["only_observed"] = sorted(so - sp)
        out["only_optimized"] = sorted(sp - so)
        out["same_sequence"] = observed["states"] == optimized["states"]
        out["same_holds"] = so == sp
    if observed and observed["crux_index"] is not None:
        m = observed["moves"][observed["crux_index"]]
        top = sorted(m["terms"].items(), key=lambda kv: -kv[1])
        out["crux"] = {
            "move_index": observed["crux_index"], "hand": m.get("hand"), "limb": m.get("limb") or m.get("hand"),
            "from": m["from"], "to": m["to"],
            "cost": m["cost"], "reach_frac": m["reach_frac"], "grip": m["grip"],
            "drivers": [(k, v) for k, v in top if v > 1e-9][:3],
            "explanation": crux_explanation(m, hid),
        }
    return out


def crux_explanation(m: dict, hid: dict) -> str:
    bits = []
    t = m["terms"]
    top = max(t.values()) if t else 0.0
    if t.get("reach", 0) >= top * 0.5 and m.get("reach_frac", 0) > 0:
        bits.append(f"a long normalized reach ({m['reach_frac']:.0%} of arm span)")
    if m["grip"] >= 4:
        bits.append(f"a poorly rated target hold (grip {m['grip']}/5)")
    if t.get("foot", 0) > 0.1:
        bits.append("little foot support below the target")
    if t.get("hang", 0) > 0.1:
        bits.append("moving a hand with the feet cut loose")
    if t.get("cross", 0) > 0.1:
        bits.append("crossed hands")
    if t.get("direction", 0) > 0.15:
        bits.append("a sideways/downward move")
    if t.get("ftravel", 0) > 0.2:
        bits.append("a long foot move")
    limb = limb_name(m.get("limb") or m.get("hand"))
    src = hid.get(m["from"], {}).get("label", m["from"] if m["from"] is not None else "loose")
    dst = hid.get(m["to"], {}).get("label", m["to"] if m["to"] is not None else "loose")
    what = ", ".join(bits) if bits else "an accumulation of moderate costs"
    return f"Moving the {limb} {src} -> {dst} is the highest-cost move under our model, driven by {what}."
