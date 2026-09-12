"""Personalized beta optimization: hand-pair state graph + Dijkstra.

State  s = (hold under LEFT hand, hold under RIGHT hand)
Action = move ONE hand to another on-route hold
Edge   feasible iff the hand-to-hand span after the move fits inside the
       climber's reach envelope (a fraction of THEIR measured arm span),
       the move doesn't drop too far, and the target is on the route.
Cost   C(move | climber) = w_r * ReachCost + w_g * GripCost + w_m
                         + w_t * TravelCost + w_d * DirectionCost
                         + w_x * CrossCost + w_f * FootSupportCost
       -- every term is dimensionless and explainable (see move_cost).
Search exact shortest path (Dijkstra) from the start state to any state
       with a hand on the finish hold. Same cost function scores the
       climber's observed sequence, so the comparison is apples to apples.

Scope (deliberate): this optimizes the major HAND sequence; feet enter
only through the foot-support proxy. Explicit four-limb state search is
the natural extension.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, replace, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

LEFT, RIGHT = "LEFT", "RIGHT"


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


@dataclass
class Weights:
    w_reach: float = 1.0     # (normalized reach / r_ref)^2 : longer reaches cost super-linearly more
    r_ref: float = 0.4       # normalized reach (hand-to-hand span / arm span) that costs exactly w_reach
    w_grip: float = 0.8      # (grip rating - 1) / 4 of the target hold: 1 -> 0, 5 -> 1
    w_move: float = 0.35     # fixed cost of every hand movement (re-grip / instability moment)
    w_travel: float = 0.25   # moving-hand travel distance / arm span
    w_dir: float = 0.4       # sideways or downward travel (0 for a straight-up move)
    w_cross: float = 0.5     # hands crossed after the move
    w_foot: float = 0.4      # no foothold in the climber's leg window below the target

    @classmethod
    def geometric(cls):
        """Pure shortest-geometric-path objective (for the 'why not just
        the shortest path?' comparison)."""
        return cls(w_reach=0, w_grip=0, w_move=0, w_travel=1.0, w_dir=0, w_cross=0, w_foot=0)


@dataclass
class Feasibility:
    max_reach_frac: float = 0.85   # hand-to-hand span after the move, as a fraction of arm span
    max_down_frac: float = 0.20    # a hand may drop at most this fraction of arm span
    max_travel_frac: float = 1.30  # moving hand travel cap (fraction of arm span)
    match_finish: bool = False     # require both hands on the finish hold


def dist(a, b) -> float:
    return float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))


def grip_penalty(grip: int) -> float:
    return (float(np.clip(grip, 1, 5)) - 1.0) / 4.0


def foot_support_costs(holds: list, climber: Climber) -> Dict[int, float]:
    """Lower-body context proxy: for each hold, is there ANY hold inside
    the window where this climber's feet could be while a hand is on it?
    Window (below the hand hold): 0.35..1.25 x body-height proxy vertically,
    +-0.6 x body-height proxy laterally. 0 = supported, 1 = nothing usable."""
    H = climber.body_height_proxy_px
    out = {}
    usable = [k for k in holds if k.get("on_route", True)]
    for h in holds:
        best = 1.0
        for k in usable:
            if k["id"] == h["id"]:
                continue
            dy = k["y"] - h["y"]  # image y grows downward, so dy>0 means k is below h
            dx = abs(k["x"] - h["x"])
            v = max(0.0, (0.35 * H - dy) / (0.35 * H), (dy - 1.25 * H) / (0.5 * H), (dx - 0.6 * H) / (0.4 * H))
            best = min(best, float(np.clip(v, 0.0, 1.0)))
            if best == 0.0:
                break
        out[h["id"]] = best
    return out


def move_cost(h_from: dict, h_to: dict, h_other: dict, hand: str, climber: Climber,
              W: Weights, F: Feasibility, foot_costs: Optional[Dict[int, float]] = None):
    """Cost of moving `hand` from h_from to h_to while the other hand stays
    on h_other. Returns (total, terms, feasible, info)."""
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
    foot = foot_costs.get(h_to["id"], 0.0) if foot_costs else 0.0
    g = grip_penalty(h_to.get("grip", 3))

    terms = {
        "reach": W.w_reach * (r / W.r_ref) ** 2,
        "grip": W.w_grip * g,
        "move": W.w_move,
        "travel": W.w_travel * t,
        "direction": W.w_dir * dir_pen,
        "cross": W.w_cross * cross,
        "foot": W.w_foot * foot,
    }
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


def hand_holds(holds: list) -> list:
    """Holds a hand may use: on-route and not marked foot-only."""
    return [h for h in holds if h.get("on_route", True) and h.get("role") != "foot"]


def hold_graph_edges(holds: list, climber: Climber, F: Feasibility):
    """Undirected hold-level feasibility graph: two holds are connected iff
    this climber can hold both at once (span <= reach envelope). This is
    what the UI draws; the search itself runs over hand-pair states."""
    route = hand_holds(holds)
    edges = []
    for i in range(len(route)):
        for j in range(i + 1, len(route)):
            r = dist(route[i], route[j]) / climber.arm_span_px
            if r <= F.max_reach_frac:
                edges.append((route[i]["id"], route[j]["id"], r))
    return edges


def _is_goal(state, finish_ids, match_finish):
    L, R = state
    if match_finish:
        return L in finish_ids and R in finish_ids
    return L in finish_ids or R in finish_ids


def _dijkstra(hid: dict, route: list, climber, W, F, foot_costs, start_state, finish_ids):
    start = tuple(start_state)
    best = {start: 0.0}
    prev = {}
    pq = [(0.0, 0, start)]
    counter = 1
    while pq:
        c, _, s = heapq.heappop(pq)
        if c > best.get(s, float("inf")):
            continue
        if _is_goal(s, finish_ids, F.match_finish):
            moves = []
            cur = s
            while cur in prev:
                p, mv = prev[cur]
                moves.append(mv)
                cur = p
            moves.reverse()
            return moves, c
        L, R = s
        for hand in (LEFT, RIGHT):
            frm, other = (L, R) if hand == LEFT else (R, L)
            for b in route:
                if b["id"] == frm:
                    continue
                total, terms, feasible, info = move_cost(hid[frm], b, hid[other], hand, climber, W, F, foot_costs)
                if not feasible:
                    continue
                ns = (b["id"], R) if hand == LEFT else (L, b["id"])
                nc = c + total
                if nc < best.get(ns, float("inf")):
                    best[ns] = nc
                    prev[ns] = (s, {"hand": hand, "from": frm, "to": b["id"], "other": other,
                                    "cost": total, "terms": terms, "feasible": True, **info})
                    counter += 1
                    heapq.heappush(pq, (nc, counter, ns))
    return None, None


def summarize(moves: list, start_state, label: str, climber: Climber, hid: dict) -> dict:
    states = [tuple(start_state)]
    for m in moves:
        L, R = states[-1]
        states.append((m["to"], R) if m["hand"] == LEFT else (L, m["to"]))
    holds_used = []
    for s in states:
        for x in s:
            if x not in holds_used:
                holds_used.append(x)
    costs = [m["cost"] for m in moves]
    crux = int(np.argmax(costs)) if costs else None
    term_totals = {}
    for m in moves:
        for k, v in m["terms"].items():
            term_totals[k] = term_totals.get(k, 0.0) + v
    return {
        "label": label,
        "climber": asdict(climber),
        "start_state": list(start_state),
        "states": [list(s) for s in states],
        "moves": moves,
        "total_cost": float(sum(costs)),
        "term_totals": term_totals,
        "n_moves": len(moves),
        "max_reach_frac": float(max((m["reach_frac"] for m in moves), default=0.0)),
        "mean_reach_frac": float(np.mean([m["reach_frac"] for m in moves])) if moves else 0.0,
        "mean_grip": float(np.mean([hid[m["to"]].get("grip", 3) for m in moves])) if moves else None,
        "holds_used": holds_used,
        "crux_index": crux,
        "crux_cost": float(costs[crux]) if crux is not None else None,
        "n_infeasible": int(sum(1 for m in moves if not m["feasible"])),
    }


def optimize_beta(holds: list, climber: Climber, W: Weights, F: Feasibility,
                  start_state: Tuple[int, int], finish_ids, label="Optimized beta",
                  relax: bool = True) -> Optional[dict]:
    """Exact minimum-cost hand sequence. If the graph is disconnected under
    the current envelope, the envelope is loosened in 0.05 steps (reported
    in result['relaxed_to']) so the user always gets an answer + a warning."""
    route = hand_holds(holds)
    hid = {h["id"]: h for h in holds}
    finish_ids = set(finish_ids)
    if not route or not finish_ids or any(s not in hid for s in start_state):
        return None
    foot_costs = foot_support_costs(holds, climber)
    F_used = replace(F)
    for _ in range(12):
        moves, cost = _dijkstra(hid, route, climber, W, F_used, foot_costs, start_state, finish_ids)
        if moves is not None:
            res = summarize(moves, start_state, label, climber, hid)
            res["feasibility"] = asdict(F_used)
            res["relaxed_to"] = None if F_used.max_reach_frac == F.max_reach_frac else F_used.max_reach_frac
            return res
        if not relax:
            break
        F_used = replace(F_used, max_reach_frac=F_used.max_reach_frac + 0.05,
                         max_down_frac=F_used.max_down_frac + 0.05)
    return None


def score_sequence(holds: list, climber: Climber, W: Weights, F: Feasibility,
                   start_state: Tuple[int, int], placements: list, label="Observed beta") -> dict:
    """Score an actual (observed or user-edited) placement sequence with
    the SAME objective. Infeasible-under-model moves are kept and flagged."""
    hid = {h["id"]: h for h in holds}
    foot_costs = foot_support_costs(holds, climber)
    state = list(start_state)
    moves = []
    for p in placements:
        if p["hold_id"] not in hid:
            continue
        hand = p["hand"]
        i_move, i_other = (0, 1) if hand == LEFT else (1, 0)
        frm, other = state[i_move], state[i_other]
        if frm == p["hold_id"]:
            continue
        total, terms, feasible, info = move_cost(hid[frm], hid[p["hold_id"]], hid[other], hand, climber, W, F, foot_costs)
        moves.append({"hand": hand, "from": frm, "to": p["hold_id"], "other": other, "cost": total,
                      "terms": terms, "feasible": feasible, "frame": p.get("frame"), **info})
        state[i_move] = p["hold_id"]
    return summarize(moves, start_state, label, climber, hid)


def calibrate_envelope(F: Feasibility, observed: Optional[dict], margin=0.03, cap=1.0) -> Feasibility:
    """A move the climber actually performed is feasible for them by
    definition: widen the reach envelope to cover the observed maximum."""
    if not observed or not observed["moves"]:
        return F
    obs_max = observed["max_reach_frac"]
    if obs_max + margin > F.max_reach_frac:
        return replace(F, max_reach_frac=float(min(cap, obs_max + margin)))
    return F


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
            "move_index": observed["crux_index"], "hand": m["hand"], "from": m["from"], "to": m["to"],
            "cost": m["cost"], "reach_frac": m["reach_frac"], "grip": m["grip"],
            "drivers": [(k, v) for k, v in top if v > 1e-9][:3],
            "explanation": crux_explanation(m, hid),
        }
    return out


def crux_explanation(m: dict, hid: dict) -> str:
    bits = []
    t = m["terms"]
    if t.get("reach", 0) >= max(t.values()) * 0.5:
        bits.append(f"a long normalized reach ({m['reach_frac']:.0%} of arm span)")
    if m["grip"] >= 4:
        bits.append(f"a poorly rated target hold (grip {m['grip']}/5)")
    if t.get("foot", 0) > 0.1:
        bits.append("little foot support below the target")
    if t.get("cross", 0) > 0.1:
        bits.append("crossed hands")
    if t.get("direction", 0) > 0.15:
        bits.append("a sideways/downward move")
    hand = "left" if m["hand"] == LEFT else "right"
    src = hid.get(m["from"], {}).get("label", m["from"])
    dst = hid.get(m["to"], {}).get("label", m["to"])
    what = ", ".join(bits) if bits else "an accumulation of moderate costs"
    return f"Moving the {hand} hand {src} -> {dst} is the highest-cost move under our model, driven by {what}."
