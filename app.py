"""Send It -- Kilter board move coach (Streamlit UI).

Run:  streamlit run app.py

Layout, one container per section so a design mockup maps one-to-one onto code:
  SIDEBAR        route source, units, Deeper insight button
  HEADER         title + status line, Deeper-insight panel (visible on every step)
  STEP 1 ROUTE   route image -> lit holds -> roles -> board angle
  STEP 2 CLIMB   climb video -> alignment -> results -> tips, flags, extras
  STEP 3 EXPLORE body-size simulation, feet plan
Every user-facing string lives in sendit/ui_text.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import astuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from sendit import viz, kilter, register, injury, coach
from sendit import beta as beta_mod
from sendit import ui_text as T
from sendit import units as U
from sendit import stabilize as stab_mod
from sendit.holds import GRIP_LABELS, make_hold, next_id
from sendit.optimizer import Weights, Feasibility, LEFT, RIGHT, HANDS
from sendit.pipeline import (analyze_video, apply_route, load_analysis, load_analysis_pose, recompute_observed,
                             recompute_feet, run_optimization, feet_for_result, fourlimb_cache_key,
                             holds_for_video_frame)
from sendit.pose import load_pose

ROOT = os.path.dirname(os.path.abspath(__file__))
coach.load_env(os.path.join(ROOT, ".env"))
DEMOS = json.load(open(os.path.join(ROOT, "demo_assets", "demos.json")))["demos"]
DEMO = DEMOS[0]

st.set_page_config(page_title=T.PAGE_TITLE, page_icon="🧗", layout="wide")

# Existing CSS only (metric cards + line colours). No new styling: the visual pass happens in Claude Design.
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {background: linear-gradient(135deg,#16181d,#22262e); border:1px solid #2f3440; border-radius:14px; padding:14px 16px;}
.metric-card .lab {font-size:0.78rem; color:#9aa3b2; text-transform:uppercase; letter-spacing:.08em;}
.metric-card .val {font-size:1.7rem; font-weight:700; color:#f5f7fa; line-height:1.15;}
.metric-card .sub {font-size:0.8rem; color:#b7c0cf;}
.obs {color:#ff8c00;} .opt {color:#00c8ff;} .crux {color:#ff4040;}
.hero-title {font-size:2.0rem; font-weight:800; margin-bottom:0;}
</style>
""", unsafe_allow_html=True)

WEIGHT_DEFAULTS = {"w_reach": 1.0, "w_grip": 0.8, "w_move": 0.35, "w_travel": 0.25, "w_dir": 0.4, "w_cross": 0.5, "w_foot": 0.4}
STYLE_PRESETS = {
    T.STYLE_BALANCED: dict(WEIGHT_DEFAULTS),
    T.STYLE_FEWER: {**WEIGHT_DEFAULTS, "w_move": 1.0},
    T.STYLE_SHORTER: {**WEIGHT_DEFAULTS, "w_reach": 2.0, "w_move": 0.2},
}
CANVAS_W = 600
COMPARISON_MAX_H = 640


# ============================================================================ helpers
def card(label, value, sub="", cls=""):
    st.markdown(f'<div class="metric-card"><div class="lab">{label}</div>'
                f'<div class="val {cls}">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def hid_map():
    return {h["id"]: h for h in st.session_state.holds}


def label_of(hid, i):
    return hid.get(i, {}).get("label", f"H{i}")


def limb_tag(m):
    limb = m.get("limb") or m.get("hand")
    return {LEFT: "L", RIGHT: "R", "LEFT_FOOT": "Lf", "RIGHT_FOOT": "Rf"}.get(limb, "?")


def sequence_text(res, hid):
    if not res:
        return "—"
    L, R = res["start_state"][0], res["start_state"][1]
    parts = [f"start L:{label_of(hid, L)} R:{label_of(hid, R)}"]
    for m in res["moves"]:
        tgt = label_of(hid, m["to"]) if m["to"] is not None else "loose"
        parts.append(f"{limb_tag(m)}→{tgt}")
    return "  ·  ".join(parts)


def route_dir_for(image_bytes: bytes) -> str:
    return os.path.join(ROOT, "demo_output", "routes", hashlib.md5(image_bytes).hexdigest()[:10])


def route_file(rdir):
    return os.path.join(rdir, "route.json")


def load_route_file(rdir):
    p = route_file(rdir)
    return json.load(open(p)) if os.path.exists(p) else None


def save_route_file(rdir, holds, angle, body, name):
    os.makedirs(rdir, exist_ok=True)
    with open(route_file(rdir), "w") as f:
        json.dump({"name": name, "holds": holds, "angle": angle, "body": body}, f, indent=1)
    with open(os.path.join(rdir, "holds_edited.json"), "w") as f:   # kept for scripts/build_demo_cache.py
        json.dump(holds, f, indent=1)


def ensure_state():
    ss = st.session_state
    defaults = {
        "units": U.METRIC, "source": T.SOURCE_DEMO, "route": None, "holds": None, "analysis": None, "pose": None,
        "placements": None, "foot_events": [], "fourlimb_store": {}, "pending": None, "align": None,
        "corners": {"route": [], "video": []}, "selected": None, "edit_mode": T.MODE_PICK, "move_pending": None,
        "last_click": None, "last_corner_click": {"route": None, "video": None},
        "body": {"height_m": None, "span_m": None}, "style": T.STYLE_BALANCED,
        "max_reach": 0.85, "max_down": 0.2, "match_finish": True, "scale_pct": 100, "show_feet": False, "fast_beam": False,
        "insight_open": False, "insight_cache": {}, "video_view": T.VIDEO_ORIGINAL, "sweep_rows": None,
    }
    for k, v in defaults.items():
        ss.setdefault(k, v)
    for k, v in WEIGHT_DEFAULTS.items():
        ss.setdefault(k, v)


# ============================================================================ loading: route, demo, video
def set_route(image_bgr, image_path, kind, name, rdir, holds, angle=40, body=None):
    ss = st.session_state
    ss.route = {"image": image_bgr, "image_path": image_path, "kind": kind, "name": name, "dir": rdir, "angle": int(angle)}
    ss.holds = [dict(h) for h in holds]
    ss.body = dict(body or {"height_m": None, "span_m": None})
    ss.analysis, ss.pose, ss.placements, ss.foot_events, ss.pending, ss.align = None, None, None, [], None, None
    ss.fourlimb_store, ss.selected, ss.move_pending, ss.last_click = {}, None, None, None
    ss.corners = {"route": [], "video": []}
    ss.insight_cache, ss.insight_open, ss.sweep_rows, ss.scale_pct, ss.show_feet = {}, False, None, 100, False


def refresh_observed():
    ss = st.session_state
    if ss.analysis is None:
        return
    ss.placements = recompute_observed(ss.analysis, ss.holds, ss.pose)
    ss.foot_events = recompute_feet(ss.analysis, ss.holds, ss.pose)
    ss.sweep_rows = None


def load_demo():
    ss = st.session_state
    d = DEMO
    cache = os.path.join(ROOT, d["cache"])
    img_path = os.path.join(ROOT, d["route_image"])
    img = cv2.imread(img_path)
    rf = load_route_file(cache)
    if rf:
        holds, angle, body = rf["holds"], rf.get("angle", 40), rf.get("body")
    else:
        cur = os.path.join(cache, "holds_edited.json")
        holds = json.load(open(cur)) if os.path.exists(cur) else kilter.detect_route_holds(img)
        angle, body = d.get("angle", 40), None
    set_route(img, img_path, "photo", d["name"], cache, holds, angle, body)
    if os.path.exists(os.path.join(cache, "analysis.json")):
        a = load_analysis(cache)
        a["cache_dir"] = cache
        a["video"] = os.path.join(ROOT, d["video"])
        if a.get("pose_file") != "pose_route.json":   # cache built before the route-image flow: photo == background frame
            a = apply_route(a, holds, np.eye(3), img_path, {"method": "identity", "inliers": 0})
        ss.analysis, ss.pose = a, load_analysis_pose(a)
        ss.align = a.get("alignment") or {"method": "auto", "inliers": 0}
        fl = os.path.join(cache, "fourlimb.json")
        ss.fourlimb_store = json.load(open(fl)) if os.path.exists(fl) else {}
        refresh_observed()
    ss.source_loaded = T.SOURCE_DEMO


def load_own_image(up, kind):
    ss = st.session_state
    data = up.getvalue()
    rdir = route_dir_for(data)
    os.makedirs(rdir, exist_ok=True)
    img_path = os.path.join(rdir, "route.png")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error(T.ROUTE_NONE_DETECTED)
        return
    cv2.imwrite(img_path, img)
    rf = load_route_file(rdir)
    if rf:
        holds, angle, body, msg = rf["holds"], rf.get("angle", 40), rf.get("body"), T.ROUTE_LOADED_SAVED
    else:
        holds, angle, body = kilter.detect_route_holds(img), 40, None
        msg = T.ROUTE_DETECTED.format(n=len(holds)) if holds else T.ROUTE_NONE_DETECTED
    name = os.path.splitext(up.name)[0]
    set_route(img, img_path, kind, name, rdir, holds, angle, body)
    ss.route_msg = msg
    ss.route_image_key = up.name + str(len(data))


def _run_with_progress(video, cache):
    bar = st.progress(0.0, text=T.STAGE_NAMES["pose"])
    stage_w = {"pose": (0.0, 0.6), "stabilize": (0.6, 0.9), "background": (0.9, 0.97), "holds": (0.97, 0.99), "observed beta": (0.99, 1.0)}

    def prog(stage, frac):
        key = stage.replace(" (cached)", "")
        lo, hi = stage_w.get(key, (0.0, 1.0))
        bar.progress(min(1.0, lo + (hi - lo) * frac), text=f"{T.STAGE_NAMES.get(key, key)} {frac:.0%}")

    t = time.time()
    try:
        analysis = analyze_video(video, cache, progress=prog, wall_type="led", detect_holds=False)
    except Exception as e:  # malformed video, no person visible, etc.
        bar.empty()
        st.error(T.ANALYZE_FAIL.format(err=e))
        st.stop()
    bar.progress(1.0, text=T.STAGE_DONE.format(s=time.time() - t))
    try:
        pose = load_pose(os.path.join(cache, "pose.json"))
        viz.render_pose_overlay_video(video, pose, os.path.join(cache, "overlay.mp4"))
    except Exception:  # noqa
        st.warning(T.OVERLAY_FAIL)
    return analysis


def analyze_own_video(up):
    ss = st.session_state
    os.makedirs(os.path.join(ROOT, "demo_output", "uploads"), exist_ok=True)
    path = os.path.join(ROOT, "demo_output", "uploads", up.name)
    data = up.getbuffer()
    with open(path, "wb") as f:
        f.write(data)
    cache = os.path.join(ROOT, "demo_output", "uploads", hashlib.md5(bytes(data)).hexdigest()[:10])
    analysis = _run_with_progress(path, cache)
    bg = cv2.imread(os.path.join(cache, "background.png"))
    ss.pending = {"analysis": analysis, "bg": bg}
    ss.corners = {"route": [], "video": []}
    if ss.route["kind"] == "photo":
        H, n = register.align_images(bg, ss.route["image"])
        if H is not None and register.plausible(H, bg.shape[1], bg.shape[0]):
            finish_alignment(H, {"method": "auto", "inliers": int(n)})
            return
        ss.align = {"method": "corners", "pending": True, "auto_failed": True}
    else:
        ss.align = {"method": "corners", "pending": True, "auto_failed": False}


def finish_alignment(H, info):
    ss = st.session_state
    a = apply_route(ss.pending["analysis"], ss.holds, H, ss.route["image_path"], info)
    ss.analysis, ss.pose = a, load_analysis_pose(a)
    ss.align, ss.pending, ss.fourlimb_store, ss.insight_cache = info, None, {}, {}
    ss.corners = {"route": [], "video": []}
    refresh_observed()


# ============================================================================ results
def current_weights():
    return Weights(**{k: float(st.session_state[k]) for k in WEIGHT_DEFAULTS})


def current_feas():
    ss = st.session_state
    return Feasibility(max_reach_frac=float(ss.max_reach), max_down_frac=float(ss.max_down), match_finish=bool(ss.match_finish))


def body_morph():
    ss = st.session_state
    return U.apply_body_inputs(ss.analysis["morphology"], ss.body.get("height_m"), ss.body.get("span_m"))


def start_and_rest():
    """Kilter rules in the setup: start = the green holds, finish = the pink holds.
    The observed hand assignment on the start holds is kept when it agrees with them."""
    ss = st.session_state
    setup = kilter.kilter_setup(ss.holds)
    start_set = set(setup["start_ids"])
    placements = list(ss.placements or [])
    st_obs, consumed = beta_mod.initial_state(placements)
    if st_obs and (not start_set or set(st_obs) <= start_set):
        return st_obs, placements[consumed:], setup
    if setup["start_state"]:
        while placements and placements[0]["hold_id"] in start_set:
            placements.pop(0)
        return tuple(setup["start_state"]), placements, setup
    return None, placements, setup


@st.cache_data(show_spinner=False, max_entries=256)
def _opt_cached(holds_json, morph_json, placements_json, w_tuple, f_tuple, scale, start_tuple, finish_tuple, fourlimb, fl_kwargs):
    return run_optimization(json.loads(holds_json), json.loads(morph_json), json.loads(placements_json), Weights(*w_tuple),
                            Feasibility(*f_tuple), morph_scale=scale, start_state=list(start_tuple) if start_tuple else None,
                            finish_ids=list(finish_tuple) if finish_tuple else None, fourlimb=fourlimb, **dict(fl_kwargs))


def compute(scale=1.0, fourlimb=False, fl_kwargs=()):
    ss = st.session_state
    st_, rest, setup = start_and_rest()
    morph, _ = body_morph()
    fin = setup["finish_ids"] or None
    args = (json.dumps(ss.holds, sort_keys=True), json.dumps(morph, sort_keys=True), json.dumps(rest, sort_keys=True),
            astuple(current_weights()), astuple(current_feas()), float(scale), tuple(st_) if st_ else None, tuple(fin) if fin else None)
    R = _opt_cached(*args, False, ())
    if fourlimb and "error" not in R:
        key = fourlimb_cache_key(ss.holds, R["weights"], R["feasibility"], float(scale), R["finish_ids"])
        pre = ss.fourlimb_store.get(key)
        if pre is not None:
            R = dict(R)
            R["optimized_4limb"] = pre
        else:
            R = _opt_cached(*args, True, tuple(sorted(dict(fl_kwargs).items())))
    return R


def fourlimb_precomputed(R):
    ss = st.session_state
    key = fourlimb_cache_key(ss.holds, R["weights"], R["feasibility"], 1.0, R["finish_ids"])
    return key in ss.fourlimb_store


def is_partial(R):
    obs = R.get("observed")
    fin = set(R.get("finish_ids", []))
    return bool(obs and obs["moves"] and not (set(obs["states"][-1]) & fin))


def tracking_gaps():
    ss = st.session_state
    return coach.tracking_gaps(ss.pose, ss.analysis["fps"], ss.analysis["n_frames"])


def insight_key():
    ss = st.session_state
    payload = {"route": ss.route["name"], "holds": ss.holds, "video": ss.analysis.get("video") if ss.analysis else None,
               "w": {k: ss[k] for k in WEIGHT_DEFAULTS}, "f": [ss.max_reach, ss.max_down, ss.match_finish],
               "body": ss.body, "angle": ss.route["angle"], "units": ss.units}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


@st.cache_resource(show_spinner=False)
def _llm_status():
    return coach.llm_available()


def apply_style():
    ss = st.session_state
    for k, v in STYLE_PRESETS[ss.style].items():
        ss[k] = v


def apply_preset(action):
    ss = st.session_state
    if action == "shorter":
        ss.scale_pct = 85
    elif action == "measured":
        ss.scale_pct = 100
    elif action == "feet":
        ss.show_feet = True


# ============================================================================ SIDEBAR
ensure_state()
ss = st.session_state
with st.sidebar:
    st.markdown(f"## 🧗 {T.APP_TITLE}")
    st.caption(T.SIDEBAR_CAPTION)
    src = st.radio(T.SOURCE_LABEL, [T.SOURCE_DEMO, T.SOURCE_OWN], horizontal=True, key="source")
    if src == T.SOURCE_DEMO and ss.get("source_loaded") != T.SOURCE_DEMO:
        load_demo()
    elif src == T.SOURCE_OWN and ss.get("source_loaded") != T.SOURCE_OWN:
        ss.route, ss.holds, ss.analysis, ss.pose, ss.placements, ss.pending, ss.align = None, None, None, None, None, None, None
        ss.source_loaded = T.SOURCE_OWN
    unit_pick = st.radio(T.UNITS_LABEL, [T.UNITS_METRIC, T.UNITS_IMPERIAL], horizontal=True,
                         index=0 if ss.units == U.METRIC else 1)
    ss.units = U.METRIC if unit_pick == T.UNITS_METRIC else U.IMPERIAL
    st.markdown("---")
    llm = _llm_status()
    if st.button(T.INSIGHT_BUTTON, type="primary", width="stretch", disabled=not llm.get("ok"), help=T.INSIGHT_HELP):
        ss.insight_open = True
    if not llm.get("ok"):
        st.caption(T.INSIGHT_DISABLED_CAPTION)

# ============================================================================ HEADER
st.markdown(f'<p class="hero-title">{T.APP_TITLE}</p>', unsafe_allow_html=True)
if ss.route is None:
    st.caption(T.HEADER_NO_ROUTE)
elif ss.analysis is None:
    st.caption(T.HEADER_ROUTE_ONLY.format(route=ss.route["name"]))
else:
    st.caption(T.HEADER_READY.format(route=ss.route["name"]))

# ---- Deeper-insight panel: between the header and the steps, so it is visible whichever step is open
insight_box = st.container()
with insight_box:
    if ss.insight_open:
        st.markdown(f"#### {T.INSIGHT_TITLE}")
        if ss.analysis is None or ss.route is None:
            st.info(T.INSIGHT_NEEDS_RESULTS)
        else:
            key = insight_key()
            hid = hid_map()
            R = compute()
            if "error" in R or R.get("optimized") is None:
                st.warning(T.NO_LINE_FOUND)
            else:
                partial = is_partial(R)
                if key not in ss.insight_cache:
                    with st.spinner(T.INSIGHT_WORKING):
                        A = ss.analysis
                        morph, _ = body_morph()
                        rep = injury.injury_report(R, hid, A.get("move_context"), A["fps"])
                        summary = coach.summary_for_llm(R, hid, partial, move_context=A.get("move_context"), foot_events=ss.foot_events,
                                                        pose=ss.pose, fps=A["fps"], board_angle_deg=ss.route["angle"],
                                                        arm_span_m=ss.body.get("span_m"), height_m=ss.body.get("height_m"),
                                                        injury=rep, gaps=tracking_gaps())
                        txt = coach.llm_insights(summary)
                        ss.insight_cache[key] = coach.parse_insights(txt) if txt else None
                parsed = ss.insight_cache.get(key)
                if parsed and (parsed.get("your_climb") or parsed.get("suggested")):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**{T.INSIGHT_YOUR}**")
                        for line in parsed.get("your_climb", []):
                            st.markdown(f"- {line}")
                    with c2:
                        st.markdown(f"**{T.INSIGHT_SUGGESTED}**")
                        for line in parsed.get("suggested", []):
                            st.markdown(f"- {line}")
                    st.caption(T.INSIGHT_FOOTER)
                else:
                    st.info(T.INSIGHT_UNAVAILABLE)
                    for line in coach.rule_based(R, hid, partial):
                        st.markdown(f"- {line}")
        if st.button(T.INSIGHT_CLOSE, key="insight_close"):
            ss.insight_open = False
            st.rerun()
        st.markdown("---")

tab_route, tab_climb, tab_explore = st.tabs([T.TAB_ROUTE, T.TAB_CLIMB, T.TAB_EXPLORE])


# ============================================================================ STEP 1 · ROUTE
with tab_route:
    # ---- 1a. route image input (own route only)
    if ss.source == T.SOURCE_OWN:
        kind_pick = st.radio(T.ROUTE_KIND_LABEL, [T.ROUTE_KIND_PHOTO, T.ROUTE_KIND_SCREENSHOT], horizontal=True)
        st.caption(T.ROUTE_KIND_WHY)
        kind = "photo" if kind_pick == T.ROUTE_KIND_PHOTO else "screenshot"
        up_img = st.file_uploader(T.ROUTE_UPLOAD_LABEL, type=["jpg", "jpeg", "png"])
        if up_img is not None and ss.get("route_image_key") != up_img.name + str(up_img.size):
            load_own_image(up_img, kind)
            st.rerun()
        if ss.route is not None and ss.route["kind"] != kind:
            ss.route["kind"] = kind
        if ss.get("route_msg"):
            st.info(ss.route_msg)

    if ss.route is None:
        st.info(T.HEADER_NO_ROUTE)
    else:
        route = ss.route
        # ---- 1b. board angle (stored with the route; display-only difficulty scaling in step 2)
        angle = st.number_input(T.BOARD_ANGLE_LABEL, min_value=0, max_value=70, value=int(route["angle"]), step=5)
        st.caption(T.BOARD_ANGLE_CAPTION)
        if int(angle) != route["angle"]:
            route["angle"] = int(angle)
            ss.insight_cache = {}

        for w in kilter.role_warnings(ss.holds):
            st.warning(w)

        left, right = st.columns([3, 2])
        # ---- 1c. hold editor (click canvas on the route image, role colours)
        with left:
            st.markdown(f"#### {T.HOLDS_HEADER}")
            mode = st.radio(T.CLICK_MODE_LABEL, T.CLICK_MODES, horizontal=True, index=T.CLICK_MODES.index(ss.edit_mode))
            if mode != ss.edit_mode:
                ss.edit_mode, ss.move_pending = mode, None
            if mode == T.MODE_MOVE and ss.move_pending is not None:
                st.info(T.MOVE_STEP2.format(hold=label_of(hid_map(), ss.move_pending)))
            else:
                st.caption(T.MODE_HINTS[mode])
            img = viz.to_pil(route["image"])
            viz.draw_holds(img, ss.holds, selected=ss.move_pending if mode == T.MODE_MOVE else ss.selected,
                           fill="role", label_mode="label")
            box = viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0])
            img = img.crop(box)
            from streamlit_image_coordinates import streamlit_image_coordinates
            click = streamlit_image_coordinates(img, key="hold_canvas", width=CANVAS_W)
            if click and click != ss.last_click:
                ss.last_click = click
                shown_w = click.get("width") or CANVAS_W
                sc = shown_w / img.width
                cx, cy = box[0] + click["x"] / sc, box[1] + click["y"] / sc
                dists = [(np.hypot(h["x"] - cx, h["y"] - cy), h["id"]) for h in ss.holds]
                nearest = min(dists)[1] if dists else None
                near_px = 0.04 * max(route["image"].shape[:2])
                near_enough = bool(dists) and min(dists)[0] < near_px
                if mode == T.MODE_PICK:
                    ss.selected = nearest if near_enough else None
                elif mode == T.MODE_ADD:
                    nid = next_id(ss.holds)
                    ss.holds.append(make_hold(nid, cx, cy, source="manual"))
                    ss.selected = nid
                    refresh_observed()
                elif mode == T.MODE_REMOVE and near_enough:
                    ss.holds = [h for h in ss.holds if h["id"] != nearest]
                    ss.selected = None
                    refresh_observed()
                elif mode == T.MODE_MOVE:
                    if ss.move_pending is None:
                        ss.move_pending = nearest if near_enough else None
                    else:
                        h = hid_map().get(ss.move_pending)
                        if h is not None:
                            h["x"], h["y"] = float(cx), float(cy)
                            refresh_observed()
                        ss.selected, ss.move_pending = ss.move_pending, None
                st.rerun()
            st.caption(T.LEGEND_ROLES)

        # ---- 1d. selected hold: role and grip rating
        with right:
            st.markdown(f"#### {T.SELECTED_HEADER}")
            hid = hid_map()
            if ss.selected is None or ss.selected not in hid:
                st.info(T.SELECT_HINT)
            else:
                h = hid[ss.selected]
                st.markdown(T.HOLD_LINE.format(label=h["label"], role=T.role_name(h.get("role"))))
                role_pick = st.selectbox(T.ROLE_LABEL, T.ROLE_ORDER, index=T.ROLE_ORDER.index(h.get("role") if h.get("role") in T.ROLE_ORDER else None),
                                         format_func=T.role_name)
                g = st.select_slider(T.GRIP_LABEL, options=[1, 2, 3, 4, 5], value=int(h.get("grip", 3)),
                                     format_func=lambda v: f"{v} · {GRIP_LABELS[v]}")
                st.caption(T.GRIP_CAPTION)
                if (role_pick != h.get("role")) or (g != h.get("grip", 3)):
                    h["role"], h["grip"], h["on_route"] = role_pick, int(g), True
                    refresh_observed()
                    st.rerun()
                if st.button(T.REMOVE_HOLD):
                    ss.holds = [x for x in ss.holds if x["id"] != h["id"]]
                    ss.selected = None
                    refresh_observed()
                    st.rerun()

            # ---- 1e. hold set actions
            st.markdown(f"#### {T.ALL_HOLDS_HEADER}")
            b1, b2 = st.columns(2)
            if b1.button(T.RESET_HOLDS, width="stretch"):
                ss.holds = kilter.detect_route_holds(route["image"])
                ss.selected, ss.move_pending = None, None
                refresh_observed()
                st.toast(T.RESET_TOAST)
                st.rerun()
            if b2.button(T.SAVE_ROUTE, width="stretch"):
                save_route_file(route["dir"], ss.holds, route["angle"], ss.body, route["name"])
                st.toast(T.SAVED_TOAST)
            st.caption(T.HOLD_COUNT.format(n=len(ss.holds), manual=sum(1 for h in ss.holds if h.get("source") == "manual")))

            # ---- 1f. hand placements from the video (status line + sequence expander)
            if ss.analysis is None:
                st.caption(T.NO_VIDEO_YET)
            else:
                n_pl = len(ss.placements or [])
                st.markdown(f"**{T.DETECTED_PLACEMENTS.format(n=n_pl)}**" if n_pl else f"**{T.DETECTED_NONE}**")
                with st.expander(T.SHOW_SEQUENCE):
                    A = ss.analysis
                    if n_pl:
                        rows = [{T.SEQUENCE_COLUMNS["#"]: i + 1, T.SEQUENCE_COLUMNS["hand"]: "L" if p["hand"] == LEFT else "R",
                                 T.SEQUENCE_COLUMNS["hold"]: label_of(hid, p["hold_id"]), T.SEQUENCE_COLUMNS["time"]: round(p["frame"] / A["fps"], 2)}
                                for i, p in enumerate(ss.placements)]
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                        pose_path, stab_path = os.path.join(A["cache_dir"], "pose.json"), os.path.join(A["cache_dir"], "stab.json")
                        if os.path.exists(pose_path) and os.path.exists(stab_path) and os.path.exists(A["video"]):
                            stab = stab_mod.load(stab_path)
                            if stab["static"]:
                                stab = {**stab, "H": [[[1, 0, 0], [0, 1, 0], [0, 0, 1]]] * len(stab["H"])}
                            sheet = viz.contact_sheet(A["video"], load_pose(pose_path), ss.placements, ss.holds, n=5,
                                                      holds_for_frame=lambda i, _s=stab: holds_for_video_frame(ss.holds, A, _s, i), fps=A["fps"])
                            if sheet is not None:
                                st.image(sheet, caption=T.KEY_FRAMES_CAPTION, width="stretch")

        # ---- 1g. hold table (collapsed)
        with st.expander(T.HOLD_TABLE_EXPANDER):
            df = pd.DataFrame([{T.HOLD_TABLE_COLUMNS["label"]: h["label"], T.HOLD_TABLE_COLUMNS["grip"]: h.get("grip", 3),
                                T.HOLD_TABLE_COLUMNS["role"]: T.role_name(h.get("role"))} for h in ss.holds])
            st.dataframe(df, hide_index=True, width="stretch")


# ============================================================================ STEP 2 · CLIMB
with tab_climb:
    if ss.route is None:
        st.info(T.NEED_ROUTE_FIRST)
    else:
        route = ss.route
        # ---- 2a. video input (own route) and alignment
        if ss.source == T.SOURCE_OWN:
            up_vid = st.file_uploader(T.VIDEO_UPLOAD_LABEL, type=["mp4", "mov", "m4v"])
            st.caption(T.VIDEO_GUIDE)
            if up_vid is not None and st.button(T.ANALYZE_BUTTON, type="primary"):
                analyze_own_video(up_vid)
                st.rerun()
        if ss.pending is not None and ss.align and ss.align.get("pending"):
            # ---- 2b. corner method: four board corners in the route image and in one video frame
            st.warning(T.ALIGN_FAILED if ss.align.get("auto_failed") else T.ALIGN_SCREENSHOT)
            from streamlit_image_coordinates import streamlit_image_coordinates
            which = "route" if len(ss.corners["route"]) < 4 else "video"
            k = len(ss.corners[which])
            if k < 4:
                st.info(T.CORNER_INSTRUCTION.format(corner=T.CORNER_NAMES[k], which=T.CORNER_ROUTE if which == "route" else T.CORNER_VIDEO, k=k + 1))
            cc1, cc2 = st.columns(2)
            for col, key_, img_bgr, title in ((cc1, "route", route["image"], T.CORNER_ROUTE_TITLE), (cc2, "video", ss.pending["bg"], T.CORNER_VIDEO_TITLE)):
                with col:
                    st.markdown(f"**{title}**")
                    pil = viz.to_pil(img_bgr)
                    d = viz.ImageDraw.Draw(pil)
                    for (px, py) in ss.corners[key_]:
                        d.ellipse([px - 12, py - 12, px + 12, py + 12], outline=(255, 80, 80), width=6)
                    click = streamlit_image_coordinates(pil, key=f"corner_{key_}", width=CANVAS_W)
                    if click and click != ss.last_corner_click[key_] and key_ == which and k < 4:
                        ss.last_corner_click[key_] = click
                        sc = (click.get("width") or CANVAS_W) / pil.width
                        ss.corners[key_].append((click["x"] / sc, click["y"] / sc))
                        st.rerun()
            if st.button(T.CORNER_RESET):
                ss.corners = {"route": [], "video": []}
                st.rerun()
            if len(ss.corners["route"]) == 4 and len(ss.corners["video"]) == 4:
                H = register.homography_from_corners(ss.corners["video"], ss.corners["route"])
                finish_alignment(H, {"method": "corners", "inliers": 4})
                st.rerun()
        elif ss.align:
            if ss.align.get("method") == "auto":
                st.success(T.ALIGN_MATCHED.format(n=ss.align.get("inliers", 0)) if ss.align.get("inliers") else T.ALIGN_MATCHED_SHORT)
            elif ss.align.get("method") == "corners":
                st.success(T.ALIGN_CORNERS_DONE)

        # ---- 2c. optional body inputs (override the video measurement, supply the pixel scale)
        with st.expander(T.BODY_EXPANDER):
            st.caption(T.BODY_EXPLAIN + " " + T.BODY_ZERO_HINT)
            unit = U.unit_label(ss.units)
            bc1, bc2 = st.columns(2)
            changed_body = False
            for col, key_, label in ((bc1, "height_m", T.HEIGHT_LABEL), (bc2, "span_m", T.SPAN_LABEL)):
                cur = ss.body.get(key_)
                shown = float(round(U.from_metres(cur, ss.units), 1)) if cur else 0.0
                v = col.number_input(label.format(unit=unit), min_value=0.0, max_value=300.0, value=shown,
                                     step=1.0 if ss.units == U.METRIC else 0.5, key=f"body_{key_}_{ss.units}")
                new = U.to_metres(float(v), ss.units) if v > 0 else None
                if (new is None) != (cur is None) or (new is not None and abs(new - (cur or 0)) > 1e-4):
                    ss.body[key_] = new
                    changed_body = True
            if changed_body:
                ss.insight_cache = {}
                if os.path.exists(route_file(route["dir"])):
                    save_route_file(route["dir"], ss.holds, route["angle"], ss.body, route["name"])
                st.rerun()

        if ss.analysis is None:
            st.info(T.NEED_ANALYSIS)
        else:
            A = ss.analysis
            hid = hid_map()
            morph, px_per_m = body_morph()

            # ---- 2d. video player: original / skeleton overlay, full width
            view = st.radio(T.VIDEO_VIEW_LABEL, [T.VIDEO_ORIGINAL, T.VIDEO_SKELETON], horizontal=True, key="video_view")
            ov = os.path.join(A["cache_dir"], "overlay.mp4")
            vpath = A["video"] if view == T.VIDEO_ORIGINAL else (ov if os.path.exists(ov) else A["video"])
            if os.path.exists(vpath):
                st.video(vpath)
            else:
                st.info(T.VIDEO_MISSING)

            # ---- 2e. measurements card row
            m1, m2, m3 = st.columns([1, 1, 2])
            with m1:
                card(T.CARD_SPAN, U.length_text(morph["arm_span_px"], px_per_m, morph["arm_span_px"], ss.units) if px_per_m else T.CARD_FROM_VIDEO, T.CARD_SPAN_SUB)
            with m2:
                card(T.CARD_LEG, U.length_text(morph.get("leg_len_px"), px_per_m, morph["arm_span_px"], ss.units), T.CARD_LEG_SUB)
            with m3:
                st.radio(T.STYLE_LABEL, T.STYLES, horizontal=True, key="style", on_change=apply_style)
                st.caption(T.STYLE_CAPTIONS[ss.style])
            if px_per_m is None:
                st.caption(T.CARD_RELATIVE_PROMPT)

            # ---- 2f. your climb vs suggested
            R = compute()
            if "error" in R or R.get("optimized") is None:
                st.error(T.NO_LINE_FOUND)
            else:
                obs, opt, cmp_ = R["observed"], R["optimized"], R["comparison"]
                partial = is_partial(R)
                gaps = tracking_gaps()
                c1, c2, c3 = st.columns(3)
                with c1:
                    card(T.CARD_YOURS, T.moves_text(obs["n_moves"]) if obs else "—", "", "obs")
                with c2:
                    card(T.CARD_SUGGESTED, T.moves_text(opt["n_moves"]), "", "opt")
                with c3:
                    if not obs:
                        card(T.CARD_SAVED, "—", T.NO_OBSERVED)
                    elif partial:
                        card(T.CARD_SAVED, T.CARD_SAVED_PARTIAL, T.CARD_SAVED_PARTIAL_SUB)
                    elif cmp_.get("same_sequence"):
                        card(T.CARD_SAVED, T.CARD_SAME, T.CARD_SAME_SUB)
                    else:
                        card(T.CARD_SAVED, f"{max(0.0, cmp_.get('improvement_frac', 0.0)):.0%}", T.CARD_SAVED_SUB)
                mult = kilter.difficulty_multiplier(route["angle"])
                st.caption(T.DIFFICULTY_LINE.format(angle=route["angle"], mult=f"{mult:.1f}"))
                if partial or gaps:
                    st.warning(T.PARTIAL_BANNER.format(ranges=coach.gaps_text(gaps)) if gaps else T.PARTIAL_BANNER_CLIP)
                if not obs:
                    st.info(T.NO_OBSERVED)
                if opt.get("relaxed_to"):
                    st.info(T.RELAXED)
                cmp_img = viz.render_comparison(route["image"], ss.holds, obs, opt, cmp_, viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0]),
                                                partial=partial, max_height=COMPARISON_MAX_H, titles=(T.CARD_YOURS, T.CARD_SUGGESTED))
                st.image(cmp_img)
                st.caption(viz.LEGEND_LINE)
                with st.expander(T.FULL_SIZE):
                    st.image(viz.render_comparison(route["image"], ss.holds, obs, opt, cmp_, viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0]),
                                                   partial=partial, titles=(T.CARD_YOURS, T.CARD_SUGGESTED)), width="stretch")

                # ---- 2g. why + tips
                why = coach.why_block(R, hid, partial)
                if why:
                    st.markdown(f"**{T.WHY_HEADER}**")
                    st.markdown(" ".join(why))
                st.markdown(f"**{T.TIPS_HEADER}**")
                for line in coach.rule_based(R, hid, partial):
                    st.markdown(f"- {line}")

                # ---- 2h. injury flags (heuristic)
                st.markdown(f"**{T.INJURY_HEADER}** · {T.INJURY_DISCLAIMER}")
                rep = injury.injury_report(R, hid, A.get("move_context"), A["fps"])
                i1, i2 = st.columns(2)
                for col, title, rows in ((i1, T.CARD_YOURS, rep["observed"]), (i2, T.CARD_SUGGESTED, rep["suggested"])):
                    with col:
                        st.markdown(f"*{title}*")
                        if rows:
                            st.dataframe(pd.DataFrame([{T.INJURY_COLUMNS["severity"]: r["severity"], T.INJURY_COLUMNS["move"]: r["move"],
                                                        T.INJURY_COLUMNS["why"]: r["why"], T.INJURY_COLUMNS["instead"]: r["instead"]} for r in rows]),
                                         hide_index=True, width="stretch")
                        else:
                            st.caption(T.INJURY_NONE)
                if rep["suggested"]:
                    st.caption(T.INJURY_SUGGESTED_HAS)

                # ---- 2i. collapsed extras
                with st.expander(T.SAFER_EXPANDER):
                    for tip in T.SAFER_TIPS:
                        st.markdown(f"- {tip}")
                with st.expander(T.LIMITS_EXPANDER):
                    for line in T.LIMITS:
                        st.markdown(f"- {line}")
                with st.expander(T.MEASURE_EXPANDER):
                    ratio = morph["ratios"].get("arm_span_over_height_proxy")
                    st.markdown(f"- {T.APE_LABEL}: **{ratio:.2f}**" if ratio else f"- {T.APE_LABEL}: —")
                    st.markdown(f"- {T.TRACKED_LABEL.format(pct=round(100 * morph['pose_detection_rate']))}")
                    ctx = A.get("move_context") or []
                    if ctx:
                        st.markdown(f"**{T.POSE_TABLE_HEADER}**")
                        st.dataframe(pd.DataFrame([{T.POSE_TABLE_COLUMNS["move"]: i + 1,
                                                    T.POSE_TABLE_COLUMNS["elbow"]: (round(c["support_elbow_min_deg"]) if c["support_elbow_min_deg"] else None),
                                                    T.POSE_TABLE_COLUMNS["hip"]: round(c["hip_travel_px"] / morph["arm_span_px"], 2),
                                                    T.POSE_TABLE_COLUMNS["duration"]: round(c["duration_frames"] / A["fps"], 2)} for i, c in enumerate(ctx)]),
                                     hide_index=True, width="stretch")
                with st.expander(T.DETAILS_EXPANDER):
                    st.markdown(f"**{T.DETAILS_RAW_HEADER}**")
                    rows = [{"line": T.CARD_YOURS, "cost": round(obs["total_cost"], 2), "moves": obs["n_moves"], "max reach (× arm span)": round(obs["max_reach_frac"], 2)}] if obs else []
                    rows.append({"line": T.CARD_SUGGESTED, "cost": round(opt["total_cost"], 2), "moves": opt["n_moves"], "max reach (× arm span)": round(opt["max_reach_frac"], 2)})
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                    if cmp_.get("crux"):
                        st.markdown(f"- crux: {cmp_['crux']['explanation']}")
                    if R["feasibility"].max_reach_frac > current_feas().max_reach_frac + 1e-9:
                        st.caption(T.AUTO_WIDENED.format(r=R["feasibility"].max_reach_frac))
                    st.markdown(f"**{T.DETAILS_SEQUENCES}**")
                    st.markdown(f"<span class='obs'><b>{T.CARD_YOURS}</b></span>: {sequence_text(obs, hid)}", unsafe_allow_html=True)
                    st.markdown(f"<span class='opt'><b>{T.CARD_SUGGESTED}</b></span>: {sequence_text(opt, hid)}", unsafe_allow_html=True)
                    if obs:
                        st.markdown(f"**{T.DETAILS_DIFF}**")
                        st.image(viz.render_diff(route["image"], ss.holds, obs, opt, cmp_, viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0])), width="stretch")
                with st.expander(T.HOW_EXPANDER):
                    st.markdown(T.HOW_IT_WORKS)
                    st.markdown(f"**{T.HOW_FACTS_HEADER}**")
                    srch = opt.get("search", {})
                    n_hand = sum(1 for h in ss.holds if h.get("on_route", True) and h.get("role") != "foot")
                    facts = [T.HOW_FACT_FRAMES.format(n=A["n_frames"], fps=A["fps"]),
                             T.HOW_FACT_CAMERA_STATIC if A["camera"]["static"] else T.HOW_FACT_CAMERA_MOVING.format(px=A["camera"]["max_shift_px"]),
                             T.HOW_FACT_TRACKED.format(pct=morph["pose_detection_rate"]),
                             T.HOW_FACT_HOLDS.format(n=len(ss.holds)),
                             T.HOW_FACT_ALIGN.format(method=(ss.align or {}).get("method", "unknown")),
                             T.HOW_FACT_STATE.format(n=n_hand),
                             T.HOW_FACT_ENVELOPE.format(r=R["feasibility"].max_reach_frac),
                             T.HOW_FACT_SEARCH.format(method="exact A*" if srch.get("exact", True) else f"beam {srch.get('beam')}",
                                                      n=srch.get("n_expanded", 0), ms=srch.get("runtime_s", 0) * 1000)]
                    for f_ in facts:
                        st.markdown(f"- {f_}")
                with st.expander(T.ADVANCED_EXPANDER):
                    st.caption(T.ADVANCED_CAPTION)
                    for k, (label, lo, hi, step) in T.SLIDERS.items():
                        st.slider(label, lo, hi, step=step, key=k)
                    st.slider(T.REACH_LIMIT_LABEL, 0.5, 1.1, step=0.05, key="max_reach")
                    st.slider(T.DOWN_LIMIT_LABEL, 0.0, 0.6, step=0.05, key="max_down")
                    st.checkbox(T.MATCH_FINISH_LABEL, key="match_finish")


# ============================================================================ STEP 3 · EXPLORE
with tab_explore:
    if ss.route is None or ss.analysis is None:
        st.info(T.NEED_ROUTE_FIRST if ss.route is None else T.NEED_ANALYSIS)
    else:
        route, A, hid = ss.route, ss.analysis, hid_map()
        morph, px_per_m = body_morph()
        st.markdown(f"#### {T.EXPLORE_HEADER}")
        st.caption(T.EXPLORE_CAPTION)
        # ---- 3a. presets
        p1, p2, p3 = st.columns([1, 1, 2])
        p1.button(T.PRESET_SHORTER, on_click=apply_preset, args=("shorter",), width="stretch", help=T.PRESET_SHORTER_EXPLAIN)
        p2.button(T.PRESET_FEET, on_click=apply_preset, args=("feet",), width="stretch")
        p3.caption(T.PRESET_SHORTER_EXPLAIN)

        # ---- 3b. body-size slider (real units when a scale exists, else percent)
        span_m = morph["arm_span_px"] / px_per_m if px_per_m else None
        if span_m:
            unit = U.unit_label(ss.units)
            span_u = U.from_metres(span_m, ss.units)
            lo, hi = int(round(span_u * 0.7)), int(round(span_u * 1.25))
            cur = int(round(span_u * ss.scale_pct / 100))
            val = st.slider(T.SIM_SLIDER_UNITS.format(unit=unit), lo, hi, value=min(hi, max(lo, cur)), step=1)
            new_pct = int(round(100 * val / span_u))
            sim_m = U.to_metres(val, ss.units)
            st.caption(T.SIM_CAPTION_UNITS.format(span=U.format_length(span_m, ss.units), sim=U.format_length(sim_m, ss.units),
                                                  delta=U.format_delta(sim_m - span_m, ss.units)))
        else:
            val = st.slider(T.SIM_SLIDER_PCT, 70, 125, value=int(ss.scale_pct), step=5)
            new_pct = int(val)
        if new_pct != ss.scale_pct:
            ss.scale_pct = new_pct
            st.rerun()
        if ss.scale_pct != 100 and st.button(T.BACK_TO_MEASURED, on_click=apply_preset, args=("measured",)):
            pass

        # ---- 3c. measured vs simulated
        Rm = compute(1.0)
        Rs = compute(ss.scale_pct / 100.0)
        pm, ps = Rm.get("optimized"), Rs.get("optimized")
        if not pm or not ps:
            st.error(T.SIM_FAIL)
        else:
            if span_m:
                amount = U.format_length(abs(span_m * (1 - ss.scale_pct / 100)), ss.units)
            else:
                amount = f"{abs(100 - ss.scale_pct)} %"
            sim_label = T.CARD_YOU if ss.scale_pct == 100 else (T.CARD_SIM_SHORTER if ss.scale_pct < 100 else T.CARD_SIM_TALLER).format(amount=amount)
            same = pm["states"] == ps["states"]
            diff_h = sorted(set(ps["holds_used"]) ^ set(pm["holds_used"]))
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card(T.CARD_YOU, T.moves_text(pm["n_moves"]), "", "opt")
            with c2:
                card(sim_label, T.moves_text(ps["n_moves"]), "")
            with c3:
                card(T.CARD_DIFFERENT, T.CARD_DIFF_NO if same else T.CARD_DIFF_YES,
                     T.CARD_DIFF_YES_SUB if not same else (T.CARD_DIFF_NO_SUB if ps["total_cost"] > pm["total_cost"] + 1e-9 else T.CARD_DIFF_SAME_SUB))
            with c4:
                card(T.CARD_DIFF_HOLDS, ", ".join(label_of(hid, i) for i in diff_h) if diff_h else T.CARD_DIFF_HOLDS_NONE, "")
            box = viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0])
            left = viz.render_beta_panel(route["image"], ss.holds, pm, T.PANEL_YOU, viz.C_OPTIMIZED, box, crux=False)
            rightp = viz.render_beta_panel(route["image"], ss.holds, ps, sim_label, (255, 120, 200), box, crux=False)
            cc1, cc2 = st.columns(2)
            cc1.image(viz._limit_height(left, COMPARISON_MAX_H) if hasattr(viz, "_limit_height") else left, width="stretch")
            cc2.image(viz._limit_height(rightp, COMPARISON_MAX_H) if hasattr(viz, "_limit_height") else rightp, width="stretch")
            st.caption(T.SIM_LEGEND)
            with st.expander(T.SWEEP_EXPANDER):
                st.markdown(f"<span class='opt'><b>{T.CARD_YOU}</b></span>: {sequence_text(pm, hid)}", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#ff78c8'><b>{sim_label}</b></span>: {sequence_text(ps, hid)}", unsafe_allow_html=True)
                if st.button(T.SWEEP_BUTTON):
                    rows = []
                    for pct in (70, 80, 90, 100, 110, 120):
                        rr = compute(pct / 100.0).get("optimized")
                        if rr:
                            rows.append({T.SWEEP_COLUMNS["pct"]: pct, T.SWEEP_COLUMNS["moves"]: rr["n_moves"],
                                         T.SWEEP_COLUMNS["holds"]: " ".join(label_of(hid, i) for i in rr["holds_used"])})
                    ss.sweep_rows = rows
                if ss.sweep_rows:
                    st.dataframe(pd.DataFrame(ss.sweep_rows), hide_index=True, width="stretch")

        # ---- 3d. feet plan (hands + feet), off by default
        st.markdown("---")
        pre = fourlimb_precomputed(Rm) if "error" not in Rm else False
        est = 1 if pre else 15
        show_feet = st.toggle(T.FEET_TOGGLE, key="show_feet")
        st.caption(T.FEET_WAIT_FAST if pre else T.FEET_WAIT_SLOW.format(s=est))
        if show_feet:
            fl_kwargs = {"fourlimb_exact_threshold": 0, "fourlimb_max_expansions": 1, "fourlimb_beam": 80} if ss.fast_beam else {}
            with st.spinner(T.FEET_SPINNER.format(s=est)):
                R4 = compute(1.0, fourlimb=True, fl_kwargs=tuple(sorted(fl_kwargs.items())))
            q = R4.get("optimized_4limb")
            if q is None:
                st.error(T.FEET_NONE)
            else:
                obs = R4.get("observed")
                f1, f2 = st.columns(2)
                with f1:
                    card(T.CARD_WITH_FEET, T.CARD_WITH_FEET_SUB.format(h=q["n_hand_moves"], f=q["n_foot_moves"]), "", "opt")
                with f2:
                    card(T.CARD_FEET_ON, ", ".join(label_of(hid, i) for i in q["feet_used"]) or T.CARD_FEET_NONE, "")
                feet_obs = feet_for_result(ss.foot_events, obs) if obs else None
                box = viz.crop_box(ss.holds, route["image"].shape[1], route["image"].shape[0])
                st.image(viz.render_comparison(route["image"], ss.holds, obs, q, R4["comparison"], box, partial=is_partial(R4),
                                               feet_obs=feet_obs, feet_opt=q["feet_by_state"], max_height=COMPARISON_MAX_H,
                                               titles=(T.CARD_YOURS, T.CARD_SUGGESTED)))
                st.caption(viz.LEGEND_LINE + " · " + T.FEET_LEGEND)
                with st.expander(T.FEET_DETAILS):
                    st.markdown(f"<span class='opt'><b>{T.CARD_SUGGESTED}</b></span>: {sequence_text(q, hid)}", unsafe_allow_html=True)
        with st.expander(T.ADVANCED_FEET):
            st.checkbox(T.BEAM_LABEL, key="fast_beam")
