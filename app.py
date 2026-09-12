"""Send It -- Personalized Climbing Beta Optimizer (Streamlit UI).

Run:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sendit import viz
from sendit.holds import GRIP_LABELS, make_hold, next_id
from sendit.optimizer import Weights, Feasibility, hold_graph_edges, LEFT
from sendit.pipeline import analyze_video, load_analysis, recompute_observed, run_optimization
from sendit.pose import load_pose

ROOT = os.path.dirname(os.path.abspath(__file__))
DEMOS = json.load(open(os.path.join(ROOT, "demo_assets", "demos.json")))["demos"]
DEMO_BY_KEY = {d["key"]: d for d in DEMOS}

st.set_page_config(page_title="Send It · Beta Optimizer", page_icon="🧗", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.metric-card {background: linear-gradient(135deg,#16181d,#22262e); border:1px solid #2f3440; border-radius:14px; padding:14px 16px;}
.metric-card .lab {font-size:0.78rem; color:#9aa3b2; text-transform:uppercase; letter-spacing:.08em;}
.metric-card .val {font-size:1.7rem; font-weight:700; color:#f5f7fa; line-height:1.15;}
.metric-card .sub {font-size:0.8rem; color:#b7c0cf;}
.obs {color:#ff8c00;} .opt {color:#00c8ff;} .crux {color:#ff4040;}
.hero-title {font-size:2.0rem; font-weight:800; margin-bottom:0;}
.tagline {color:#9aa3b2; margin-top:0;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; background:#2b3140; color:#dfe5ee; font-size:0.78rem; margin-right:6px;}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def card(label, value, sub="", cls=""):
    st.markdown(f'<div class="metric-card"><div class="lab">{label}</div>'
                f'<div class="val {cls}">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def curated_path(cache_dir):
    return os.path.join(cache_dir, "holds_edited.json")


def load_source(demo_key=None, upload_path=None, wall_type="auto", force=False):
    """Load cached analysis for a demo, or run the pipeline on an upload."""
    if demo_key:
        d = DEMO_BY_KEY[demo_key]
        cache = os.path.join(ROOT, d["cache"])
        video = os.path.join(ROOT, d["video"])
        if os.path.exists(os.path.join(cache, "analysis.json")) and not force:
            analysis = load_analysis(cache)
            mode = "cached"
        else:
            analysis = _run_with_progress(video, cache, d.get("wall_type", "auto"), force)
            mode = "live"
    else:
        cache = os.path.join(ROOT, "demo_output", "uploads", hashlib.md5(open(upload_path, "rb").read()).hexdigest()[:10])
        analysis = _run_with_progress(upload_path, cache, wall_type, force)
        mode = "live"
    analysis["cache_dir"] = cache
    pose_wall = load_pose(os.path.join(cache, "pose_wall.json"))
    bg = cv2.imread(os.path.join(cache, "background.png"))
    holds = analysis["holds"]
    if os.path.exists(curated_path(cache)):
        holds = json.load(open(curated_path(cache)))
        curated = True
    else:
        curated = False
    return analysis, pose_wall, bg, holds, mode, curated


def _run_with_progress(video, cache, wall_type, force):
    bar = st.progress(0.0, text="Starting pipeline…")
    stage_w = {"pose": (0.0, 0.55), "stabilize": (0.55, 0.85), "background": (0.85, 0.9), "holds": (0.9, 0.95), "observed beta": (0.95, 1.0)}

    def prog(stage, frac):
        key = stage.replace(" (cached)", "")
        lo, hi = stage_w.get(key, (0.0, 1.0))
        bar.progress(min(1.0, lo + (hi - lo) * frac), text=f"{stage} … {frac:.0%}")

    t = time.time()
    analysis = analyze_video(video, cache, force=force, progress=prog, wall_type=wall_type)
    bar.progress(1.0, text=f"Done in {time.time() - t:.1f}s")
    # overlay video for the climber tab
    try:
        from sendit.viz import render_pose_overlay_video
        pose = load_pose(os.path.join(cache, "pose.json"))
        render_pose_overlay_video(video, pose, os.path.join(cache, "overlay.mp4"))
    except Exception as e:  # noqa
        st.warning(f"Overlay video not rendered: {e}")
    return analysis


def ensure_state():
    ss = st.session_state
    ss.setdefault("source_key", None)
    ss.setdefault("analysis", None)
    ss.setdefault("holds", None)
    ss.setdefault("placements", None)
    ss.setdefault("selected", None)
    ss.setdefault("edit_mode", "select")
    ss.setdefault("last_click", None)
    ss.setdefault("box", None)
    ss.setdefault("finish_choice", "observed")


def set_source(demo_key=None, upload_path=None, wall_type="auto", force=False):
    analysis, pose_wall, bg, holds, mode, curated = load_source(demo_key, upload_path, wall_type, force)
    ss = st.session_state
    ss.analysis, ss.pose_wall, ss.bg = analysis, pose_wall, bg
    ss.holds = [dict(h) for h in holds]
    ss.placements = recompute_observed(analysis, ss.holds, pose_wall) if curated else analysis["placements"]
    ss.mode, ss.curated = mode, curated
    ss.selected, ss.last_click = None, None
    ss.box = viz.crop_box(ss.holds, bg.shape[1], bg.shape[0])
    ss.source_key = demo_key or upload_path
    ss.finish_choice = "observed"


def hid_map():
    return {h["id"]: h for h in st.session_state.holds}


def refresh_observed():
    ss = st.session_state
    ss.placements = recompute_observed(ss.analysis, ss.holds, ss.pose_wall)


def finish_ids_from_choice():
    ss = st.session_state
    if ss.finish_choice == "observed":
        return None  # pipeline default: role == finish
    if ss.finish_choice == "top":
        route = [h for h in ss.holds if h.get("on_route", True) and h.get("role") != "foot"]
        return [min(route, key=lambda h: h["y"])["id"]] if route else None
    return [int(ss.finish_choice)]


def optimize(weights, feas, morph_scale=1.0):
    ss = st.session_state
    return run_optimization(ss.holds, ss.analysis["morphology"], ss.placements, weights, feas,
                            morph_scale=morph_scale, finish_ids=finish_ids_from_choice())


def cost_breakdown_chart(results: list, colors: list):
    """Stacked bars: per-move cost by term, one group per beta."""
    fig = go.Figure()
    terms = ["reach", "grip", "move", "travel", "direction", "cross", "foot"]
    palette = {"reach": "#4f8cff", "grip": "#ff6b6b", "move": "#a0a7b4", "travel": "#7bd389",
               "direction": "#ffd166", "cross": "#c77dff", "foot": "#ff9f43"}
    for res, col in zip(results, colors):
        if not res:
            continue
        x = [f"{res['label'][:3]} · {i + 1}{'L' if m['hand'] == LEFT else 'R'}" for i, m in enumerate(res["moves"])]
        for t in terms:
            y = [m["terms"].get(t, 0.0) for m in res["moves"]]
            if sum(y) < 1e-9:
                continue
            fig.add_bar(name=t, x=x, y=y, marker_color=palette[t], legendgroup=t,
                        showlegend=(res is results[0]), hovertemplate=f"{t}: %{{y:.2f}}<extra></extra>")
    fig.update_layout(barmode="stack", height=320, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", yaxis_title="move cost", xaxis_title="hand moves (observed left, optimized right)")
    return fig


def sequence_text(res, hid):
    if not res:
        return "—"
    L, R = res["start_state"]
    parts = [f"start L:{hid[L]['label'] if L in hid else L} R:{hid[R]['label'] if R in hid else R}"]
    for m in res["moves"]:
        parts.append(f"{'L' if m['hand'] == LEFT else 'R'}→{hid[m['to']]['label'] if m['to'] in hid else m['to']} ({m['reach_frac']:.0%})")
    return "  ·  ".join(parts)


# ----------------------------------------------------------------------------- sidebar
ensure_state()
with st.sidebar:
    st.markdown("## 🧗 Send It")
    st.caption("Personalized climbing beta optimizer")
    src = st.radio("Source", ["Demo climb", "Upload video"], horizontal=True)
    if src == "Demo climb":
        names = {d["name"]: d["key"] for d in DEMOS}
        pick = st.selectbox("Choose a demo", list(names.keys()))
        key = names[pick]
        c1, c2 = st.columns(2)
        if c1.button("Load", type="primary", width="stretch") or st.session_state.source_key is None:
            set_source(demo_key=key)
        if c2.button("Re-run live", width="stretch", help="Recompute pose, stabilization and holds from the video (not cached)"):
            set_source(demo_key=key, force=True)
        st.caption(DEMO_BY_KEY[key]["blurb"])
    else:
        up = st.file_uploader("Climbing video (mp4/mov/avi)", type=["mp4", "mov", "avi", "m4v"])
        wall = st.selectbox("Wall type", ["auto", "color", "led"], format_func=lambda x: {"auto": "Auto-detect", "color": "Normal gym wall (colour holds)", "led": "Lit board (Kilter / Moon / Tension)"}[x])
        if up is not None and st.button("Analyze", type="primary"):
            os.makedirs(os.path.join(ROOT, "demo_output", "uploads"), exist_ok=True)
            path = os.path.join(ROOT, "demo_output", "uploads", up.name)
            with open(path, "wb") as f:
                f.write(up.getbuffer())
            set_source(upload_path=path, wall_type=wall)

    st.markdown("---")
    st.markdown("### Optimizer settings")
    max_reach = st.slider("Reach envelope (max hand-to-hand span, × arm span)", 0.5, 1.1, 0.85, 0.05,
                          help="An edge is feasible only if the climber can hold both holds at once. Auto-widened to cover any reach the climber actually performed.")
    w_reach = st.slider("Reach weight", 0.0, 3.0, 1.0, 0.1, help="(span / 0.4 arm span)² per move")
    w_grip = st.slider("Grip-quality weight", 0.0, 3.0, 0.8, 0.1, help="(rating−1)/4 of the target hold")
    w_move = st.slider("Per-move penalty", 0.0, 2.0, 0.35, 0.05, help="Fixed cost of every hand movement")
    with st.expander("Advanced terms"):
        w_travel = st.slider("Travel weight", 0.0, 2.0, 0.25, 0.05)
        w_dir = st.slider("Sideways / downward weight", 0.0, 2.0, 0.4, 0.05)
        w_cross = st.slider("Crossed-hands weight", 0.0, 2.0, 0.5, 0.05)
        w_foot = st.slider("Foot-support weight", 0.0, 2.0, 0.4, 0.05)
        max_down = st.slider("Max downward move (× arm span)", 0.0, 0.6, 0.2, 0.05)
        match_finish = st.checkbox("Require both hands on finish", value=False)
    WEIGHTS = Weights(w_reach=w_reach, w_grip=w_grip, w_move=w_move, w_travel=w_travel, w_dir=w_dir, w_cross=w_cross, w_foot=w_foot)
    FEAS = Feasibility(max_reach_frac=max_reach, max_down_frac=max_down, match_finish=match_finish)

if st.session_state.analysis is None:
    st.info("Load a demo climb or upload a video from the sidebar.")
    st.stop()

ss = st.session_state
A = ss.analysis
hid = hid_map()
bg = ss.bg

# ----------------------------------------------------------------------------- header
st.markdown('<p class="hero-title">Send It · Personalized Climbing Beta Optimizer</p>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Video → pose → holds → <b>your</b> movement-cost graph → minimum-cost beta. '
            'Here is what you did, what the optimizer recommends, and why.</p>', unsafe_allow_html=True)
pills = [f"{'cached analysis' if ss.mode == 'cached' else 'live analysis'}",
         f"{A['n_frames']} frames @ {A['fps']:.0f} fps", f"pose in {A['timing']['pose_s']:.1f}s",
         "camera: static" if A["camera"]["static"] else f"camera motion compensated ({A['camera']['max_shift_px']:.0f}px pan)",
         f"holds: {A['hold_method']}" + (" + human-corrected" if ss.curated else "")]
st.markdown(" ".join(f'<span class="pill">{p}</span>' for p in pills), unsafe_allow_html=True)

tab_route, tab_climber, tab_opt, tab_person, tab_method = st.tabs(
    ["1 · Route & holds", "2 · Climber", "3 · Optimize", "4 · Personalize", "Method"])

# ----------------------------------------------------------------------------- tab 1: holds editor
with tab_route:
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Detected holds (click to edit)")
        ss.edit_mode = st.radio("Click mode", ["select", "add", "remove", "move selected"], horizontal=True,
                                index=["select", "add", "remove", "move selected"].index(ss.edit_mode))
        img = viz.to_pil(bg)
        viz.draw_holds(img, ss.holds, selected=ss.selected)
        box = ss.box
        img = img.crop(box)
        disp_w = 560
        scale = disp_w / img.width
        from streamlit_image_coordinates import streamlit_image_coordinates
        click = streamlit_image_coordinates(img, key="hold_canvas", width=disp_w)
        if click and click != ss.last_click:
            ss.last_click = click
            cx = box[0] + click["x"] / scale
            cy = box[1] + click["y"] / scale
            dists = [(np.hypot(h["x"] - cx, h["y"] - cy), h["id"]) for h in ss.holds]
            nearest = min(dists)[1] if dists else None
            near_enough = dists and min(dists)[0] < 0.08 * A["morphology"]["arm_span_px"]
            if ss.edit_mode == "select":
                ss.selected = nearest if near_enough else None
            elif ss.edit_mode == "add":
                nid = next_id(ss.holds)
                ss.holds.append(make_hold(nid, cx, cy, source="manual"))
                ss.selected = nid
                refresh_observed()
            elif ss.edit_mode == "remove" and near_enough:
                ss.holds = [h for h in ss.holds if h["id"] != nearest]
                ss.selected = None
                refresh_observed()
            elif ss.edit_mode == "move selected" and ss.selected is not None:
                hid_map()[ss.selected]["x"], hid_map()[ss.selected]["y"] = cx, cy
                refresh_observed()
            st.rerun()
        st.caption("Green ring = start · yellow ring = finish · fill colour = grip rating (green excellent → red terrible) · grey = off-route")
    with right:
        st.markdown("#### Selected hold")
        if ss.selected is None or ss.selected not in hid_map():
            st.info("Click a hold in *select* mode to rate it, mark it start/finish/foot-only, or toggle route membership.")
        else:
            h = hid_map()[ss.selected]
            st.markdown(f"**Hold {h['id']}** · source: `{h['source']}` · ({h['x']:.0f}, {h['y']:.0f})")
            g = st.select_slider("Grip quality (1 = excellent … 5 = terrible)", options=[1, 2, 3, 4, 5], value=int(h["grip"]),
                                 format_func=lambda v: f"{v} · {GRIP_LABELS[v]}")
            role = st.selectbox("Role", ["none", "start", "finish", "foot"], index=["none", "start", "finish", "foot"].index(h["role"] or "none"),
                                format_func=lambda r: {"none": "regular hand hold", "start": "start hold", "finish": "finish hold", "foot": "foot-only hold"}[r])
            on = st.toggle("On route (hands may use it)", value=bool(h.get("on_route", True)))
            changed = (g != h["grip"]) or ((None if role == "none" else role) != h["role"]) or (on != h.get("on_route", True))
            if changed:
                h["grip"], h["role"], h["on_route"] = int(g), (None if role == "none" else role), bool(on)
                refresh_observed()
                st.rerun()
            if st.button("Delete this hold"):
                ss.holds = [x for x in ss.holds if x["id"] != h["id"]]
                ss.selected = None
                refresh_observed()
                st.rerun()
        st.markdown("#### Hold set")
        c1, c2, c3 = st.columns(3)
        if c1.button("Reset to auto-detected"):
            ss.holds = [dict(h) for h in A["holds"]]
            ss.selected = None
            refresh_observed()
            st.rerun()
        if c2.button("Save as curated"):
            with open(curated_path(A["cache_dir"]), "w") as f:
                json.dump(ss.holds, f, indent=1)
            ss.curated = True
            st.success("Saved. This hold set now loads by default for this video.")
        if c3.button("Re-detect sequence"):
            refresh_observed()
            st.rerun()
        n_route = sum(1 for h in ss.holds if h.get("on_route", True))
        st.caption(f"{len(ss.holds)} holds · {n_route} on route · "
                   f"{sum(1 for h in ss.holds if h['source'] == 'dwell')} inferred from where the climber's hands dwelled · "
                   f"{sum(1 for h in ss.holds if h['source'] == 'manual')} added manually")
        df = pd.DataFrame([{"id": h["id"], "grip": h["grip"], "role": h["role"] or "", "on_route": h["on_route"], "source": h["source"]} for h in ss.holds])
        st.dataframe(df, height=220, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- tab 2: climber
with tab_climber:
    m = A["morphology"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Arm span (measured)", f"{m['arm_span_px']:.0f} px", "fingertip to fingertip, robust 92nd-percentile across the clip")
    with c2:
        card("Leg length", f"{m['leg_len_px']:.0f} px" if m.get("leg_len_px") else "n/a", "thigh + shin")
    with c3:
        card("Span / height proxy", f"{m['ratios']['arm_span_over_height_proxy']:.2f}" if m["ratios"].get("arm_span_over_height_proxy") else "n/a", "calibration-free ratio (torso+legs as height proxy)")
    with c4:
        card("Pose detection", f"{m['pose_detection_rate']:.0%}", f"{m['n_pose_frames']} frames with a skeleton")
    st.caption("All lengths are in this video's pixel space. Reach costs are ratios (hold distance ÷ arm span), so the pixel scale cancels. "
               "Measured, not calibrated: no real-world units are claimed.")
    v1, v2 = st.columns([1, 1])
    with v1:
        st.markdown("#### Pose tracking")
        ov = os.path.join(A["cache_dir"], "overlay.mp4")
        if os.path.exists(ov):
            st.video(ov)
        else:
            st.info("Overlay video not rendered for this source.")
    with v2:
        st.markdown("#### Observed hand sequence (from wrist/index contact dwell)")
        if ss.placements:
            rows = []
            for i, p in enumerate(ss.placements):
                rows.append({"#": i + 1, "hand": "L" if p["hand"] == LEFT else "R", "hold": hid.get(p["hold_id"], {}).get("label", p["hold_id"]),
                             "frame": p["frame"], "time (s)": round(p["frame"] / A["fps"], 2)})
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=min(400, 40 + 35 * len(rows)), width="stretch")
        else:
            st.warning("No hand contacts found with the current hold set. Add holds where the climber's hands were, or lower the reach envelope.")
        sheet = viz.contact_sheet(A["video"], load_pose(os.path.join(A["cache_dir"], "pose.json")) if os.path.exists(os.path.join(A["cache_dir"], "pose.json")) else ss.pose_wall,
                                  [p for p in A["placements"]], A["holds"], n=5) if A["placements"] and A["camera"]["static"] else None
        if sheet is not None:
            st.image(sheet, caption="Key frames at auto-detected placements (original camera frame)", width="stretch")
    ctx = A.get("move_context") or []
    if ctx:
        with st.expander("Measured pose context per observed move (diagnostic, not in the objective)"):
            st.dataframe(pd.DataFrame([{"move": i + 1, "support-arm elbow min (°)": (round(c["support_elbow_min_deg"]) if c["support_elbow_min_deg"] else None),
                                        "hip travel (× arm span)": round(c["hip_travel_px"] / m["arm_span_px"], 2), "duration (s)": round(c["duration_frames"] / A["fps"], 2)}
                                       for i, c in enumerate(ctx)]), hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- tab 3: optimize
with tab_opt:
    top = st.columns([2, 1])
    with top[1]:
        route_hand = [h for h in ss.holds if h.get("on_route", True) and h.get("role") != "foot"]
        finish_opts = ["observed", "top"] + [str(h["id"]) for h in sorted(route_hand, key=lambda h: h["y"])]
        ss.finish_choice = st.selectbox("Finish hold", finish_opts, index=finish_opts.index(ss.finish_choice) if ss.finish_choice in finish_opts else 0,
                                        format_func=lambda v: {"observed": "highest hold the climber reached", "top": "top-most route hold"}.get(v, f"hold {v}"))
    R = optimize(WEIGHTS, FEAS)
    if "error" in R:
        st.error(R["error"])
        st.stop()
    obs, opt, geo, cmp_ = R["observed"], R["optimized"], R["geometric"], R["comparison"]
    if opt is None:
        st.error("No feasible beta found even after relaxing the reach envelope. Check start/finish roles and route membership.")
        st.stop()
    with top[0]:
        if obs and obs["n_moves"] > 0:
            imp = cmp_.get("improvement_frac", 0.0)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                card("Observed cost", f"{obs['total_cost']:.2f}", f"{obs['n_moves']} hand moves · max reach {obs['max_reach_frac']:.0%}", "obs")
            with c2:
                card("Optimized cost", f"{opt['total_cost']:.2f}", f"{opt['n_moves']} hand moves · max reach {opt['max_reach_frac']:.0%}", "opt")
            with c3:
                card("Cost reduction", f"{imp:+.0%}" if not cmp_.get("same_sequence") else "0%", "(observed − optimized) ÷ observed, same objective")
            with c4:
                cx = cmp_.get("crux")
                card("Highest-cost observed move", f"{cx['cost']:.2f}" if cx else "—",
                     f"{'L' if cx and cx['hand'] == LEFT else 'R'} {hid.get(cx['from'], {}).get('label', '?') if cx else ''} → {hid.get(cx['to'], {}).get('label', '?') if cx else ''}", "crux")
        else:
            st.warning("No observed hand sequence to compare against; showing the optimized beta only.")
    if R["feasibility"].max_reach_frac > FEAS.max_reach_frac + 1e-9:
        st.caption(f"Reach envelope auto-widened to {R['feasibility'].max_reach_frac:.2f} × arm span because the climber demonstrated that reach on video.")
    if opt.get("relaxed_to"):
        st.warning(f"Graph was disconnected under the envelope; loosened to {opt['relaxed_to']:.2f} × arm span to find a path.")

    st.image(viz.render_comparison(bg, ss.holds, obs, opt, cmp_, ss.box), width="stretch")
    if cmp_.get("crux"):
        st.markdown(f"**Predicted crux (highest-cost move under our model):** {cmp_['crux']['explanation']}")
    if obs and cmp_.get("same_sequence"):
        st.success("The observed beta already matches the optimizer's minimum-cost sequence for this climber.")
    elif obs and cmp_.get("same_holds"):
        st.info("Same holds as the observed beta, but with a different hand order.")
    elif obs:
        so = ", ".join(f"H{i}" for i in cmp_.get("only_observed", [])) or "none"
        sp = ", ".join(f"H{i}" for i in cmp_.get("only_optimized", [])) or "none"
        st.markdown(f"**Difference:** the optimizer drops **{so}** and adds **{sp}**; shared holds: {', '.join(f'H{i}' for i in cmp_.get('shared_holds', []))}.")

    st.markdown("##### Sequences")
    st.markdown(f"<span class='obs'><b>Observed</b></span>: {sequence_text(obs, hid)}", unsafe_allow_html=True)
    st.markdown(f"<span class='opt'><b>Optimized</b></span>: {sequence_text(opt, hid)}", unsafe_allow_html=True)

    st.markdown("##### Where the cost comes from")
    st.plotly_chart(cost_breakdown_chart([obs, opt], [viz.C_OBSERVED, viz.C_OPTIMIZED]), width="stretch")

    e1, e2 = st.columns(2)
    with e1:
        with st.expander("Why not just the shortest geometric path?", expanded=False):
            if geo:
                st.markdown(f"The pure shortest-distance path (no reach normalization, no grip, no per-move cost) scores **{geo['total_cost']:.2f}** under our objective "
                            f"vs **{opt['total_cost']:.2f}** for the optimized beta ({geo['n_moves']} vs {opt['n_moves']} moves, max reach {geo['max_reach_frac']:.0%} vs {opt['max_reach_frac']:.0%}).")
                st.markdown(f"<span style='color:#be78ff'><b>Geometric</b></span>: {sequence_text(geo, hid)}", unsafe_allow_html=True)
                st.image(viz.render_beta_panel(bg, ss.holds, geo, "Shortest geometric path (baseline)", viz.C_GEOMETRIC, ss.box, crux=False), width="stretch")
    with e2:
        with st.expander("Personalized feasibility graph", expanded=False):
            edges = hold_graph_edges(ss.holds, R["climber"], R["feasibility"])
            st.image(viz.render_graph(bg, ss.holds, edges, opt, ss.box), width="stretch")
            st.caption("An edge joins two holds this climber can hold simultaneously (span ≤ reach envelope). "
                       "The search runs over hand-pair states; each state change moves one hand along one of these edges.")
    with st.expander("Diff view (both betas on one wall)"):
        st.image(viz.render_diff(bg, ss.holds, obs, opt, cmp_, ss.box), width="stretch")

# ----------------------------------------------------------------------------- tab 4: personalize
with tab_person:
    st.markdown("#### Same route, different body")
    st.caption("The climber's measured morphology is scaled to simulate a shorter or taller climber. Every feasible edge and every cost is recomputed; the optimizer re-plans.")
    scale_pct = st.slider("Simulated effective reach (% of measured)", 70, 125, 85, 5)
    Rm = optimize(WEIGHTS, FEAS, 1.0)
    Rs = optimize(WEIGHTS, FEAS, scale_pct / 100.0)
    pm, ps = Rm["optimized"], Rs["optimized"]
    if pm and ps:
        em = hold_graph_edges(ss.holds, Rm["climber"], Rm["feasibility"])
        es = hold_graph_edges(ss.holds, Rs["climber"], Rs["feasibility"])
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            card("Measured climber", f"{pm['total_cost']:.2f}", f"{pm['n_moves']} moves · {len(em)} feasible hold pairs", "opt")
        with c2:
            card(f"Simulated {scale_pct}% reach", f"{ps['total_cost']:.2f}", f"{ps['n_moves']} moves · {len(es)} feasible hold pairs")
        with c3:
            same = pm["states"] == ps["states"]
            card("Beta changes?", "no" if same else "YES", "different hold sequence" if not same else "same sequence, higher cost" if ps["total_cost"] > pm["total_cost"] else "same sequence")
        with c4:
            diff_h = sorted(set(ps["holds_used"]) ^ set(pm["holds_used"]))
            card("Holds that differ", ", ".join(f"H{i}" for i in diff_h) if diff_h else "none", "symmetric difference of hold sets")
        if ps.get("relaxed_to"):
            st.warning(f"For the simulated climber the graph was disconnected at the chosen envelope; loosened to {ps['relaxed_to']:.2f}.")
        left = viz.render_beta_panel(bg, ss.holds, pm, "MEASURED climber · optimized beta", viz.C_OPTIMIZED, ss.box, crux=False)
        right = viz.render_beta_panel(bg, ss.holds, ps, f"SIMULATED {scale_pct}% reach · optimized beta", (255, 120, 200), ss.box, crux=False)
        cc1, cc2 = st.columns(2)
        cc1.image(left, width="stretch")
        cc2.image(right, width="stretch")
        st.markdown(f"<span class='opt'><b>Measured</b></span>: {sequence_text(pm, hid)}", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#ff78c8'><b>Simulated</b></span>: {sequence_text(ps, hid)}", unsafe_allow_html=True)
        st.markdown("###### Cost of the same route across morphologies")
        rows = []
        for pct in (70, 80, 90, 100, 110, 120):
            rr = optimize(WEIGHTS, FEAS, pct / 100.0)["optimized"]
            if rr:
                rows.append({"reach %": pct, "optimized cost": round(rr["total_cost"], 2), "moves": rr["n_moves"],
                             "max reach": f"{rr['max_reach_frac']:.0%}", "holds": " ".join(f"H{i}" for i in rr["holds_used"]),
                             "envelope relaxed": rr["relaxed_to"] or ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.error("Could not compute a beta for one of the climbers.")

# ----------------------------------------------------------------------------- tab 5: method
with tab_method:
    st.markdown(r"""
#### What is optimized
A **state** is the pair of holds under the climber's hands, $s=(h_L, h_R)$. An **action** moves one hand to another
on-route hold. We run exact **Dijkstra** from the observed start state to any state with a hand on the finish hold.

#### Edge feasibility (personalized)
A move is allowed only if the hand-to-hand span after the move is within the climber's **reach envelope**
(default 0.85 × their measured arm span, auto-widened to any span they demonstrated on video), the hand does not
drop more than 0.2 × arm span, and the target is on the route and not foot-only.

#### Movement cost (dimensionless, per move)
$$C = w_r\left(\tfrac{\text{span}/\text{arm span}}{0.4}\right)^2 + w_g\,\tfrac{\text{grip}-1}{4} + w_m + w_t\,\tfrac{\text{travel}}{\text{arm span}} + w_d\,\text{dir} + w_x\,\text{cross} + w_f\,\text{foot}$$

* **reach** — quadratic in the normalized span: the same wall distance costs more for a smaller climber.
* **grip** — the user's 1–5 hold rating (subjective input; 1 → 0 penalty, 5 → full penalty).
* **move** — fixed cost per hand movement, so the optimizer does not ladder through every hold.
* **travel / direction / cross** — geometry of the moving hand: distance, sideways-or-down component, crossed hands.
* **foot** — lower-body context proxy: no hold inside the climber's leg window below the target hold.

The observed sequence is scored with the **same** function, so the comparison is apples to apples.

#### Scope and honesty
This prototype optimizes the **major hand-hold sequence**. Feet and body pose enter only as contextual cost
modifiers; explicit four-limb configuration-space search is the natural next step. Costs are heuristic, not energy or
force; hold ratings are user preferences; 2D pose has perspective error. Computer vision is imperfect in arbitrary gyms,
so every CV stage can be corrected by hand.
""")
