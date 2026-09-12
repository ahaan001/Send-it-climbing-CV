"""Judge-readable renderings: holds, observed vs optimized beta, diff view,
feasibility graph. All drawing is PIL on top of the climber-free wall image.
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# palette (RGB)
C_OBSERVED = (255, 140, 0)      # orange
C_OPTIMIZED = (0, 200, 255)     # cyan
C_GEOMETRIC = (190, 120, 255)   # violet
C_CRUX = (255, 40, 40)
C_FEET = (120, 255, 200)        # mint: feet markers
C_SHARED = (255, 255, 255)
C_OFF = (150, 150, 150)
C_START = (80, 255, 120)
C_FINISH = (255, 230, 60)
GRIP_COLORS = {1: (60, 200, 90), 2: (150, 210, 70), 3: (240, 200, 60), 4: (250, 140, 50), 5: (230, 60, 60)}


def _font(size: int, bold=True):
    candidates = []
    try:
        import matplotlib
        base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        candidates += [os.path.join(base, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    except Exception:
        pass
    candidates += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    for c in candidates:
        if os.path.exists(c):
            try:
                return ImageFont.truetype(c, size)
            except Exception:
                continue
    return ImageFont.load_default()


def to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def crop_box(holds: list, w: int, h: int, margin_frac: float = 0.08):
    """Bounding box around the route holds (+margin) so panels focus on the wall."""
    pts = [(hh["x"], hh["y"]) for hh in holds if hh.get("on_route", True)] or [(hh["x"], hh["y"]) for hh in holds]
    if not pts:
        return (0, 0, w, h)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    m = max(bw, bh) * margin_frac + 0.03 * max(w, h)
    x0, y0 = max(0, int(min(xs) - m)), max(0, int(min(ys) - m))
    x1, y1 = min(w, int(max(xs) + m)), min(h, int(max(ys) + m))
    return (x0, y0, x1, y1)


def _scale(img: Image.Image):
    return max(img.size) / 1000.0


def draw_holds(img: Image.Image, holds: list, selected: Optional[int] = None, show_labels=True,
               dim_off_route=True, radius_px: Optional[float] = None, label_mode="id"):
    d = ImageDraw.Draw(img, "RGBA")
    s = _scale(img)
    r = radius_px or 11 * s
    f = _font(int(13 * s))
    for h in holds:
        x, y = h["x"], h["y"]
        on = h.get("on_route", True)
        if not on and dim_off_route:
            d.ellipse([x - r * 0.8, y - r * 0.8, x + r * 0.8, y + r * 0.8], outline=C_OFF + (170,), width=max(1, int(2 * s)))
            continue
        col = GRIP_COLORS.get(int(h.get("grip", 3)), GRIP_COLORS[3])
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (200,), outline=(255, 255, 255, 230), width=max(1, int(2 * s)))
        if h.get("role") == "start":
            d.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], outline=C_START + (255,), width=max(2, int(3 * s)))
        elif h.get("role") == "finish":
            d.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], outline=C_FINISH + (255,), width=max(2, int(3 * s)))
        if selected is not None and h["id"] == selected:
            d.ellipse([x - r * 2.1, y - r * 2.1, x + r * 2.1, y + r * 2.1], outline=(255, 255, 255, 255), width=max(2, int(3 * s)))
        if show_labels:
            txt = str(h["id"]) if label_mode == "id" else f"{h['id']}·g{h.get('grip', 3)}"
            tw = d.textlength(txt, font=f)
            d.rounded_rectangle([x + r * 0.9, y - r * 1.3, x + r * 0.9 + tw + 6 * s, y - r * 1.3 + 15 * s],
                                radius=3 * s, fill=(0, 0, 0, 150))
            d.text((x + r * 0.9 + 3 * s, y - r * 1.3), txt, fill=(255, 255, 255, 255), font=f)
    return img


def _arrow(d: ImageDraw.ImageDraw, p0, p1, color, width, s, shorten=0.0):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    v = p1 - p0
    n = np.linalg.norm(v)
    if n < 1e-6:
        return
    u = v / n
    a = p0 + u * shorten
    b = p1 - u * shorten
    d.line([tuple(a), tuple(b)], fill=color, width=width)
    head = 12 * s
    left = b - u * head + np.array([-u[1], u[0]]) * head * 0.55
    right = b - u * head - np.array([-u[1], u[0]]) * head * 0.55
    d.polygon([tuple(b), tuple(left), tuple(right)], fill=color)


def draw_path(img: Image.Image, holds: list, result: dict, color, crux=True, number=True, width_mult=1.0):
    """Arrows for each hand move (from -> to); crux move in red. Foot moves
    are drawn by draw_feet(), so they are skipped here (numbering counts
    hand moves only)."""
    if not result or not result.get("moves"):
        return img
    d = ImageDraw.Draw(img, "RGBA")
    s = _scale(img)
    hid = {h["id"]: h for h in holds}
    f = _font(int(14 * s))
    w = max(2, int(5 * s * width_mult))
    k = 0
    for i, m in enumerate(result["moves"]):
        limb = m.get("limb") or m.get("hand")
        if limb not in ("LEFT", "RIGHT"):
            continue
        k += 1
        a, b = hid.get(m["from"]), hid.get(m["to"])
        if a is None or b is None:
            continue
        is_crux = crux and result.get("crux_index") == i and len(result["moves"]) > 1
        col = C_CRUX if is_crux else color
        _arrow(d, (a["x"], a["y"]), (b["x"], b["y"]), col + (235,), w + (2 if is_crux else 0), s, shorten=13 * s)
        if number:
            mx, my = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
            txt = f"{k}{'L' if limb == 'LEFT' else 'R'}"
            tw = d.textlength(txt, font=f)
            d.rounded_rectangle([mx - tw / 2 - 4 * s, my - 9 * s, mx + tw / 2 + 4 * s, my + 9 * s], radius=4 * s,
                                fill=col + (230,))
            d.text((mx - tw / 2, my - 8 * s), txt, fill=(0, 0, 0, 255), font=f)
        if is_crux:
            lab = "highest-cost move"
            fl = _font(int(12 * s))
            tw = d.textlength(lab, font=fl)
            d.rounded_rectangle([b["x"] + 14 * s, b["y"] + 8 * s, b["x"] + 14 * s + tw + 8 * s, b["y"] + 8 * s + 16 * s],
                                radius=3 * s, fill=C_CRUX + (220,))
            d.text((b["x"] + 18 * s, b["y"] + 9 * s), lab, fill=(255, 255, 255, 255), font=fl)
    return img


def draw_feet(img: Image.Image, holds: list, feet_by_state, color=C_FEET, hand_states=None, moves=None):
    """Feet layer. feet_by_state: list aligned with states, each [left_foot_id, right_foot_id]
    (None = not on a hold). Draws hollow squares at every foot hold used, with the
    state numbers in which that foot was there, and a thin line to the hands'
    midpoint for the state where the foot is first placed."""
    if not feet_by_state:
        return img
    d = ImageDraw.Draw(img, "RGBA")
    s = _scale(img)
    hid = {h["id"]: h for h in holds}
    f = _font(int(11 * s))
    half = 8 * s
    placed = {}   # (foot_id) -> list of state indices
    first_seen = {}
    for i, feet in enumerate(feet_by_state):
        if not feet:
            continue
        for side, fid in zip(("L", "R"), feet):
            if fid is None or fid not in hid:
                continue
            placed.setdefault(fid, []).append(f"{i}{side.lower()}")
            if (fid, side) not in first_seen:
                first_seen[(fid, side)] = i
    for fid, tags in placed.items():
        h = hid[fid]
        x, y = h["x"], h["y"]
        d.rectangle([x - half, y - half, x + half, y + half], outline=color + (255,), width=max(2, int(2.5 * s)))
        d.rectangle([x - half * 0.55, y - half * 0.55, x + half * 0.55, y + half * 0.55], fill=color + (120,))
        txt = "f " + ",".join(tags[:3]) + ("…" if len(tags) > 3 else "")
        tw = d.textlength(txt, font=f)
        d.rounded_rectangle([x + half + 2 * s, y + 2 * s, x + half + 2 * s + tw + 6 * s, y + 2 * s + 13 * s], radius=3 * s, fill=(0, 0, 0, 160))
        d.text((x + half + 5 * s, y + 2 * s), txt, fill=color + (255,), font=f)
    if hand_states:
        for (fid, side), i in first_seen.items():
            if i < len(hand_states):
                L, R = hand_states[i][0], hand_states[i][1]
                if L in hid and R in hid:
                    mx, my = (hid[L]["x"] + hid[R]["x"]) / 2, (hid[L]["y"] + hid[R]["y"]) / 2
                    d.line([(hid[fid]["x"], hid[fid]["y"]), (mx, my)], fill=color + (110,), width=max(1, int(1.5 * s)))
    return img


def _title_bar(img: Image.Image, title: str, subtitle: str = "", color=(255, 255, 255)):
    s = _scale(img)
    bar_h = int(56 * s) if subtitle else int(36 * s)
    out = Image.new("RGB", (img.width, img.height + bar_h), (18, 18, 22))
    out.paste(img, (0, bar_h))
    d = ImageDraw.Draw(out)
    d.text((10 * s, 6 * s), title, fill=color, font=_font(int(20 * s)))
    if subtitle:
        d.text((10 * s, 30 * s), subtitle, fill=(210, 210, 210), font=_font(int(13 * s), bold=False))
    return out


def render_beta_panel(bg_bgr: np.ndarray, holds: list, result: Optional[dict], title: str, color,
                      box=None, subtitle: str = "", show_labels=True, crux=True, feet=None) -> Image.Image:
    """feet: optional list aligned with result['states'] of [left_foot, right_foot]
    hold ids (None = none); for four-limb results pass result['feet_by_state']."""
    img = to_pil(bg_bgr)
    draw_holds(img, holds, show_labels=show_labels)
    if feet:
        draw_feet(img, holds, feet, hand_states=result.get("states") if result else None)
    if result:
        draw_path(img, holds, result, color, crux=crux)
    if box:
        img = img.crop(box)
    if not subtitle and result:
        nh = result.get("n_hand_moves", result["n_moves"])
        nf = result.get("n_foot_moves", 0)
        subtitle = f"cost {result['total_cost']:.2f}  |  {nh} hand moves" + (f" + {nf} foot moves" if nf else "") + f"  |  max reach {result['max_reach_frac']:.0%} of arm span"
        srch = result.get("search")
        if srch and result.get("limbs") == "all":
            subtitle += f"  |  {'exact A*' if srch['exact'] else 'beam ' + str(srch['beam']) + ' (approx.)'}"
    return _title_bar(img, title, subtitle, color)


def render_comparison(bg_bgr: np.ndarray, holds: list, observed: Optional[dict], optimized: Optional[dict],
                      comparison: dict, box=None, partial: bool = False, feet_obs=None, feet_opt=None) -> Image.Image:
    h_img, w_img = bg_bgr.shape[:2]
    box = box or crop_box(holds, w_img, h_img)
    left = render_beta_panel(bg_bgr, holds, observed, "OBSERVED beta (from video)" + (" · partial clip" if partial else ""), C_OBSERVED, box, feet=feet_obs)
    imp = comparison.get("improvement_frac")
    sub = None
    fourlimb = bool(optimized and optimized.get("limbs") == "all")
    if fourlimb:
        sub = None   # render_beta_panel builds the hand+foot / search-method subtitle
    elif optimized and partial:
        sub = f"cost {optimized['total_cost']:.2f}  |  {optimized['n_moves']} hand moves  |  max reach {optimized['max_reach_frac']:.0%}  |  full-route plan"
    elif optimized and imp is not None:
        sub = (f"cost {optimized['total_cost']:.2f}  |  {optimized['n_moves']} hand moves  |  "
               f"max reach {optimized['max_reach_frac']:.0%}  |  {imp:+.0%} vs observed" if imp != 0 else
               f"cost {optimized['total_cost']:.2f}  |  {optimized['n_moves']} hand moves  |  same as observed")
    right = render_beta_panel(bg_bgr, holds, optimized,
                              "FULL-BODY plan (hands + feet, min-cost search)" if fourlimb else "OPTIMIZED beta (min-cost search)",
                              C_OPTIMIZED, box, subtitle=sub or "", crux=False, feet=feet_opt)
    gap = 12
    out = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), (18, 18, 22))
    out.paste(left, (0, 0))
    out.paste(right, (left.width + gap, 0))
    return out


def render_diff(bg_bgr: np.ndarray, holds: list, observed: Optional[dict], optimized: Optional[dict],
                comparison: dict, box=None) -> Image.Image:
    """Both paths on one image; holds coloured by who uses them."""
    img = to_pil(bg_bgr)
    d = ImageDraw.Draw(img, "RGBA")
    s = _scale(img)
    r = 11 * s
    hid = {h["id"]: h for h in holds}
    shared = set(comparison.get("shared_holds", []))
    only_o = set(comparison.get("only_observed", []))
    only_p = set(comparison.get("only_optimized", []))
    for h in holds:
        x, y = h["x"], h["y"]
        if h["id"] in shared:
            col = C_SHARED
        elif h["id"] in only_o:
            col = C_OBSERVED
        elif h["id"] in only_p:
            col = C_OPTIMIZED
        else:
            if h.get("on_route", True):
                d.ellipse([x - r * 0.7, y - r * 0.7, x + r * 0.7, y + r * 0.7], outline=(200, 200, 200, 160), width=max(1, int(2 * s)))
            continue
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (220,), outline=(0, 0, 0, 200), width=max(1, int(2 * s)))
    if observed:
        draw_path(img, holds, observed, C_OBSERVED, crux=True, number=False, width_mult=0.8)
    if optimized:
        draw_path(img, holds, optimized, C_OPTIMIZED, crux=False, number=False, width_mult=1.1)
    if box:
        img = img.crop(box)
    legend = "white = both   orange = observed only   cyan = optimized only   red arrow = highest-cost observed move"
    return _title_bar(img, "Observed vs optimized", legend)


def render_graph(bg_bgr: np.ndarray, holds: list, edges: list, path: Optional[dict] = None, box=None,
                 title="Personalized feasibility graph") -> Image.Image:
    """Hold-level graph: an edge means this climber can hold both holds at
    once (span within their reach envelope). Line brightness ~ closeness."""
    img = to_pil(bg_bgr)
    d = ImageDraw.Draw(img, "RGBA")
    s = _scale(img)
    hid = {h["id"]: h for h in holds}
    for i, j, r in edges:
        a, b = hid[i], hid[j]
        alpha = int(40 + 150 * max(0.0, 1.0 - r))
        d.line([(a["x"], a["y"]), (b["x"], b["y"])], fill=(255, 255, 255, alpha), width=max(1, int(2 * s)))
    draw_holds(img, holds, show_labels=True)
    if path:
        draw_path(img, holds, path, C_OPTIMIZED, crux=False)
    if box:
        img = img.crop(box)
    n_route = sum(1 for h in holds if h.get("on_route", True))
    return _title_bar(img, title, f"{n_route} route holds, {len(edges)} feasible hold pairs for this climber")


def pose_overlay_frame(frame_bgr: np.ndarray, landmarks: dict, color=(0, 255, 180)) -> np.ndarray:
    from .pose import SKELETON_EDGES
    out = frame_bgr.copy()
    for a, b in SKELETON_EDGES:
        pa, pb = landmarks.get(a), landmarks.get(b)
        if pa and pb and pa[2] > 0.4 and pb[2] > 0.4:
            cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 3, cv2.LINE_AA)
    for name, p in landmarks.items():
        if p[2] > 0.4:
            cv2.circle(out, (int(p[0]), int(p[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
    return out


def render_pose_overlay_video(video_path: str, pose: dict, out_path: str, holds: Optional[list] = None,
                              max_side: int = 720, fps_out: Optional[float] = None, holds_for_frame=None):
    """Skeleton overlay video (H.264 'avc1', browser-playable) downscaled for
    the UI. holds_for_frame(frame_idx) -> holds in that frame's pixel space
    lets route holds stick to the wall while the camera pans."""
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = fps_out or (cap.get(5) or 30)
    sc = min(1.0, max_side / max(w, h))
    ow, oh = int(w * sc), int(h * sc)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (ow, oh))
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        lm = pose["frames"].get(i)
        if lm:
            fr = pose_overlay_frame(fr, lm)
        hs = holds_for_frame(i) if holds_for_frame else holds
        if hs:
            for hh in hs:
                if hh.get("on_route", True) and 0 <= hh["x"] < w and 0 <= hh["y"] < h:
                    col = (80, 255, 120) if hh.get("role") == "start" else (60, 230, 255) if hh.get("role") == "finish" else (0, 200, 255)
                    cv2.circle(fr, (int(hh["x"]), int(hh["y"])), 12, col, 2, cv2.LINE_AA)
        writer.write(cv2.resize(fr, (ow, oh)))
        i += 1
    cap.release()
    writer.release()
    return out_path


def contact_sheet(video_path: str, pose: dict, placements: list, holds: list, n: int = 6, max_side=360,
                  holds_for_frame=None) -> Image.Image:
    """Key frames at each observed hand placement, with skeleton + the hold."""
    hid = {h["id"]: h for h in holds}
    picks = placements[:: max(1, len(placements) // n)][:n] if placements else []
    cap = cv2.VideoCapture(video_path)
    tiles = []
    for p in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(p["frame"]))
        ok, fr = cap.read()
        if not ok:
            continue
        lm = pose["frames"].get(int(p["frame"]))
        if lm:
            fr = pose_overlay_frame(fr, lm)
        hh = hid.get(p["hold_id"])
        if hh and holds_for_frame:
            hh = {x["id"]: x for x in holds_for_frame(int(p["frame"]))}.get(p["hold_id"], hh)
        if hh:
            cv2.circle(fr, (int(hh["x"]), int(hh["y"])), 18, (0, 140, 255), 4, cv2.LINE_AA)
        sc = max_side / max(fr.shape[:2])
        fr = cv2.resize(fr, (int(fr.shape[1] * sc), int(fr.shape[0] * sc)))
        cv2.putText(fr, f"{'L' if p['hand']=='LEFT' else 'R'} -> hold {p['hold_id']}  f{p['frame']}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(fr)
    cap.release()
    if not tiles:
        return None
    hmax = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(18, 18, 22)) for t in tiles]
    return to_pil(np.hstack(tiles))
