"""Every user-facing string in the Send It app, in one place.

Audience: a recreational climber with no optimisation background. Plain words,
short sentences. Words that must not appear in the primary flow: state, A*,
Dijkstra, beam, envelope, objective, px, symmetric difference, normalised,
proxy, feasible edge, joint states, homography, observed, optimized, beta,
cost. Technical wording is allowed only inside HOW_IT_WORKS and the Details /
Advanced sections.
"""

# ---------------------------------------------------------------- app / sidebar
PAGE_TITLE = "Send It"
APP_TITLE = "Send It"
SIDEBAR_CAPTION = "Move suggestions for your Kilter climbs"
SOURCE_LABEL = "Route"
SOURCE_DEMO = "Try the demo route"
SOURCE_OWN = "Use my own route"
UNITS_LABEL = "Units"
UNITS_METRIC = "Metric"
UNITS_IMPERIAL = "Imperial"

INSIGHT_BUTTON = "Deeper insight"
INSIGHT_HELP = "AI coaching on body position, built from your tracked climb."
INSIGHT_DISABLED_CAPTION = "Add XAI_API_KEY or GEMINI_API_KEY to .env to enable."
INSIGHT_TITLE = "Deeper insight"
INSIGHT_YOUR = "Your climb"
INSIGHT_SUGGESTED = "Suggested line"
INSIGHT_FOOTER = "Generated from the tracked data by an AI model. Check it against how the climb felt."
INSIGHT_UNAVAILABLE = "The AI analysis is unavailable right now. Here are the rule-based tips instead."
INSIGHT_WORKING = "Asking the coach…"
INSIGHT_NEEDS_RESULTS = "Add your climb video in step 2 first, then ask for deeper insight."
INSIGHT_CLOSE = "Close"

# ---------------------------------------------------------------- header
HEADER_READY = "{route} · analysis ready"
HEADER_ROUTE_ONLY = "{route} · add your climb video in step 2"
HEADER_NO_ROUTE = "Start with a photo of your route in step 1"

TAB_ROUTE = "1 Route"
TAB_CLIMB = "2 Climb"
TAB_EXPLORE = "3 Explore"

# ---------------------------------------------------------------- step 1: route
ROUTE_KIND_LABEL = "What are you uploading?"
ROUTE_KIND_PHOTO = "Photo of the lit board (recommended)"
ROUTE_KIND_SCREENSHOT = "Kilter app screenshot"
ROUTE_KIND_WHY = ("A photo taken from where you film lets us match it to your video automatically. "
                  "A screenshot works too, but you will mark the board's four corners.")
ROUTE_UPLOAD_LABEL = "Route image (jpg or png)"
ROUTE_DETECTED = "Found {n} lit holds. Check the colours below and fix any role with one click."
ROUTE_NONE_DETECTED = "We could not find lit holds in this image. Try a sharper photo with the lights clearly visible."
ROUTE_LOADED_SAVED = "Loaded the holds you saved for this image."
BOARD_ANGLE_LABEL = "Board angle (degrees overhang)"
BOARD_ANGLE_CAPTION = "Set to match your board."
ROUTE_NAME_LABEL = "Route name"

HOLDS_HEADER = "Holds on your route"
CLICK_MODE_LABEL = "What do you want to do?"
MODE_PICK, MODE_ADD, MODE_REMOVE, MODE_MOVE = "Pick a hold", "Add a hold", "Remove a hold", "Move a hold"
CLICK_MODES = [MODE_PICK, MODE_ADD, MODE_REMOVE, MODE_MOVE]
MODE_HINTS = {
    MODE_PICK: "Click a hold to change its role or rate it.",
    MODE_ADD: "Click where the missing hold is.",
    MODE_REMOVE: "Click a hold that is not really lit.",
    MODE_MOVE: "First click the hold you want to move.",
}
MOVE_STEP2 = "Now click where {hold} should be."
LEGEND_ROLES = "Green = start · Blue = hand · Orange = feet only · Pink = finish"
SELECTED_HEADER = "Selected hold"
SELECT_HINT = "Pick a hold to change its role or rate it."
HOLD_LINE = "**{label}** · {role}"
ROLE_LABEL = "Role (what the light colour says this hold is for)"
ROLE_NAMES = {"start": "start", None: "hand", "foot": "feet only", "finish": "finish"}
ROLE_ORDER = ["start", None, "foot", "finish"]
GRIP_LABEL = "Grip rating (1 = excellent … 5 = terrible)"
GRIP_CAPTION = "Your opinion of the hold. A worse rating steers the suggestion away from it."
REMOVE_HOLD = "Remove this hold"
ALL_HOLDS_HEADER = "All holds"
RESET_HOLDS = "Reset holds"
SAVE_ROUTE = "Save this route"
SAVED_TOAST = "Route saved. It loads like this next time."
RESET_TOAST = "Holds reset to what we detected in the image."
HOLD_COUNT = "{n} lit holds · {manual} added by you"
DETECTED_PLACEMENTS = "Detected {n} hand placements from your video."
DETECTED_NONE = "We could not see your hands on any hold. Check the holds above."
NO_VIDEO_YET = "Add your climb video in step 2 to see your hand placements here."
SHOW_SEQUENCE = "Show sequence"
SEQUENCE_COLUMNS = {"#": "#", "hand": "Hand", "hold": "Hold", "time": "Time (s)"}
KEY_FRAMES_CAPTION = "Key frames where we saw a hand land on a hold."
HOLD_TABLE_EXPANDER = "Details: hold table"
HOLD_TABLE_COLUMNS = {"label": "Hold", "grip": "Grip rating", "role": "Role"}

# ---------------------------------------------------------------- step 2: climb
VIDEO_UPLOAD_LABEL = "Your climb video (mp4 or mov)"
VIDEO_GUIDE = "Film from in front of the board, keep the whole board in view, one climber."
ANALYZE_BUTTON = "Analyze my climb"
NEED_ROUTE_FIRST = "Add your route image in step 1 first."
STAGE_NAMES = {
    "pose": "Finding your body…",
    "stabilize": "Steadying the camera…",
    "background": "Building a clean view of the board…",
    "holds": "Matching to your route…",
    "observed beta": "Reading your hand placements…",
}
STAGE_DONE = "Done in {s:.0f} s"
ANALYZE_FAIL = "We could not read this video: {err}. Use a readable video with one climber in view."
OVERLAY_FAIL = "We could not draw the skeleton video."

ALIGN_MATCHED = "Photo and video matched ({n} shared points)."
ALIGN_MATCHED_SHORT = "Photo and video matched."
ALIGN_FAILED = "Could not match. Mark the board corners."
ALIGN_SCREENSHOT = "A screenshot has nothing in common with the video, so mark the board's four corners in both images."
ALIGN_CORNERS_DONE = "Aligned from the four corners you marked."
CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"]
CORNER_INSTRUCTION = "Click the **{corner}** corner of the board in the **{which}** image ({k} of 4)."
CORNER_ROUTE = "route"
CORNER_VIDEO = "video"
CORNER_ALL_DONE = "All eight corners marked."
CORNER_RESET = "Start the corners over"
CORNER_ROUTE_TITLE = "Route image"
CORNER_VIDEO_TITLE = "Your video"

VIDEO_VIEW_LABEL = "Video"
VIDEO_ORIGINAL = "Original video"
VIDEO_SKELETON = "Skeleton overlay"
VIDEO_MISSING = "The video file is not on this computer."

BODY_EXPANDER = "Help the model know your body"
BODY_EXPLAIN = "We measure these from your video. Enter yours for more accurate reach limits."
HEIGHT_LABEL = "Height ({unit})"
SPAN_LABEL = "Arm span ({unit})"
BODY_ZERO_HINT = "Leave at 0 to use the video measurement."

CARD_SPAN = "Arm span"
CARD_LEG = "Leg length"
CARD_SPAN_SUB = "fingertip to fingertip"
CARD_FROM_VIDEO = "measured from your video"
CARD_LEG_SUB = "hip to ankle"
CARD_RELATIVE_PROMPT = "Enter your height for real-world values."

STYLE_LABEL = "Climbing style"
STYLE_BALANCED, STYLE_FEWER, STYLE_SHORTER = "Balanced", "Fewer moves", "Shorter reaches"
STYLES = [STYLE_BALANCED, STYLE_FEWER, STYLE_SHORTER]
STYLE_CAPTIONS = {
    STYLE_BALANCED: "Long reaches, bad holds and extra moves all count.",
    STYLE_FEWER: "Prefers a line with fewer moves, even if some reaches are longer.",
    STYLE_SHORTER: "Prefers shorter reaches, even if that means more moves.",
}

CARD_YOURS = "Your climb"
CARD_SUGGESTED = "Suggested"
CARD_SAVED = "Effort saved"
CARD_MOVES = "{n} moves"
CARD_MOVES_ONE = "1 move"
CARD_SAVED_SUB = "Both lines scored the same way"
CARD_SAVED_PARTIAL = "n/a"
CARD_SAVED_PARTIAL_SUB = "We only saw part of your climb"
CARD_SAME = "0 %"
CARD_SAME_SUB = "Your line already matches the suggestion"
DIFFICULTY_LINE = "Expected difficulty at {angle}°: about {mult}× the effort of the same moves on a vertical wall."
FULL_SIZE = "Full size"
PARTIAL_BANNER = ("We only tracked part of your climb (seconds {ranges} missing). "
                  "The suggested line covers the whole route; the comparison covers only the part we saw.")
PARTIAL_BANNER_CLIP = ("We only tracked part of your climb (the clip ends before the top). "
                       "The suggested line covers the whole route; the comparison covers only the part we saw.")
WHY_HEADER = "Why"
TIPS_HEADER = "Tips"
NO_LINE_FOUND = "We could not find a line from the start holds to the finish. Check the roles in step 1."
NO_OBSERVED = "We could not see a hand sequence to compare. Showing the suggested line only."
RELAXED = "We allowed a longer reach than usual to connect this route."
NEED_ANALYSIS = "Add your climb video above to see your climb next to the suggested line."

INJURY_HEADER = "Injury flags"
INJURY_DISCLAIMER = "Heuristic, not medical advice."
INJURY_NONE = "No flags on this line."
INJURY_SUGGESTED_HAS = "The suggested line has a flag too. Take it as a warning, not a rule."
INJURY_COLUMNS = {"severity": "Risk", "move": "Move", "why": "Why", "instead": "Instead"}

SAFER_EXPANDER = "Climb safer"
SAFER_TIPS = [
    "Warm up for ten minutes: easy climbs, shoulder circles, open and close your hands.",
    "On a steep board you fall onto your back. Tuck your chin, keep your arms in, land on the mat, not on your hands.",
    "Ask someone to spot you on the high moves. Their job is to guide your fall, not to catch you.",
    "Downclimb when you can. Jumping from the top loads your ankles more than the climb did.",
    "Skip a move if it hurts. There is always another day and another route.",
    "Finger pain that is sharp or in one spot is a stop signal, not something to push through.",
    "Steeper board, harder climb. Lower the angle before you try a new move at your limit.",
    "Keep your feet on the holds. Cutting loose is fun but it swings your shoulders hard.",
    "Rest between attempts. Tired fingers make the same move riskier.",
    "Check the mat is clear of shoes, chalk bags and people before you start.",
]

LIMITS_EXPANDER = "What this tool can't do"
LIMITS = [
    "Kilter-style lit boards only. Spray walls and set routes are not supported here.",
    "The board and all lit holds must be visible in both the image and the video.",
    "One climber per video.",
    "The camera should face the board and stay roughly still.",
    "LED colours vary by camera, so roles need a glance from you.",
    "Hands are modelled first; feet only approximately.",
    "Distances are 2D. Steep angles and perspective distort them.",
    "The board angle only scales expected difficulty. It does not change the suggested moves.",
    "No grade prediction.",
    "Hold ratings are your opinion, not a measurement.",
    "The injury flags are heuristics, not medical advice.",
    "Deeper insight is AI-generated from tracked data and can be wrong.",
    "The results are a suggestion for your body, not a rule.",
]

DETAILS_EXPANDER = "Details"
DETAILS_RAW_HEADER = "Raw numbers"
DETAILS_SEQUENCES = "Sequences"
DETAILS_DIFF = "Both lines on one image"
MEASURE_EXPANDER = "Optional measurement details"
APE_LABEL = "Arm span ÷ height (ape index, about 1.0 is typical)"
TRACKED_LABEL = "Frames where you were tracked: {pct} %. Gaps mean some moves may be missing."
POSE_TABLE_HEADER = "Body position per move"
POSE_TABLE_COLUMNS = {"move": "Move", "elbow": "Support-arm elbow, smallest angle (°)",
                      "hip": "Hip travel (× arm span)", "duration": "Duration (s)"}

HOW_EXPANDER = "How it works"
HOW_IT_WORKS = r"""
#### What is optimized
A **state** is the pair of holds under the climber's hands, $s=(h_L, h_R)$. An **action** moves one hand to another
lit hold. We run exact **A\*** (Dijkstra with an admissible heuristic; beam search only as a labelled fallback for the
hands-plus-feet plan) from the start state to any state with both hands on the finish hold.

#### Kilter rules in the setup
Green holds form the start state (matched if only one is lit). Pink holds are the finish, and both hands must be on
them. Orange and yellow holds are feet-only: hands never use them. Feet may use any lit hold. Unlit holds are off-route.

#### Edge feasibility (personalized)
A move is allowed only if the hand-to-hand span after the move is within the climber's **reach envelope**
(default 0.85 × their measured arm span, auto-widened to any span they demonstrated on video), the hand does not
drop more than 0.2 × arm span, and the target is a lit hand hold.

#### Movement cost (dimensionless, per move)
$$C = w_r\left(\tfrac{\text{span}/\text{arm span}}{0.4}\right)^2 + w_g\,\tfrac{\text{grip}-1}{4} + w_m + w_t\,\tfrac{\text{travel}}{\text{arm span}} + w_d\,\text{dir} + w_x\,\text{cross} + w_f\,\text{foot}$$

* **reach** — quadratic in the normalized span: the same wall distance costs more for a smaller climber.
* **grip** — the user's 1–5 hold rating (1 → 0 penalty, 5 → full penalty).
* **move** — fixed cost per hand movement, so the search does not ladder through every hold.
* **travel / direction / cross** — geometry of the moving hand: distance, sideways-or-down component, crossed hands.
* **foot** — lower-body context: no hold inside the climber's leg window below the target hold.

The observed sequence is scored with the **same** function, so "effort saved" is apples to apples. A multiplier
applied equally to every move (such as the board-angle factor) leaves the suggested line and the saved percentage
unchanged, which is why the angle is display-only.

#### Alignment
The video is stabilised with ORB features and RANSAC homographies to one reference frame. A photo of the board taken
from where you film shares features with that frame, so the same matcher aligns it automatically. A screenshot has no
shared features, so the four board corners you mark define the homography instead.

#### Scope and honesty
Costs are heuristic, not energy or force; hold ratings are user preferences; 2D pose has perspective error. Every
computer-vision stage can be corrected by hand.
"""
HOW_FACTS_HEADER = "This analysis"
HOW_FACT_FRAMES = "{n} frames at {fps:.0f} fps"
HOW_FACT_CAMERA_STATIC = "camera: static"
HOW_FACT_CAMERA_MOVING = "camera motion compensated ({px:.0f} px pan)"
HOW_FACT_TRACKED = "pose in {pct:.0%} of frames"
HOW_FACT_HOLDS = "holds from the route image ({n} lit)"
HOW_FACT_ALIGN = "alignment: {method}"
HOW_FACT_SEARCH = "search: {method}, {n:,} states expanded in {ms:.0f} ms"
HOW_FACT_ENVELOPE = "reach envelope used: {r:.2f} × arm span"
HOW_FACT_STATE = "state = (left hand, right hand); {n} lit hand holds"

ADVANCED_EXPANDER = "Advanced: tune the scoring"
ADVANCED_CAPTION = "The climbing style above sets these. Change them here if you want."
SLIDERS = {
    "w_reach": ("How much long reaches count", 0.0, 3.0, 0.1),
    "w_grip": ("How much bad holds count", 0.0, 3.0, 0.1),
    "w_move": ("How much each extra move counts", 0.0, 2.0, 0.05),
    "w_travel": ("How much hand travel counts", 0.0, 2.0, 0.05),
    "w_dir": ("How much sideways or downward moves count", 0.0, 2.0, 0.05),
    "w_cross": ("How much crossing your hands counts", 0.0, 2.0, 0.05),
    "w_foot": ("How much a missing foot hold counts", 0.0, 2.0, 0.05),
}
REACH_LIMIT_LABEL = "Longest reach allowed (share of your arm span)"
DOWN_LIMIT_LABEL = "Biggest downward hand move allowed (share of your arm span)"
MATCH_FINISH_LABEL = "Both hands must hold the finish"
AUTO_WIDENED = "We allowed reaches up to {r:.2f} × arm span because you showed that reach on video."

# ---------------------------------------------------------------- step 3: explore
EXPLORE_HEADER = "Same route, different body"
EXPLORE_CAPTION = "We shrink or stretch your measured body and re-plan the same holds."
PRESET_SHORTER = "What if I were 15 % shorter?"
PRESET_SHORTER_EXPLAIN = "Arm span, leg length and torso all scaled to 85 %. Same holds, a smaller climber."
PRESET_FEET = "Show feet too"
BACK_TO_MEASURED = "Back to my size"
SIM_SLIDER_PCT = "Body size to simulate (% of yours)"
SIM_SLIDER_UNITS = "Arm span to simulate ({unit})"
SIM_CAPTION_UNITS = "Your arm span is {span}. This simulates {sim} ({delta})."
CARD_YOU = "You"
CARD_SIM_SHORTER = "{amount} shorter"
CARD_SIM_TALLER = "{amount} taller"
CARD_DIFFERENT = "Different moves?"
CARD_DIFF_YES, CARD_DIFF_NO = "yes", "no"
CARD_DIFF_YES_SUB = "a different line suits this body"
CARD_DIFF_NO_SUB = "same line, harder for a smaller climber"
CARD_DIFF_SAME_SUB = "same line"
CARD_DIFF_HOLDS = "Holds that change"
CARD_DIFF_HOLDS_NONE = "none"
PANEL_YOU = "You"
SIM_LEGEND = "Blue = your line · Pink = the simulated climber's line"
SWEEP_EXPANDER = "Details: how the plan changes with body size"
SWEEP_BUTTON = "Compute for 70 % to 120 %"
SWEEP_COLUMNS = {"pct": "Body size (%)", "moves": "Moves", "holds": "Holds"}
SIM_FAIL = "We could not plan a line for this body size on this route."

FEET_TOGGLE = "Show feet too"
FEET_WAIT_FAST = "Ready in about a second."
FEET_WAIT_SLOW = "Planning feet too can take about {s} seconds after an edit."
FEET_SPINNER = "Planning your feet too (about {s} s)…"
FEET_NONE = "We could not plan a line with feet on this route."
CARD_WITH_FEET = "With feet"
CARD_WITH_FEET_SUB = "{h} hand moves + {f} foot moves"
CARD_FEET_ON = "Feet go on"
CARD_FEET_NONE = "no holds (feet on the mat or cut loose)"
FEET_LEGEND = "Mint squares = foot holds, with the step numbers when a foot is on them."
FEET_DETAILS = "Details: full sequence with feet"
ADVANCED_FEET = "Advanced"
BEAM_LABEL = "Faster, approximate planning when the exact plan is slow"


def moves_text(n: int) -> str:
    return CARD_MOVES_ONE if n == 1 else CARD_MOVES.format(n=n)


def role_name(role) -> str:
    return ROLE_NAMES.get(role, "hand")
