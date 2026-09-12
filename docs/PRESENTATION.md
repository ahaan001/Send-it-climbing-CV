# Send It — 3-minute judging script (Optimization track)

**Setup before judges arrive:** `streamlit run app.py` open in a browser on the *Spray wall* demo, tab **3 · Optimize**
visible. Second browser tab on **4 · Personalize**. Fallback images open in Preview: `demo_assets/spraywall/comparison.png`,
`personalize.png`.

---

**0:00–0:20 — Problem**
"Climbing routes have one grade, but climbers don't have one body. The same move is a comfortable lock-off for a tall
climber and a full-extension dyno for a short one, and the best sequence of holds — the *beta* — depends on the body
doing it. Every climbing app today analyzes what happened. Send It recommends what *should* happen for *your* body."

**0:20–0:40 — Input** *(tab 2 · Climber: play the overlay video)*
"You film a climb on your phone. Computer vision gives us three things: the climber's body proportions — arm span, legs,
torso, measured from the video itself — the holds on the wall, and the sequence of holds they actually used. The
camera was handheld, so we register every frame to one wall coordinate frame first; the hold circles stick to the
wall while the camera pans."

**0:40–1:00 — Human in the loop** *(tab 1 · Route & holds)*
"CV is imperfect in a real gym: these two dark start holds were missed by colour detection, and recovered from where
the climber's hands actually stopped. Anything that's still wrong you fix with a click — add, remove, move a hold —
and you rate how good each hold feels, 1 to 5. That rating goes straight into the objective." *(rate one hold a 5,
show the optimizer re-routing)*

**1:00–1:35 — The optimization** *(tab 3, open the feasibility graph expander briefly)*
"Here is the core. A state is the pair of holds under the two hands. A move relocates one hand. A move is *feasible*
only if the resulting hand-to-hand span fits inside this climber's reach envelope — a fraction of their own measured arm
span. Every feasible move gets a cost: reach normalized to their body, squared; the hold rating; a fixed cost per
re-grip; plus small geometric terms for sideways moves, crossed hands, and missing foot support. Then we run exact
Dijkstra on that graph for the minimum-cost beta. The climber's own sequence is scored with the identical function,
so the comparison is apples to apples."

**1:35–2:05 — Hero result** *(tab 3 comparison image + metric cards)*
"Left: what she actually did — eleven hand moves, and the highest-cost move in red: an 82-percent-of-arm-span reach
off the start. Right: the optimizer's line for her body — five moves, no reach over 46 percent, 54 percent less total
movement cost. The bar chart shows where the cost went: that first reach dominates."

**2:05–2:30 — Personalization** *(tab 4, slide to 85 %)*
"Same route, same holds, same objective — now shrink the climber's effective reach to 85 percent. Feasible edges
disappear, costs go up, and the optimizer recommends a *different* beta: it routes through holds 20 and 29 with an
extra move. That's the point: beta is a property of the climber–route pair, not the route."

**2:30–2:45 — Scope**
"For the hackathon we deliberately optimized the major hand sequence. Feet and pose only modify costs. The natural next
step is a four-limb state space — same graph machinery, bigger state — and learning the edge costs from real climbs."

**2:45–3:00 — Close**
"Grades describe the route. Send It optimizes the route for the person climbing it. Upload a video, fix what the CV
got wrong, and get the beta for *your* body."

---

### If the live app breaks
Switch to the PNGs in `demo_assets/spraywall/` (`comparison.png`, `diff.png`, `graph.png`, `personalize.png`) and
`demo_assets/kilter/`. They were rendered by the same pipeline (`scripts/build_demo_cache.py`); say so.

### Numbers to have in your head (spray-wall demo, default weights)
| | observed | optimized | simulated 85 % |
|---|---|---|---|
| hand moves | 11 | 5 | 6 |
| total cost | 21.2 | 9.8 | 11.6 |
| max reach (× arm span) | 0.82 | 0.46 | 0.46 |
| holds | 61 62 12 20 25 29 21 32 35 45 63 | 61 62 16 25 32 37 45 | 61 62 16 20 29 32 37 45 |

Pose ≈ 9 s, stabilization ≈ 8 s for 761 frames; optimization < 50 ms.
