# Send It — 3-minute judging script (Optimization track)

**Setup before judges arrive:** `streamlit run app.py`, open `http://localhost:8501/?present=1` (presentation mode:
hero-first, big type, no chrome), *Spray wall* demo, tab **3 · Optimize**. The whole talk runs on that tab with the
**Demo moves** buttons: *Rate H25 terrible* → *Reset ratings* → *Simulate 85 % climber* → *Back to measured* →
*Show full-body plan*. Fallback images open in Preview: `demo_assets/spraywall/comparison.png`, `personalize.png`,
`fourlimb.png`.

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
and you rate how good each hold feels, 1 to 5. That rating goes straight into the objective." *(select **H25** —
the second hold on the cyan line — set grip to 5; switch to tab 3: the optimizer now goes through H20 and H29 instead.
Reset H25 to 3 afterwards. Rating **H16** a 5 re-routes through H12/H18 if you want a second example.)*

**1:00–1:35 — The optimization** *(tab 3; the "Optimization at a glance" strip is on screen)*

*Version A — hands-first (default):*
"Here is the core. A state is the pair of holds under the two hands. A move relocates one hand. A move is *feasible*
only if the resulting hand-to-hand span fits inside this climber's reach envelope — a fraction of their own measured arm
span. Every feasible move gets a cost: reach normalized to their body, squared; the hold rating; a fixed cost per
re-grip; plus small geometric terms for sideways moves, crossed hands, and missing foot support. Then we run exact
A* search — Dijkstra with an admissible bound — for the minimum-cost beta. The climber's own sequence is scored with
the identical function, so the comparison is apples to apples."

*Version B — four-limb lead (use if the 14:00 check passed):*
"Here is the core. A state is where all four limbs are: left hand, right hand, left foot, right foot. A move
relocates one limb — a foot may even cut loose, at a price. A move is feasible only if the hand-to-hand span fits this
climber's measured reach envelope and every foot sits inside their leg window below the hands. Costs are
dimensionless: reach normalized to their arm span, squared; hold rating; a fixed cost per re-grip; foot travel over
leg length; a campus penalty for moving a hand with no feet. On this wall that is about forty thousand joint states;
we search them exactly with A*, and fall back to a labelled beam search when a wall gets denser. The climber's own
sequence is scored with the identical function." *(press "Show full-body plan")*

**1:35–2:05 — Hero result** *(tab 3 comparison image + metric cards)*
"Left: what she actually did — eleven hand moves, and the highest-cost move in red: an 82-percent-of-arm-span reach
off the start. Right: the optimizer's line for her body — five moves, no reach over 46 percent, 54 percent less total
movement cost. The bar chart shows where the cost went: that first reach dominates."

**2:05–2:30 — Personalization** *(press "Simulate 85 % climber" on tab 3)*
"Same route, same holds, same objective — now shrink the climber's effective reach to 85 percent. Feasible edges
disappear, costs go up, and the optimizer recommends a *different* beta: it routes through holds 20 and 29 with an
extra move. That's the point: beta is a property of the climber–route pair, not the route. On two hundred synthetic routes,
shrinking the climber by fifteen percent changes the optimal hold set on seventy percent of them."

**2:30–2:45 — Scope**

*Version A:* "For the headline numbers we optimize the major hand sequence — and the same graph already runs over all
four limbs *(press "Show full-body plan")*: feet inside a leg window, a campus penalty for cutting loose, exact A* on
forty thousand joint states. Next: hips as state, and learning the edge costs from real climbs."

*Version B:* "What we do not model yet: the hip as an explicit state, and learned costs — today the weights are
hand-tuned and every one of them is a slider. Both are the natural next step on this graph."

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

Pose ≈ 9 s, stabilization ≈ 8 s for 761 frames; hands-only optimization < 400 ms; four-limb exact A* ≈ 14 s
(precomputed for the demos), beam 80 ≈ 1 s (≈ 4 % worse on the spray wall).
Four-limb plan (measured): cost 11.8, 5 hand + 2 foot moves, feet on H20. Simulated 85 %: cost 14.8, 6 hand + 4 foot
moves, feet on H12/H18/H24/H25.
