# Judge Q&A — Send It

**What exactly are you optimizing?**
The sequence of hand placements on a route. State = (hold under left hand, hold under right hand); an action moves
one hand. We minimize the summed movement cost from the observed start state to a state with a hand on the finish.

**What is the objective function?**
Per move: `w_r·(span/arm_span/0.4)² + w_g·(grip−1)/4 + w_m + w_t·travel/arm_span + w_d·(sideways/down) + w_x·crossed
+ w_f·foot-support deficit`. Reach is quadratic in the span normalized by *this* climber's arm span; grip is the
user's 1–5 rating; the per-move term stops the optimizer from laddering through every hold; the last three are small
geometric context terms. Everything is dimensionless and shown as sliders.

**What makes an edge feasible?**
The hand-to-hand span after the move must be ≤ ρ_max × the climber's measured arm span (default 0.85, auto-widened to
any span the climber demonstrated on video), the hand may not drop more than 0.2 arm spans, and the target must be on
route and not foot-only. If the graph is disconnected we relax ρ_max in 0.05 steps and say so.

**Why graph search?**
Once holds are nodes and hand moves are edges with personalized costs, "best beta" is literally a shortest path. With
non-negative costs Dijkstra is exact and runs in milliseconds, so every slider, rating and hold edit re-optimizes live.

**Why not the shortest geometric path?**
It ignores the climber (no normalization), the holds (no grip), and re-grips (no per-move cost). The UI scores the
geometric path under our objective so you can see it lose.

**How does grip quality influence the optimizer?**
Entering a hold rated `g` adds `w_g·(g−1)/4`. Rating a hold 5 makes a nearby alternative cheaper; in the demo, rating
one hold terrible visibly re-routes the beta. It's a subjective input, treated as a preference, not ground truth.

**How does morphology change the result?**
Every reach term is a ratio to arm span, so the same wall distance costs more for a smaller climber, and edges beyond
their envelope vanish. In the spray-wall demo, scaling to 85 % reach changes the recommended sequence (extra move via
H20/H29) and raises cost 9.8 → 11.6. Leg length + torso set the foot-support window.

**What happens if CV detects the wrong hold?**
You fix it: click to add/remove/move, toggle route membership, mark start/finish/foot-only, and re-derive the observed
sequence. Missed holds are also proposed automatically from where the climber's hands dwelled (that's how the dark
start holds were recovered). Detection → correction → optimization is the intended workflow.

**Why allow manual correction?**
Because uncontrolled gym footage breaks any detector sometimes (dark holds on dark panels, paint, glare). A tool that
fails silently is useless; a tool that fails visibly and is fixable in three clicks is a product.

**Are the biomechanics validated?**
No. The cost is a biomechanics-*informed heuristic*: quadratic normalized reach, grip preference, per-move cost,
geometric context. We say "movement cost under our objective", never "energy", "force" or "fatigue".

**Are you measuring energy?**
No. All quantities are pixel ratios and user ratings. Measured: landmarks, segment lengths, hold positions, contact
timing. Computed: costs, paths. Subjective: grip. Simulated: the scaled morphology.

**How do feet factor in?**
Through a foot-support proxy: a target hand hold is penalized if no hold lies inside the climber's leg window below
it (0.35–1.25 × torso+leg length vertically, ±0.6 laterally). Foot-only holds can be marked and count for support
but not for hands.

**Why only optimize hands?**
Full beta is a high-dimensional state space: both hands, both feet, body configuration. In six hours we reduced the
state to the hand pair, which is a tractable exact search that still captures morphology, grip and pose context. The
same machinery extends to `(h_L, h_R, f_L, f_R)`; that's the next step, with learned edge costs after it.

**What is the next technical step?**
Four-limb state, support-polygon feasibility, learned costs from many successful ascents (imitation / inverse RL),
3D pose for metric distances.

**How is this different from other climbing tools?**
They analyze what happened (pose overlays, smoothness scores, hold hit rates). We turn the video into a personalized
optimization problem and *recommend* a sequence for that body. Closest prior work, BetaMove, predicts MoonBoard hand
sequences with beam search but has no body-size normalization, no video, no correction loop.

**What was technically hardest?**
Getting handheld footage into one coordinate frame (per-frame ORB+RANSAC homographies, then a median panorama that
removes the climber) so holds, hands and the graph agree; and designing a cost function whose optimum looks like beta a
human would actually climb (quadratic reach + per-move cost was the key balance).

**What does AI actually do here?**
MediaPipe pose is the sensor. The optimization is classical graph search on a model we built. An LLM, if a key is
present, only paraphrases the structured result — it never picks the beta, and the demo runs without it.

**Could ChatGPT produce the same recommendation?**
Not from a video: it has no measured arm span, no hold coordinates, no contact timing, and no feasibility graph. Given
our structured output it can *describe* the result, which is exactly the optional role we give it.
