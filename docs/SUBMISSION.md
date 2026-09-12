# Send It — Personalized Climbing Beta Optimizer (HackCMU 2026, Optimization track)

**One-liner.** Turn a phone video of a climb into a personalized optimization problem: measured body proportions +
detected holds + the sequence you actually climbed → a climber-specific hold graph with a movement-cost function →
exact Dijkstra search → the minimum-cost beta for *your* body, compared move by move with what you did.

**What it does.** MediaPipe pose gives arm span, leg and torso lengths (calibration-free ratios). Handheld footage is
registered into one wall coordinate frame with ORB+RANSAC homographies, producing a climber-free panorama for hold
detection. Holds come from LED/colour detection plus inference from where the hands dwelled, and a click-to-edit
editor with 1–5 grip ratings closes the loop. The optimizer's state is the pair of hand holds; edges are single-hand
moves feasible under the climber's reach envelope; costs combine normalized reach (quadratic), grip rating, a
per-move penalty and geometric context (direction, crossed hands, foot support). The observed sequence is scored with
the same objective, so the improvement, the predicted crux and the cost breakdown are like-for-like. A morphology
slider re-plans the route for a shorter or taller climber.

**Result on our footage.** 11 observed hand moves (cost 21.2, peak reach 82 % of arm span) vs 5 optimized moves
(cost 9.8, peak 46 %); at 85 % simulated reach the recommended beta changes (6 moves via different holds).

**Scope.** Hand-sequence optimization with feet/pose as cost modifiers; four-limb state search and learned costs are
the next steps. Costs are heuristics, not physiology.

**Stack.** Python, MediaPipe, OpenCV, NumPy, Streamlit, Plotly. Repo: https://github.com/ahaan001/Send-it-climbing-CV
