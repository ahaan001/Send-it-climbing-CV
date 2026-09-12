# UI audit — Send It before the Kilter-only rebuild

Audited on 2026-09-12 against `app.py` (765 lines), `sendit/coach.py`, `sendit/viz.py`,
`demo_assets/demos.json`, `scripts/ui_screenshots.py` and `scripts/record_walkthrough.py`.
Every row was checked against the code (line numbers are `app.py` unless noted).

**Decision key.** *keep* = stays as is (maybe with plainer copy). *rename* = same function, new label or
wording. *move* = same function, new place (step, expander or sidebar). *remove* = gone from the UI; the
package code behind it stays unless the row says otherwise.

**Target structure the decisions map onto.** Sidebar: source (demo / my route), units, Deeper insight.
Header: route name + "analysis ready", Deeper-insight panel. Steps: **1 Route** (image, holds, roles,
board angle) · **2 Climb** (video, alignment, results, injury flags, Climb safer, What this tool can't do,
How it works, Advanced scoring) · **3 Explore** (body-size simulation, feet plan).

---

## 1. Global and sidebar

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Page config (L45) title "Send It · Beta Optimizer"; `?present=1` collapses the sidebar | Sets the browser tab title; reads the `present` query param | "Beta Optimizer" is jargon; the query-param branch is presentation mode | **rename** title to "Send It"; **remove** the `?present` branch |
| Base CSS block (L60) | Defines metric-card, pill, tagline, hero-title, obs/opt/crux colours | Pill, tagline and hero classes will have no users | **keep** (no new CSS); delete the classes whose elements are removed |
| `PRESENT_CSS` (L48) and its injection (L357-359) | Hides Streamlit chrome and enlarges type when presentation mode is on | Presentation mode is being removed | **remove** |
| Sidebar heading "🧗 Send It" (L308) | Static heading | none | **keep** |
| Caption "Personalized climbing beta optimizer" (L309) | Static caption | "beta", "optimizer" | **rename** to a plain one-liner ("Move suggestions for your Kilter climbs") |
| Toggle "Presentation mode (projector)" (L310) + tooltip | Sets `ss.present`; changes layout of tab 3, hides tagline and pills, injects CSS | Presentation mode removed; tooltip advertises `?present=1` | **remove** |
| Radio "Source: Demo climb / Upload video" (L311) | Local variable; swaps the widgets below it; loads nothing by itself | Switching has no visible effect in the main area; "Source" is a developer word | **rename** to "Try the demo route / Use my own route"; picking the demo loads it, picking "my own" opens step 1 upload |
| Selectbox "Choose a demo" (L314) | Picks one of two demos; updates the blurb but does not load until Load is pressed | Sidebar describes demo B while demo A is still shown; only one demo will remain | **remove** |
| Button "Load" (L317) | `set_source(demo)`; also auto-runs on the first render. Re-clicking with the same demo reloads it and silently discards unsaved hold edits, ratings, selection and finish choice | No-op when the demo is already loaded (which is always, since it auto-loads); hidden destructive side effect | **remove** (demo loads when the source radio says demo; "Reset holds" in step 1 covers the reset case) |
| Button "Re-run live" (L319) + tooltip | Forces the full pipeline on the demo video, rewrites the demo cache on disk, progress bar in the sidebar | Developer tool; jargon tooltip; visible result identical to Load except one header pill | **remove** from the UI (`scripts/build_demo_cache.py --force` is the maintained path) |
| Demo blurb caption (L321) | Shows the *selected* demo's blurb | "homographies", "optimizer", "measured climber"; can describe a demo that is not loaded | **rename**: one plain sentence about the Kilter demo, shown in the header line |
| File uploader "Climbing video (mp4/mov/avi)" (L323) | Holds the bytes; only shown in the Upload branch | Lives in the sidebar; no guidance on what footage works; label omits m4v | **move** to step 2 as "Your climb video" with one line of guidance |
| Selectbox "Wall type" (L324) | Passed to the pipeline only when Analyze is clicked | Kilter-only now; changing it after analysis does nothing | **remove** |
| Button "Analyze" (L325) | Saves the upload, runs the pipeline with a progress bar in the sidebar, resets all state | Progress, "Done in Ns" and errors render in the narrow sidebar; no success message | **move** to step 2 as "Analyze my climb"; feedback in the main area |
| Progress bar and stage text (L114) | "pose … 55 %", "stabilize …", "observed beta …" | Stage names are pipeline jargon | **move** with Analyze; **rename** stages ("Finding your body…", "Steadying the camera…", "Matching to your route…") |
| Error "Could not analyze this video…" (L127) | `st.error` + `st.stop()` | Fine, but in the sidebar | **move** to step 2 |
| Warning "Overlay video not rendered" (L136) | Shown if the skeleton video fails | "overlay" | **rename** ("We could not draw the skeleton video") |
| Divider + heading "Optimizer settings" (L332-333) | Static | "Optimizer" | **remove** |
| Slider "Reach envelope (max hand-to-hand span, × arm span)" (L334) + tooltip | `Feasibility.max_reach_frac`; auto-widened to any reach seen on video | "envelope", "edge", "feasible"; an end user never needs it | **move** to "Advanced: tune the scoring" (bottom of step 2) as "Longest reach allowed, as a share of your arm span" |
| Slider "Reach weight" (L336) + formula tooltip | `Weights.w_reach` | Raw formula in a tooltip | **move** to Advanced as "How much long reaches count" |
| Slider "Grip-quality weight" (L337) + formula tooltip | `Weights.w_grip` | Formula | **move** to Advanced as "How much bad holds count" |
| Slider "Per-move penalty" (L338) | `Weights.w_move` | "cost" in tooltip | **move** to Advanced as "How much each extra move counts" |
| Expander "Advanced terms" (L339) | Holds five sliders and a checkbox | Two levels of settings in the sidebar | **remove** (merged into the single Advanced expander) |
| Slider "Travel weight" (L340) | `Weights.w_travel` | no meaning given | **move** to Advanced as "How much hand travel counts" |
| Slider "Sideways / downward weight" (L341) | `Weights.w_dir` | no meaning given | **move** to Advanced as "How much sideways or downward moves count" |
| Slider "Crossed-hands weight" (L342) | `Weights.w_cross` | no meaning given | **move** to Advanced as "How much crossing your hands counts" |
| Slider "Foot-support weight" (L343) | `Weights.w_foot` | no meaning given | **move** to Advanced as "How much a missing foot hold counts" |
| Slider "Max downward move (× arm span)" (L344) | `Feasibility.max_down_frac` | no meaning given | **move** to Advanced as "Biggest downward hand move allowed" |
| Checkbox "Require both hands on finish" (L345) | `Feasibility.match_finish`, default off | Kilter rule says both hands finish, so the default is wrong | **keep**, default **on**; **move** to Advanced as "Both hands must hold the finish" |
| *(new)* Units toggle, "Deeper insight" button, "Climbing style" preset selector | — | — | added per spec (D18, G31, F30) |

## 2. Header, pre-load state and tab bar

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Info "Load a demo climb or upload a video from the sidebar." (L350) | Guard when nothing is loaded | Dead code: the first demo auto-loads on the first render, so this never shows | **remove**; replace with step-1 guidance when no route is loaded |
| Hero title "Send It · Personalized Climbing Beta Optimizer" (L385) | Static | "Beta Optimizer" | **rename** to "Send It" |
| Tagline "Video → pose → holds → your movement-cost graph → minimum-cost beta…" (L387) | Static; hidden in presentation mode | Pipeline jargon | **remove**; replaced by one line "<route name> · analysis ready" |
| Pills row (L389-395): cached/live, frames @ fps, pose time, camera static / "N px pan", holds method + human-corrected | Five facts about the run; hidden in presentation mode | Pixels, pipeline stats, no meaning for a climber | **remove**; the facts go into "How it works" |
| Tabs "1 · Route & holds / 2 · Climber / 3 · Optimize / 4 · Personalize / Method" (L397) | Five tabs | Too many; "Optimize", "Personalize", "Method" are not climber words | **rename** to "1 Route / 2 Climb / 3 Explore"; Method becomes a collapsed "How it works" at the bottom of step 2 |
| *(new)* Deeper-insight output container with Close button | — | — | added between the header and the tabs (G31) |

## 3. Tab 1 · Route & holds

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Header "Detected holds (click to edit)" (L404) | Static | fine | **rename** to "Holds on your route" (holds now come from the route image) |
| Radio "Click mode: select / add / remove / move selected" (L405) | Sets `ss.edit_mode`; nouns; "move selected" needs a prior selection and gives no instruction | Not verbs; the two-step move has no on-screen guidance | **rename** to "Pick a hold / Add a hold / Remove a hold / Move a hold", with an instruction line for the two-step move |
| Hold canvas (`streamlit_image_coordinates`, L414) | Click maps to the nearest hold within 8 % of arm span; add / remove / move edit `ss.holds`, call `refresh_observed()` and rerun. Drawn on the video background; hold ids drawn bare ("25") while text says "H25" | Background is a video frame, not the route image; role shown as rings; id labels inconsistent with the text | **keep**; draw on the route image with role colours and "H"-style ids everywhere |
| Legend caption "Green ring = start · yellow ring = finish · fill = grip rating … grey = off-route" (L439) | Static | Describes the old colour scheme | **rename** to "Green = start · Blue = hand · Orange = feet only · Pink = finish" |
| Header "Selected hold" (L441) | Static | fine | **keep** |
| Info "Click a hold in *select* mode to rate it, mark it start/finish/foot-only, or toggle route membership." (L443) | Shown when nothing is selected | "route membership" | **rename** ("Pick a hold to change its role or rate it") |
| Hold info line "Hold N · source: led · (x, y)" (L446) | Static per selected hold | Pixels and detector source | **rename** to "Hold H7 · start hold" |
| Select-slider "Grip quality (1 = excellent … 5 = terrible)" (L447) | Edits the hold's 1–5 rating; recomputes; reruns | "quality" reads as a property of the hold, but this is your own opinion of it | **rename** to "Grip rating (1 = excellent … 5 = terrible)" |
| Selectbox "Role: regular hand / start / finish / foot-only" (L449) | Edits what the hold is for: start = where both hands begin, finish = the top hold you must hold, foot-only = feet may use it but hands may not, regular = any hand hold. Recomputes | Options do not match the Kilter light rules; nothing explains what a role is; no preset from the LED colour | **rename** to "Role (what the light colour says this hold is for)" with options "start / hand / feet only / finish", preset from the LED colour and fixable in one click |
| Toggle "On route (hands may use it)" (L451) | Sets `on_route`; the colour filter writes the same field | Two controls for one field; every detected hold on a Kilter image is lit, so membership is implicit | **remove** ("Remove this hold" covers a wrong detection) |
| Button "Delete this hold" (L457) | Removes the hold; recomputes; reruns | fine | **rename** to "Remove this hold" |
| Header "Hold set" (L462) | Static | fine | **rename** to "All holds" |
| Multiselect "Route colour filter" (L466) + tooltip | Sets `on_route` by colour for colour-detected holds; clearing every colour cannot stick (it resets to all colours on the next rerun) | Belongs to the colour detector; broken edge case; "Inferred" in tooltip | **remove** |
| Button "Reset to auto-detected" (L475) | Restores the pipeline's holds; recomputes | No confirmation toast | **keep** as "Reset holds" with a toast |
| Button "Save as curated" (L480) + success message (L484) | Writes `holds_edited.json`; sets `ss.curated`; shows a one-rerun success line | "curated" is jargon; message speaks of "this video" | **rename** to "Save this route" with a toast |
| Button "Re-detect sequence" (L485) | Calls `refresh_observed()` and reruns | No-op: every edit already refreshes, and the sequence is not shown on this tab | **remove**; replaced by the always-visible line "Detected N hand placements from your video" plus a "Show sequence" expander |
| Caption "N holds · n on route · n inferred from where the climber's hands dwelled · n added manually" (L489) | Counts by source | "dwell", "inferred" | **rename** to "N lit holds · n added by you" |
| Hold table (L493) | Columns id / grip / role / on_route / source | Table in the main flow; "grip" is a bare number, "source" is the detector name, "on_route" is jargon | **move** into a collapsed "Details: hold table" expander with columns "Hold · Grip rating · Role"; drop on_route and source |
| *(new)* Route-image uploader with photo / screenshot choice, board-angle input, role warnings, alignment status, corner-marking step | — | — | added per spec (A2, A3, A5, A6, C15) |

## 4. Tab 2 · Climber

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Card "Arm span (measured)" NNN px (L500) | 92nd-percentile fingertip span from pose | Pixels; "robust 92nd-percentile" | **keep**; **move** to the top of step-2 results, in real units when a scale exists, else relative with "Enter your height for real-world values" |
| Card "Leg length" NNN px (L502) | Thigh + shin from pose | Pixels | **keep** (same treatment) |
| Card "Span / height proxy" (L504) | Arm span ÷ height proxy | "proxy" | **move** to "Optional measurement details" as "Arm span ÷ height (ape index, about 1.0 is typical)" |
| Card "Pose detection" NN % (L506) | Share of frames with a skeleton | No meaning given | **move** to "Optional measurement details" as "Frames where you were tracked: N %. Gaps mean some moves may be missing." |
| Caption "All lengths are in this video's pixel space…" (L507) | Static | Pixels; "calibrated" | **remove** |
| Header "Pose tracking" + skeleton video at half width (L511-514) | Plays `overlay.mp4` | Half width; no way to see the original clip | **move** to the top of the results, full width, with an "Original video / Skeleton overlay" toggle |
| Info "Overlay video not rendered for this source." (L516) | Fallback | "overlay", "source" | **rename** |
| Header "Observed hand sequence (from wrist/index contact dwell)" (L518) | Static | "observed", "dwell" | **rename**; becomes the "Show sequence" expander in step 1 |
| Sequence table # / hand / hold / frame / time (L524) | One row per placement | Table in the main flow; frame numbers | **move** into "Show sequence"; drop the frame column |
| Warning "No hand contacts found… lower the reach envelope." (L526) | Shown when no placements | The advice is wrong (the reach setting does not affect contact detection); jargon | **rename** to "We could not see your hands on any hold. Check the holds in step 1." |
| Contact-sheet image "Key frames at auto-detected placements" (L536) | Five key frames, sampled with a stride (on a 13-move clip it skips the last four moves) | Jargon caption; silently skips moves | **move** into "Show sequence"; caption reworded; sample the first N rather than a stride |
| Expander "Measured pose context per observed move (diagnostic, not in the objective)" + table (L539-540) | Support-arm elbow minimum, hip travel, duration | "objective", "observed"; the data is what Deeper insight and injury rules need | **keep** collapsed; **rename** to "Body position per move"; **move** under "Optional measurement details" |
| *(new)* "Help the model know your body" (height, arm span), partial-tracking banner, injury flags, Climb safer, What this tool can't do, How it works | — | — | added per spec (D19, E29, H36-38, B9) |

## 5. Tab 3 · Optimize

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Preset row label "Demo moves" (L237, rendered on tabs 3 and 4) | Static label | Not needed with two presets | **remove** |
| Preset "Rate H25 terrible" (L239, spray wall only) | Sets hold 25 grip to 5; the hold turns red in every panel and the line re-plans | Spray-wall demo is being hidden | **remove** |
| Preset "Reset ratings" (L239, spray wall only) | Restores baseline grips | No-op when no rating changed; spray wall only | **remove** |
| Preset "Simulate 85 % climber" (L239, both demos) | Sets `scale_pct=85`, `sim_on_opt=True`; shows the simulated strip on tab 3 | Effect appears on tab 3 as a strip and on tab 4 as a slider value; label unclear | **rename** to "What if I were 15 % shorter?"; **move** to step 3 with one line saying what is scaled |
| Preset "Back to measured" (L239, both demos) | Sets `sim_on_opt=False`: hides the strip on tab 3; nothing visible on tab 4 | No-op on tab 4 and whenever the strip is already hidden | **rename** to a slider reset ("Back to my size"), shown only when the slider is not at 100 % |
| Preset "Show full-body plan" (L239, both demos) | Sets `show_feet=True`; the toggle it flips sits below the fold | Effect is out of view; no wait warning | **rename** to "Show feet too"; **move** to step 3 with a wait warning and spinner |
| Selectbox "Finish hold" (observed / top / each hold) (L551) | Sets `ss.finish_choice`; the demo defaults to "top" | Kilter rule: finish = the pink holds; a second place to define the finish | **remove** (finish comes from the pink holds; step-1 role editor overrides) |
| Optimizer error + `st.stop()` (L555) | Halts the whole page; tabs 4 and Method render empty | Blanks other tabs with no explanation | **rename** to plain copy; no `st.stop()` |
| Error "No feasible beta found even after relaxing the reach envelope…" (L559) | Shown when no line exists | "feasible", "beta", "envelope" | **rename** to "We could not find a line from the start holds to the finish. Check the roles in step 1." |
| Card "Observed cost" (L572) | Cost, moves, max reach % | "cost", "observed", a percentage with no meaning | **rename** to "Your climb: N moves" (raw cost into details) |
| Card "Optimized cost" (L574) | Same for the suggestion | Same | **rename** to "Suggested: N moves" |
| Card "Cost reduction" (L577 partial branch / L579) | "n/a" when the clip is partial, else a signed % with the formula "(observed − optimized) ÷ observed, same objective" | Three banned words in one sub-line | **rename** to "Effort saved: X %" with "Both lines scored the same way" |
| Card "Highest-cost observed move" (L582) | Crux cost and hand/holds | "cost"; duplicated by the red arrow | **remove** (the "Hardest" arrow and the Why block carry it) |
| Warning "No observed hand sequence to compare against; showing the optimized beta only." (L585) | Fallback | Jargon | **rename** |
| Caption "Reach envelope auto-widened to X × arm span…" (L587) | Explains the widened limit | Jargon | **move** into details |
| Warning "Graph was disconnected under the envelope; loosened to X…" (L589) | Relaxation notice | "graph", "envelope" | **rename** to "We allowed a longer reach than usual to connect the route." |
| Comparison image (L592, `viz.render_comparison`) | Two panels with title bars "OBSERVED beta (from video) · cost … | N hand moves | max reach N %" and "OPTIMIZED beta (min-cost search)"; numbered arrows; crux labelled "highest-cost move"; cropped to the holds' box, taller than the viewport | Jargon and numbers drawn into the image; too tall | **keep**; simplified to "Your climb" / "Suggested" with "N moves", "Hardest", short ids, a height cap and a "Full size" expander |
| Simulated-strip header "Same route, simulated 85 % reach — different beta … cost X → Y" (L601) | Appears after the preset; says "higher cost" without checking the cost | Jargon; unchecked claim; duplicates tab 4 | **move** to step 3 (as its side-by-side header); **rename** |
| Simulated-strip images MEASURED / SIMULATED (L605-606) | Two panels with cost subtitles inside the image | Duplicates tab 4 | **remove** from step 2 (step 3 shows the same pair) |
| Toggle "Full-body plan (hands + feet)" (L610) | Runs the four-limb search (instant if precomputed, 15-20 s live) | Below the fold; no wait warning | **rename** to "Show feet too"; **move** to step 3 with a warning and a spinner showing the estimate |
| Checkbox "Faster approximate search (beam) when not precomputed" (L611) + tooltip | Sends beam settings to the four-limb search | "beam", "A*" | **move** to an Advanced expander in step 3 |
| Spinner "Searching the hands + feet state space…" (L615) | Wait indicator | "state space" | **rename** to "Planning your feet too (about N s)…" |
| Error "No full-body plan found." (L620) | Fallback | fine | **rename** to plain copy |
| Card "Full-body plan cost" (L626) | Cost plus hand/foot move counts | "cost" | **rename** to "With feet: N hand moves + N foot moves" |
| Card "Search" (L628) | exact / beam, states expanded, runtime, estimated joint states | Search statistics | **remove**; facts go to "How it works" |
| Card "Feet used" (L630) | Foot holds along the plan | fine | **keep** as "Feet go on: …" |
| Card "Hands-only plan" (L632) | Hands-only cost "not directly comparable" | "cost"; admits it is not comparable | **remove** |
| Full-body comparison image (L634) | Observed feet vs planned feet | Same issues as the main comparison | **keep** in step 3 with the simplified renderer |
| Caption "State = (left hand, right hand, left foot, right foot)… leg window 0.35–1.25 × body-height proxy…" (L636) | Explains the four-limb model | "state", "proxy" | **move** to "How it works" |
| Line "Full-body plan: start L:… R:… · L→H3 (46 %)…" (L640) | Sequence text with reach % | Percentages | **move** into a collapsed details expander in step 3 |
| Expander "Full-body plan cost breakdown" + chart (L641-642) | Stacked per-move cost bars | "cost" | **remove** |
| Header "Optimization at a glance" + glance strip cards State / Feasible move / Objective / Solver (L644-645, L367-381) | Search summary | Every card uses banned words | **remove**; content moves to "How it works" |
| Details expander wrapper (presentation mode only, L646) | Collapses everything below in present mode | Presentation mode | **remove** wrapper |
| Line "Predicted crux (highest-cost move under our model): …" (L649) | Crux explanation | "cost", "model" | **rename** into the "Why" block |
| Info "The observed sequence stops before the selected finish hold… choose 'highest hold the climber reached'…" (L651) | Partial-clip notice | Refers to the removed finish selector | **rename** to the partial banner with untracked time ranges |
| Success "The observed beta already matches the optimizer's minimum-cost sequence…" (L653), Info "Same holds… different hand order" (L655), "Difference: the optimizer drops … and adds …" (L659) | Comparison verdicts | "beta", "optimizer", "minimum-cost" | **rename** into the three-sentence "Why" block |
| Header "Sequences" + Observed / Optimized sequence lines (L661-663) | Hold ids with reach % per move | Percentages; jargon | **move** into a collapsed "Details" expander |
| Header "Coaching insights" + rule-based bullets (L665-668, `coach.rule_based`) | Sentences with costs, %, "cost drivers" | Numbers and jargon in every line | **rename** to cue-style tips (rewrite in `coach.py`) |
| Button "Explain with LLM (paraphrases the structured result only)" (L670) | Rendered only when a key is set and the startup probe passes; the Grok probe fails, so it never appears; the reply is not cached and vanishes on rerun | Invisible; inside details; result lost | **rename** to "Deeper insight"; **move** to the sidebar; output in a header container; cached; shown disabled with a caption when no key |
| Header "Where the cost comes from" + stacked cost chart (L673-674) | Per-move cost by term | "cost" | **remove** |
| Expander "Why not just the shortest geometric path?" (L678-683) | Geometric baseline cost, sequence and image | Judge-facing jargon | **remove** (one sentence in "How it works") |
| Expander "Personalized feasibility graph" (L685-688) | Graph image + "edge" caption | "feasibility", "edge", "state" | **remove** (one sentence in "How it works") |
| Expander "Diff view (both betas on one wall)" (L690-691) | Both lines on one image; legend says observed / optimized | "beta", "observed", "optimized" | **move** into step-2 details; legend reworded |
| *(new)* "Climbing style" selector (Balanced / Fewer moves / Shorter reaches), "Expected difficulty at N°" line, Why block, injury flags | — | — | added per spec (F30, E24, E25, H36) |

## 6. Tab 4 · Personalize

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Header "Same route, different body" (L695) | Static | fine | **keep** (step 3) |
| Caption "The climber's measured morphology is scaled… Every feasible edge and every cost is recomputed…" (L696) | Static | "morphology", "feasible edge", "cost" | **rename** to "We shrink or stretch your measured body and re-plan the same holds." |
| Preset row (L697) | Same five/three buttons as tab 3 | See tab 3 rows | covered above |
| Slider "Simulated effective reach (% of measured)" 70–125 (L698) | Sets `scale_pct`; re-plans | Percent only; "measured" | **rename** to "What if I were shorter or taller?", in cm or inches once a scale exists, else % |
| Card "Measured climber" (L707) | Cost, moves, feasible pairs | "cost", "feasible" | **rename** to "You: N moves" (cost and pair counts into details) |
| Card "Simulated N % reach" (L709) | Same for the scaled body | Same | **rename** to "15 % shorter: N moves" |
| Card "Beta changes?" yes/no (L712) | Compares the two sequences | "beta", "cost" | **rename** to "Different moves? yes / no" |
| Card "Holds that differ" (L715) | Symmetric difference of hold sets | "symmetric difference" printed verbatim | **rename** to "Holds that change: H3, H7" |
| Warning "For the simulated climber the graph was disconnected…" (L717) | Relaxation notice (never fires on the demos) | "graph", "envelope" | **rename** to plain copy |
| Images "MEASURED climber · optimized beta" / "SIMULATED N % reach · optimized beta" (L721-722) | Side-by-side panels with cost subtitles inside the image; pink has no legend | Jargon inside the image | **keep**; titles "You" / "15 % shorter", sub-line "N moves", one legend line |
| Lines "Measured: …" / "Simulated: …" sequence text (L723-724) | Hold ids with reach % | Percentages | **move** into a collapsed details expander |
| Header "Cost of the same route across morphologies" + sweep table (L725-733) | Re-plans at 70–120 %; up to five fresh searches on the first visit | Table in the main flow; jargon; slow on first open | **move** into a collapsed expander computed only when opened; **rename** |
| Error "Could not compute a beta for one of the climbers." (L735) | Fallback | "beta"; unreachable for the measured body | **rename** |

## 7. Method tab, demos and scripts

| Control | What it does today | Problem | Decision |
|---|---|---|---|
| Method tab text (L739-765) | States, Dijkstra, cost formula, scope | Jargon by design; says "Dijkstra" while the Solver card says "A*" | **move** into the collapsed "How it works" at the bottom of step 2; fix the solver name |
| `demos.json`: spray-wall demo | Own footage, colour holds, five presets | Spray wall leaves the UI | **remove** the entry (code and assets stay in the package) |
| `demos.json`: Kilter demo | CruxCam clip, LED holds, `default_finish: top`, three presets | Needs a route image and the two-preset list | **keep**; becomes the only demo, with a `route_image` field |
| `scripts/ui_screenshots.py` presentation block (L83-101) and tab names | Screenshots 10-13 via `?present=1` | Presentation mode removed; tab names change | **remove** the block; re-script for the three steps |
| `scripts/record_walkthrough.py` | Records the present-mode walkthrough | Presentation mode removed | **rename** its flow to the three steps without `?present=1` |
| `docs/screenshots/10_present_*.png` … `13_present_fullbody.png` | Presentation captures | Obsolete | **remove**; regenerate 01-09 |

---

## Counts

| Decision | Rows (by primary decision) |
|---|---|
| keep | 15 |
| rename | 48 |
| move | 29 |
| remove | 32 |
| total | 124 |

## Verified behaviours that drove the decisions

- **"Load" never has a visible effect** in normal use: the first demo auto-loads on the first render (L317), so a click reloads what is shown and silently drops unsaved edits.
- **"Re-detect sequence" is a no-op**: every add / remove / move / rating / role edit already calls `refresh_observed()` (L179), and the sequence is shown only on tabs 2 and 3.
- **"Back to measured"** only clears `sim_on_opt` (L222); on tab 4 the slider stays at 85 %, so nothing changes there.
- **"Reset ratings"** does nothing unless a rating differs from the baseline (L218-220).
- **"Show full-body plan"** flips a toggle rendered below the fold (L610) with no wait warning.
- **The LLM button never appears**: it is gated on `_llm_ok()` (L669), and the Grok probe fails on this machine.
- **The pre-load info (L350) is dead code** because of the auto-load.
- **Optimizer errors call `st.stop()` inside tab 3** (L556, L560), which blanks tabs 4 and Method.
- **Hold ids are inconsistent**: images draw "25", text says "H25".
- **The contact sheet skips moves** on longer clips (stride sampling in `viz.contact_sheet`).
- **The "lower the reach envelope" advice (L526) is wrong**: contact detection never reads that setting.

## Judgement calls to confirm

1. **Finish hold selector (L551): remove.** The pink holds define the finish; role edits in step 1 cover overrides. Say so if you want it kept under Advanced.
2. **"On route" toggle (L451): remove.** Every hold from the image is lit; "Remove this hold" handles a false detection.
3. **"Re-run live" (L319): remove from the UI.** `scripts/build_demo_cache.py --force` remains the rebuild path.
4. **Simulated strip on tab 3: remove**, since step 3 shows the same side-by-side after the preset.
5. **"Show sequence" expander lives in step 1** (per C15), holding the sequence table and the key-frame sheet.
6. **"Require both hands on finish" defaults to on** for Kilter and stays reachable under Advanced.
