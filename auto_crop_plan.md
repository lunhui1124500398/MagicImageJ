# Auto Crop Plan

## Goal

Build a high-recall semi-automatic ROI proposal workflow for low-SNR TEM data.

The immediate goal is not full automation. The immediate goal is:

1. Let the user define a coarse focus area, or provide a few example particles.
2. Generate many candidate boxes inside the focus area.
3. Let the user delete, refine, and optionally mark the primary target.
4. Save all accepted and rejected feedback in a training-friendly format.

## Current Decisions

### 1. Interaction Style

- Use a semi-automatic workflow first.
- Favor recall over precision.
- Allow slightly oversized boxes if that helps preserve moving particles.
- Accept occasional two-particle boxes, but preserve a way to mark the main target.
- Treat `focus_regions` as a search-area guide, not a hard box-clipping rule.
- Do not require proposal boxes to stay fully inside `focus_regions`, because
  edge particles may legitimately cross the boundary.

### 2. Data Collection Principle

We should not only save final accepted ROIs. We should also save the user intent
that explains why some regions matter and others do not.

Each reviewed image should eventually collect:

- `focus_regions`: where the user wants the algorithm to search.
- `ignore_regions`: where the user intentionally does not care.
- `accepted_candidates`: auto proposals that the user kept.
- `rejected_candidates`: auto proposals that the user deleted.
- `primary_point`: when a box contains multiple particles, mark the main target.
- `visible_frame_range`: optional later annotation for track disappearance.

### 3. Temporal Disappearance

This is intentionally decoupled from the first training target.

For now:

- Detection training uses reviewed overview frames only.
- Per-track crop sequences are still exported and preserved.
- Particle disappearance is stored as `unknown` unless later annotated.

This means the user does **not** need to manually remove vanished particles
before the first export.

### 4. Why Detection First Instead of Segmentation

The data is low-SNR and many particle boundaries are not clean enough for a
segmentation-first workflow.

So the staged approach is:

1. Detect or propose boxes first.
2. Add temporal logic later if needed.
3. Only consider segmentation or SAM-style refinement after box proposals are
   already useful.

## Training Format

We now keep two synchronized dataset views:

### Canonical export

Purpose:

- Preserve the rich supervision needed for future iteration.
- Avoid locking ourselves into a single model family too early.

Files:

- `canonical/manifest.jsonl`
- `canonical/tracks.jsonl`
- `canonical/sessions.json`
- `canonical/review_templates/*.json`

Notes:

- `manifest.jsonl` stores full-frame reviewed images and their accepted boxes.
- `tracks.jsonl` stores per-particle crop sequence metadata.
- `review_templates` store placeholders for focus regions, rejected candidates,
  primary points, and future visibility labels.

### Derived YOLO detect export

Purpose:

- Provide a direct path for local detector training.

Files:

- `yolo_detect/images/*`
- `yolo_detect/labels/*`
- `yolo_detect/data.yaml`

Important:

- Only reviewed overview frames are exported as positive full-frame detection
  examples in the first pass.
- This avoids poisoning the detector with labels on later frames where a
  particle may have disappeared.

## Data Scale Guidance

The current session already contains useful signal:

- 53 reviewed ROIs
- 53 crop sequences
- 137 frames per crop sequence

But these 137-frame sequences are highly correlated. For model development, the
meaningful unit is the number of independent reviewed particles and reviewed
full frames, not the raw image count.

Suggested milestones:

- First usable local detector: 300 to 800 independent reviewed ROIs.
- More stable version: 1000 to 3000 independent reviewed ROIs.

## Local Model vs API

Primary direction:

- Train locally first.

Reason:

- The user has strong local hardware.
- The task is specialized.
- The task needs dense box proposals more than general image reasoning.

Possible model roles:

- Local detector: main candidate generator.
- Local temporal refiner: later use for visibility and box inflation.
- API model: optional later use for candidate ranking or review assistance, not
  the main detector.

## Progress

### 2026-03-29

Completed:

- Reviewed current MagicImageJ ROI export flow.
- Confirmed that `processing_log.json` preserves the manual ROI set.
- Confirmed that a reviewed overview frame is available.
- Added `scripts/export_auto_crop_training_data.py`.
- Defined a canonical export format plus a derived YOLO detect export.
- Explicitly separated box placement from temporal disappearance.
- Added `scripts/propose_auto_crop_candidates.py` for local proposal testing.
- Ran the first blind proposal pass on the current session and exported preview
  images plus candidate JSON.
- Added `scripts/annotate_auto_crop_regions.py` so users can draw polygonal
  `focus_regions` and `ignore_regions` directly on the overview image.
- Verified the first focus-guided proposal pass on the current session.

Current blind-pass result on the existing session:

- Top 120 heuristic candidates overlap 28 of 53 manual ROIs.
- This is not yet strong enough for production use.
- The result reinforces that focus / ignore regions are not optional in
  practice; they are part of the intended interaction design.

Current focus-guided result on the existing session:

- The first user-drawn focus polygon covers about 69 percent of the image.
- All exported candidate centers fall inside the focus region.
- Top 120 overlap hits improved from 28 to 45 manual ROIs.
- Top 120 false positives decreased from 88 to 61.
- The next optimization target is proposal ranking quality inside the focus
  region, not a stricter geometric focus constraint.

Current V3 result on the existing session:

- Added blob-oriented scoring, line-like structure penalty, and spatially
  diversified candidate selection.
- Top 20 overlap hits improved from 6 in V2 to 15 in V3.
- Top 40 overlap hits improved from 14 in V2 to 28 in V3.
- Top 120 overlap hits improved from 45 in V2 to 51 in V3.
- The current prototype is now strong enough to start collecting reviewed
  sessions beyond the first demo session.

Current V3.5 result on the existing session:

- Added temporal reranking that re-checks each proposal against the full Enh
  sequence instead of trusting the overview frame alone.
- This stage is designed as a gentle re-ranker, not a replacement for the
  single-frame proposal stage.
- On the current session it improved the very front of the ranked list and
  reduced some false positives in the top 40.
- Example result on the current focus-guided session:
  `top10 overlap hits 4 -> 7`, `top40 overlap hits 28 -> 28`,
  `top40 false positives 21 -> 19` when comparing V3 to V3.5.
- The current interpretation is that temporal evidence is useful, but it should
  stay as a moderate bonus term until more reviewed sessions exist.

Next:

1. Use user examples and future accepted / rejected feedback for re-ranking.
2. Add `primary_point` support to the review workflow.
3. Start accumulating reviewed sessions from other datasets in the canonical format.
4. Move the validated interaction into MagicImageJ when the local prototype feels stable.

## Prototype Usage

### Export training data

```bash
python scripts/export_auto_crop_training_data.py ^
  --session-root "D:\path\to\session_root" ^
  --output-root "D:\path\to\training_export"
```

### Bootstrap a review template before manual batch-crop exists

```bash
python scripts/bootstrap_auto_crop_review.py ^
  --session-root "D:\path\to\session_root" ^
  --frame-index 44 ^
  --output-dir "D:\path\to\review_templates"
```

### Generate local candidate proposals

```bash
python scripts/propose_auto_crop_candidates.py ^
  --session-root "D:\path\to\session_root" ^
  --output-dir "D:\path\to\proposal_output"
```

Enable temporal reranking:

```bash
python scripts/propose_auto_crop_candidates.py ^
  --session-root "D:\path\to\session_root" ^
  --output-dir "D:\path\to\proposal_output" ^
  --review-json "D:\path\to\review_template.json" ^
  --temporal-rerank
```

### Draw focus / ignore polygons interactively

```bash
python scripts/annotate_auto_crop_regions.py ^
  --review-json "D:\path\to\review_template.json"
```

Controls:

- Left click: add a polygon vertex.
- Right click or `Enter`: finish the current polygon.
- `F`: switch to `focus` mode.
- `I`: switch to `ignore` mode.
- `Backspace`: remove the last vertex of the current polygon.
- `Esc`: cancel the current unfinished polygon.
- `U`: undo the last saved polygon in the current mode.
- `C`: clear all saved polygons in the current mode.
- `S`: save the review JSON and a preview overlay image.
- `Q`: quit the editor.

Saved outputs:

- Updated review JSON in place.
- Region preview image: `<review_json_stem>_regions_preview.png`

### Convert proposals back into a MagicImageJ-loadable ROI JSON

```bash
python scripts/export_proposals_to_magicimagej_roi.py ^
  --proposal-json "D:\path\to\frame_0044_candidates.json" ^
  --output-json "D:\path\to\proposal_rois.json" ^
  --top-k 40
```

Optional focus and ignore rectangles:

```bash
python scripts/propose_auto_crop_candidates.py ^
  --session-root "D:\path\to\session_root" ^
  --output-dir "D:\path\to\proposal_output" ^
  --ignore-rect 1300,0,2134,260 ^
  --focus-rect 0,80,1700,845
```

Optional review template input:

- Edit the exported review template JSON manually, or use the polygon editor.
- Fill `focus_regions` and `ignore_regions`.
- Re-run the proposal script with `--review-json`.

Key outputs:

- Labeled preview: `frame_XXXX_candidates_top40_labeled.png`
- Full candidate overlay: `frame_XXXX_candidates_top120.png`
- Score heatmap: `frame_XXXX_score_heatmap.png`
- Candidate bundle: `frame_XXXX_candidates.json`

## Proposal V2 Direction

The second proposal pass should improve along five practical axes:

1. Use polygonal `focus_regions` and `ignore_regions` as a first-class search mask
   instead of treating them as an optional add-on.
2. Spend almost all candidate budget inside focus regions so that irrelevant
   corners do not consume ranked proposal slots.
3. Add region-aware ranking so one strong textured area does not monopolize the
   whole candidate list.
4. Re-rank using user examples such as accepted candidates or seed boxes to
   bias proposals toward what the user actually considers worth reviewing.
5. Keep proposal boxes slightly inflated and multi-scale so moving particles are
   less likely to be clipped in the first pass.

The current feedback also adds one important constraint:

- Do not over-constrain the proposal geometry to stay fully inside the focus
  polygon. The useful improvement target is better ranking inside the allowed
  area, not stricter clipping.

## Open Questions

- Should the first auto-proposal stage operate on one overview frame only, or
  also use temporal summary maps such as min, max, or variance?
- Should user feedback be stored directly inside `processing_log.json`, or in a
  separate auto-crop review file?

## When To Start Other Data

Short answer:

- Start now for data accumulation.
- Start a first pilot detector training run after roughly 5 to 10 reviewed
  sessions, or after roughly 300 to 800 independent accepted ROIs.

Why this timing is reasonable:

- The current local prototype is already useful enough to produce candidate
  boxes and reviewed feedback in a consistent format.
- More sessions now help us learn across different backgrounds, drift patterns,
  and particle motion cases.
- Training immediately on only one session would overfit badly and would not
  tell us much about real generalization.

Suggested rollout:

1. Continue refining the proposal workflow on the current session until the
   user feels the candidate list is saving time.
2. Begin reviewing additional sessions with the same focus / ignore + proposal
   + correction workflow.
3. Save every reviewed session in the canonical export format even before model
   training begins.
4. Launch a first local YOLO-style pilot detector when enough reviewed sessions
   have accumulated.
5. After that, use accepted and rejected proposals to train a stronger ranking
   stage, not just a raw detector.

Current overall strategy:

1. Use user-drawn `focus_regions` to define where search is worth spending.
2. Generate high-recall box proposals inside the focus area.
3. Let the user keep, delete, and later mark the main target inside ambiguous
   boxes.
4. Export all review feedback in a training-friendly canonical format.
5. After enough reviewed sessions exist, train a local detector and then a
   preference-aware re-ranker that better matches the user's selection style.
