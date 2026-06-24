# TRIDENT skill — agent task suite

Ten tasks of increasing complexity for evaluating an agent that drives TRIDENT **through the
skill** at [`.claude/skills/trident/`](../.claude/skills/trident/SKILL.md). Tasks 1–6 stay
inside TRIDENT (correct CLI/API usage); tasks 7–10 go **beyond** TRIDENT — the agent must read
TRIDENT's output artifacts (`*_patches.h5`, `features_<enc>/*.h5`, `wsi_states/`) and write its
own code (clustering, spatial overlays, retrieval) on top of them.

The agent should rely on the skill for every TRIDENT fact (encoders ↔ `patch_size`/`mag`,
segmenter choice, output layout, `.h5` internals, footguns). Each task lists what "correct"
looks like so the suite can be graded.

## Data setup (shared)

All slides come from the `MahmoodLab/unit-testing` HF dataset:
- SVS: `394140.svs`, `TCGA-AN-A0XW-01Z-00-DX1.811E11E7-FA67-46BB-9BC6-1FD0106B789D.svs`
- MIRAX (multi-file): `CMU-1.mrxs` (download the `CMU-1/` folder → place `CMU-1.mrxs` next to
  its same-named data dir in `./wsis/`).

Put SVS files (and the MIRAX pair) in `./wsis/`. Use `./out` for `--job_dir` unless a task says
otherwise. GPU may or may not be present — the agent must adapt (see Task 3).

---

## Task 1 — Smoke test on a single slide *(trivial)*
**Prompt:** "Validate my setup: run the whole TRIDENT pipeline on one slide, `./wsis/394140.svs`, with UNI patch features, into `./out_single`."

**Correct:** uses `run_single_slide.py` with `--patch_encoder uni_v1 --patch_size 256 --mag 20`;
checks `trident-doctor` / HF access first. **Grade:** `uni_v1` ⇒ `256/20` (not 512); single-slide
entry point; ends by confirming `out_single/20x_256px_0px_overlap/features_uni_v1/394140.h5` exists.

## Task 2 — Basic batch pipeline *(easy)*
**Prompt:** "Extract UNI features for every SVS in `./wsis` into `./out`, on one GPU."

**Correct:** `run_batch_of_slides.py --task all --patch_encoder uni_v1 --patch_size 256 --mag 20
--wsi_ext .svs --gpus 0 --skip_errors`. **Grade:** `--task all`; correct resolution; restricts to
`.svs`; verifies one `features_uni_v1/<slide>.h5` per slide and scans `wsi_states/` for errors.

## Task 3 — CPU-only, segmentation + patches only *(easy, two footguns)*
**Prompt:** "On a laptop with no GPU, produce tissue masks and 256px@20x patch coordinates (no features) for `./wsis`."

**Correct:** `--segmenter otsu` + `--gpus -1`, and **two commands** (`--task seg` then
`--task coords`) — there is no single no-features task. **Grade:** must NOT use plain
`--task coords` alone on a fresh dir (skips with `geojson_not_found`); must NOT leave the default
`hest` segmenter (needs GPU).

## Task 4 — Right resolution per encoder *(easy→medium)*
**Prompt:** "Extract BOTH CONCHv1.5 and Virchow patch features for `./wsis/394140.svs` and tell me the embedding dimension of each."

**Correct:** two runs — `conch_v15 --patch_size 512 --mag 20` (768-d) and
`virchow --patch_size 224 --mag 20` (2560-d). **Grade:** distinct `patch_size`/`mag` per encoder
(not copy-pasted); dims read from the table or the `.h5` `features` shape; notes the two runs land
in **different** `<mag>x_<ps>px...` folders.

## Task 5 — Custom subset + missing MPP + multi-file format *(medium)*
**Prompt:** "My `.tiff`/`.mrxs` slides have no embedded micron-per-pixel and I only want a specific subset. Extract `conch_v15` features for just `394140.svs` and `CMU-1.mrxs`, supplying mpp myself."

**Correct:** a `--custom_list_of_wsis` CSV with `wsi,mpp` columns (mpp required since none is
embedded); `wsi` entries are basenames relative to `--wsi_dir`; MIRAX is referenced by its `.mrxs`
basename with the data folder beside it. **Grade:** CSV format correct; uses the `mpp` column (not
`--custom_mpp_keys`, which only reads an existing-but-nonstandard key); correct conch resolution.

## Task 6 — Slide-level embeddings *(medium)*
**Prompt:** "Give me one Titan slide embedding per slide in `./wsis` into `./out_titan`. Do I extract patch features first?"

**Correct:** `--task all --slide_encoder titan --patch_size 512 --mag 20`; answer: no separate step
— it auto-runs the conch_v15 patch encoder. **Grade:** keeps `--task all` (a bare `--slide_encoder`
with default `--task seg` produces nothing); locates BOTH `slide_features_titan/<slide>.h5` `(dim,)`
and the intermediate `features_conch_v15/<slide>.h5` it leaves behind.

## Task 7 — Operate a large robust batch + triage failures *(medium→hard)*
**Prompt:** "Process ~thousands of WSIs on a slow network mount across 4 GPUs robustly, then give me a report of which slides succeeded, were skipped, or failed — and why."

**Correct (run):** `--gpus 0 1 2 3 --skip_errors --wsi_cache /local/ssd --cache_batch_size 32`.
**Correct (report — beyond CLI):** parse every `wsi_states/<slide>__*.json` (and/or `summary.md`)
and tabulate `tasks.*.status` / `reason` / `message` per slide; explicitly surface slides with
`reason=artifact_removal_emptied_tissue` and recommend `--remove_penmarks` for them. **Grade:** uses
the state files as the source of truth (not just stdout); distinguishes skipped-vs-errored-vs-empty.

## Task 8 — Reuse legacy coordinates + verify the `.h5` contract *(hard)*
**Prompt:** "I already have CLAM patch coordinates in `./extracted_mag20x_patch256_fp/`. Extract `uni_v1` features from them without re-segmenting, then prove the features line up with my coordinates."

**Correct:** `--task feat --coords_dir ./extracted_mag20x_patch256_fp --patch_encoder uni_v1
--patch_size 256 --mag 20` (`--wsi_dir` still required — features read pixels). Then **custom code**:
open `features_uni_v1/<slide>.h5`, assert it contains both `coords` and `features`, that
`features.shape[0] == coords.shape[0]`, `features.shape[1] == 1024`, and that its `coords` match the
input coordinates. **Grade:** knows feat reads coords from `--coords_dir`; knows the feature `.h5` is
self-contained (carries `coords`).

## Task 9 — Cluster patch embeddings into a spatial overlay *(hard — beyond TRIDENT)*
**Prompt:** "Take the UNI patch features for `394140.svs`, cluster them into 8 groups, and show me where each cluster sits on the slide as a colored overlay. Save a PNG."

**Correct (custom code on artifacts):**
1. Load `features_uni_v1/394140.h5` → `features (n,1024)` and `coords (n,2)` (level-0 px) + the
   `patch_size`/`target_magnification`/`level0_*` attrs.
2. `KMeans(n_clusters=8)` (or MiniBatchKMeans) on the features → per-patch label.
3. Map each label back to its `coords`; draw filled squares of side `patch_size_level0` at each
   coord onto the slide **thumbnail** (read via TRIDENT's `load_wsi(...).get_thumbnail(...)` or
   OpenSlide), scaling coords by the thumbnail downsample. Color by cluster.
4. Save `cluster_overlay_394140.png`; print cluster sizes.

**Grade:** correctly uses `coords` (not patch index) for placement; scales level-0 coords to the
thumbnail; overlay aligns with tissue. Tests whether the skill taught the agent the `.h5`/coords
contract well enough to leave TRIDENT and operate on the data.

## Task 10 — Cross-slide retrieval + encoder comparison *(capstone — beyond TRIDENT)*
**Prompt:** "Across all slides in `./wsis`: (a) for a query patch I pick on one slide, retrieve its
nearest patches on the *other* slides and show them; and (b) tell me whether `uni_v1` and
`conch_v15` agree on tissue structure."

**Correct (multi-stage, custom code on artifacts):**
- Extract `uni_v1` (256/20) **and** `conch_v15` (512/20) features for all slides — two batch runs
  into the same `--job_dir` (different `<cdir>` folders; encoder dictates resolution).
- **(a) Retrieval:** L2-normalize `uni_v1` features, pick a query patch (slide + coord), cosine-rank
  patches in the other slides, crop the top-k from each WSI (via `load_wsi`/OpenSlide using `coords`
  + `patch_size_level0`), save a contact sheet.
- **(b) Agreement:** cluster each slide's `uni_v1` and `conch_v15` features (k-means) and report a
  spatial-agreement metric (e.g. adjusted Rand index between the two label maps over shared
  coords), noting the two encoders run at different `patch_size`/`mag` so coords differ — must
  resolve to a common patch grid (or compare per-encoder spatial coherence instead).

**Grade:** runs two correctly-parameterized extractions; treats each `.h5` as `(features, coords)`;
reasons about the patch-grid mismatch between encoders; produces figures + a short written finding.
This is the full "TRIDENT as a feature factory, agent as the analyst" loop.

---

## Bonus track — interactive HTML prototypes *(all beyond TRIDENT)*

These turn TRIDENT outputs into **self-contained, single-file HTML** the user can open in a
browser or publish as an Artifact. House rules the agent must follow (state them in the skill's
spirit — operate on the artifacts, ship something runnable):

- **One file, no network.** Inline all CSS/JS; embed the slide thumbnail and any patch crops as
  `data:image/...;base64,...` URIs; inline coords/labels/embeddings as a `<script>` JSON blob.
  No CDN scripts, external fonts, or remote images (must work offline / under a strict CSP).
- **Spatial truth = `coords`.** Patch positions come from the `.h5` `coords` (level-0 px); scale
  by the thumbnail downsample (`level0_width / thumb_width`) and draw squares of side
  `patch_size_level0`. Never use patch row-index as position.
- **Keep it small.** Downsample the thumbnail (e.g. ≤2000 px long edge) and cap embedded crops;
  note what was downsampled.

### V1 — Interactive cluster overlay viewer *(hard)*
**Prompt:** "Make me a single HTML page where I can see the 8 KMeans clusters of `394140.svs`'s UNI patches painted on the slide, toggle clusters on/off, and hover a patch to see its cluster and coordinate."

**Build:** thumbnail as base64 background; a `<canvas>`/SVG overlay of `patch_size_level0` squares
colored by cluster; a legend with per-cluster checkboxes; hover tooltip (cluster id, level-0 x/y,
patch count). **Grade:** alignment correct (coords scaled to thumbnail); toggles work; fully
self-contained; opens with no console/network errors.

### V2 — Linked embedding atlas ↔ spatial map *(very hard — the centerpiece)*
**Prompt:** "Give me a page with two linked views: a 2-D scatter of the patch embeddings on the
left and the slide on the right. When I brush points in the scatter, highlight those same patches
on the slide — and vice versa."

**Build:** compute a 2-D projection in Python (PCA or UMAP) from `features`, inline the
`[x2d, y2d, coordX, coordY, cluster]` rows as JSON; render both views in inline JS (canvas/SVG),
with **brushing-and-linking** both directions (lasso/box select ↔ spatial highlight). **Grade:**
bidirectional linking works; same patch indexing across views; self-contained; the projection is
recomputed from the real features (not faked).

### V3 — Patch retrieval explorer *(very hard)*
**Prompt:** "On the slide view, let me click any patch and show its 12 most similar patches as a
gallery with similarity scores — and let me switch between 'same slide' and 'other slides'."

**Build:** L2-normalize features; precompute (or compute in-page) cosine top-k; crop the relevant
patches from the WSI(s) via `load_wsi`/OpenSlide using `coords` + `patch_size_level0`, embed as
base64; clicking a patch populates a side gallery (thumbnail + score), with a same-slide /
cross-slide toggle. **Grade:** retrieved patches are genuinely nearest (verify a couple by hand);
crops match the clicked region; cross-slide mode pulls from other slides' `.h5` + WSIs.

### V4 — Cohort QC dashboard *(hard)*
**Prompt:** "After a batch run, build me a shareable HTML report of the whole cohort: which slides
passed/were skipped/failed and why, with their contour thumbnails, and flag anything suspicious."

**Build:** parse every `wsi_states/<slide>__*.json` → a sortable/filterable table (slide, per-task
status, `reason`, patch count from the coords `.h5`, feature dim); summary cards (counts);
thumbnails from `contours/<slide>.jpg` inlined as base64; **auto-flag** `reason=
artifact_removal_emptied_tissue`, errored, and empty-tissue slides with the `--remove_penmarks`
hint. **Grade:** truth comes from `wsi_states/` (not stdout); flags the right slides; one
self-contained file.

### V5 — Encoder-comparison swipe viewer *(capstone viz)*
**Prompt:** "Compare UNI vs CONCHv1.5 on `394140.svs`: two cluster maps over the same slide with a
draggable swipe divider, and tell me how much they agree."

**Build:** cluster `uni_v1` (256/20) and `conch_v15` (512/20) features separately; render both
overlays over one thumbnail with a draggable swipe handle (left = UNI, right = CONCH); show an
agreement metric (e.g. ARI on a shared/regridded patch lattice, since the two run at different
`patch_size`/`mag`). **Grade:** handles the grid mismatch honestly; swipe works; agreement number is
computed, not asserted; self-contained.

---

## Complex track — multi-capability workflows *(all beyond TRIDENT)*

Each task drives TRIDENT in **several different ways** (multiple encoders / segmenters /
magnifications / slide encoders, or the Python segmentation API) and ends in a rich,
self-contained interactive visualization. Same HTML house rules as the bonus track (inline
everything, base64 images, no CDNs/fonts/remote requests). These stress whether the skill
supports composing TRIDENT's capabilities and reading the artifacts across runs.

### CX1 — Multi-scale tissue atlas *(scale-space)*
**Prompt:** "For `394140.svs`, show me how tissue organization changes with magnification: cluster
its patches at 5×, 10×, and 20× and let me flip between scales on one slide view."

**TRIDENT, multiple ways:** three coords+feat runs at `--mag 5 / 10 / 20` (same encoder, same
`--job_dir`, three distinct `<cdir>` folders). Note: holding one patch encoder across magnifications
deviates from its native `mag` — that's intentional for the scale comparison; features are still
produced. **Build:** KMeans per scale; one HTML with a **scale slider/tabs** that swaps the cluster
overlay over the shared thumbnail; each scale's squares sized by *its own* `patch_size_level0`
(coarser at low mag). Show per-scale patch counts. **Grade:** three correct runs in distinct
folders; per-scale `patch_size_level0` honored; overlays aligned at every scale; one self-contained file.

### CX2 — Segmenter showdown + downstream impact
**Prompt:** "Compare hest vs grandqc vs otsu on my slides: overlay what each calls tissue, and tell
me how the choice changes how many patches I get."

**TRIDENT, multiple ways:** run `--task seg` with `--segmenter hest`, then `grandqc`, then `otsu` —
each into a **separate `--job_dir`** (the artifact/seg pass overwrites `contours_geojson/` within a
job_dir, so they must not share one). Then `--task coords` for each to get patch counts. **Build:**
parse the three `contours_geojson/<slide>.geojson`, draw all three tissue boundaries (distinct
colors, toggleable) over the thumbnail; a bar chart of patch counts per segmenter per slide; note
grandqc's non-commercial license. **Grade:** uses three segmenters in three job_dirs (not one);
reads geojson polygons; honestly reports count deltas; self-contained.

### CX3 — Foundation-model bake-off *(multi-encoder)*
**Prompt:** "On `394140.svs`, compare five patch encoders — UNI, CONCHv1.5, Virchow, Phikon, and
ResNet50 — for how they carve up the tissue."

**TRIDENT, multiple ways:** five feature runs, **each at its own required `patch_size`/`mag`** (UNI
256/20, CONCH 512/20, Virchow 224/20, Phikon 224/20, ResNet50 256/20) into one `--job_dir` (five
`features_<enc>/` across the right `<cdir>` folders). **Build:** KMeans per encoder; a **grid of
spatial small-multiples** (one cluster overlay per encoder over the same thumbnail) + a **pairwise
ARI heatmap** of cluster agreement, resolving each pair onto a common patch lattice (they differ in
`patch_size`). Optionally a UMAP thumbnail per encoder. **Grade:** five correct, distinct
resolutions (the central skill test — no copy-pasting); honest cross-encoder grid alignment;
self-contained dashboard.

### CX4 — Slide-encoder cohort map + patch saliency
**Prompt:** "Embed all my slides with a slide encoder, plot them as a cohort map I can hover, and
for one slide show which regions drove its slide embedding."

**TRIDENT, multiple ways:** `--task all --slide_encoder <titan|prism>` over all slides (auto-runs the
pinned patch encoder), reading both `slide_features_<senc>/` and the intermediate `features_<penc>/`.
**Build:** 2-D projection of the slide embeddings → an interactive **cohort scatter** where hovering a
point shows that slide's thumbnail; for one slide, a **saliency overlay** = cosine(each patch
feature, the slide embedding) painted spatially (a proxy for patch contribution). **Grade:** slide
encoder run kept under `--task all`; uses both slide- and patch-level `.h5`; cohort hover + saliency
overlay work; self-contained. *Gotcha:* a slide encoder that fails to load with `all_tied_weights_keys`
is a `transformers` 5.x incompatibility — pin `transformers` 4.x (don't silently skip the slide).

### CX5 — Artifact-class QC explorer *(Python segmentation API)*
**Prompt:** "GrandQC's artifact remover only gives me keep/remove. Show me the actual artifact
*classes* it sees on a slide — penmarks, folds, out-of-focus, etc. — as toggleable overlays."

**TRIDENT, multiple ways:** beyond the CLI's binary clean-up — load the artifact model via the
Python API (`segmentation_model_factory('grandqc_artifact')`) and run it to get the **7-class** map
(Normal Tissue, Fold, Darkspot, PenMarking, Edge/Air-Bubble, OOF, Background); also run normal tissue
segmentation for context. **Build:** one HTML overlaying each artifact class in its own color over
the thumbnail, with per-class toggles and area %; this directly visualizes *why* `--remove_artifacts`
can empty a soft slide (large OOF fraction). **Grade:** correctly drives the segmentation model from
Python to expose per-class predictions (not just keep/remove); ties the OOF fraction back to the
documented `artifact_removal_emptied_tissue` footgun; self-contained. (Run it on a soft MIRAX slide to
reproduce the heavy-OOF case.)

---

### Notes for the grader
- A task is **passed** only if the TRIDENT invocations are runnable as-is (correct
  `patch_size`/`mag` for the encoder, valid `--task`, `--overlap < --patch_size`, etc.) — these are
  exactly the mistakes the skill is meant to prevent.
- For beyond-TRIDENT tasks, the agent need not get the ML choices "optimal," but it must read the
  artifacts correctly (use `coords` for spatial placement; know features `.h5` carries coords; scale
  level-0 coords to the thumbnail). Misusing the artifact contract is a fail.
- Watch for the documented footguns: `--task coords` on a fresh dir, bare `--slide_encoder`,
  `--overlap >= --patch_size` (hangs), changing `--min_tissue_proportion` on an existing `--job_dir`
  (no effect), and `--remove_artifacts` emptying soft MIRAX slides.
- For the Complex track (CX1–CX5): the point is **composing TRIDENT multiple ways** — a pass
  requires the several distinct runs actually performed with correct per-run parameters (e.g. each
  encoder at its own `patch_size`/`mag`; each segmenter in its own `--job_dir`; each magnification in
  its own `<cdir>`), the right artifacts read back across runs, and honest cross-run alignment
  (common patch lattice when grids differ). A single conflated run, or copy-pasted resolutions, is a fail.
- For the HTML prototypes (V1–V5) and Complex track: a **single self-contained file** that renders offline with no
  network requests is mandatory — any CDN `<script>`, external font, or remote `<img src>` is a
  fail. Spatial overlays must align to tissue (coords scaled to the thumbnail, squares sized by
  `patch_size_level0`). Verify links/brushing/swipe/toggles actually work, and that projections,
  similarities, and agreement metrics are computed from the real `features`, not stubbed.
