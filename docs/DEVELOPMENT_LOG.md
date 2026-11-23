# Development log: decisions, reasoning, current state

Written 2026-11-23 at the end of a long session. Covers LEGO set 21205 piece recognition from photos of built towers.
Source: the working session itself (what was tried, what failed, why). Parts of the earlier work (data collection, labeling, the gallery build)
are described only as far as I knew them; those sections are marked **(thin)** and should be checked against the code.

---

## 1. Goal

Recognize every piece of LEGO set 21205 (45 part+colour rows, 212 pieces) in photos of real built towers, and eventually rebuild the tower
in 3D as an LDraw `.ldr` file.

Tower facts that drive every design choice:
- Flat walls, one stud thick, up to 16 studs wide.
- Standing on a printed 6x12 base plate.
- Photographed roughly from the angle in `real/image copy*.png` (the user's reference perspective), by hand, on a busy desk (keyboard, mat).

## 2. Recognition design (the stable part)

| Piece | Decision | Why |
|---|---|---|
| Shape | Colour-blind ResNet18 fingerprint embedding (`models/shape_embed.pt`), matched by kNN against a **gallery** (`data/gallery.npz`) of real photos | New pieces can be added by photographing them. No retraining. Colour-blind so one part in 5 colours is one shape class. |
| Colour | Separate Lab-space colour recognizer on the mask pixels (`L.brick_lab`) | Shape model ignores colour on purpose; colour is cheap and reliable when the mask is clean. |
| API | `lego_lookup.py` (`L.load_embedder`, `L.load_gallery`, `L.embed`, `L.recognize(..., closed_set=True, shape_min_sim=-1)`, `L.crop_box`) | One module reused by every notebook. |
| Closed set | Only the 45 rows of set 21205 can be answered | The set is known; closed-set answers are far more accurate than open-set. |

Data files: `data/lego_colors.csv` (set colours: White, Tan, Black, Dark Green, Sand Green, Reddish Brown, Light Bluish Gray, Pearl Gold),
`data/element_map.csv` (part + colour rows), `data/labels.csv`, `data/boxes.csv`, `data/ranges.csv`, `data/gallery.npz`,
`data/exif_orientation_before.csv`. **(thin)**

## 3. What was tried, in order, and why it was dropped or kept

### 3.1 Digital (rendered) towers: tried, abandoned
- `ldraw_render.py` renders LDraw towers with moderngl (orthographic and perspective, base plate, random camera, random widths 6-12).
- `notebooks/tower_seg.ipynb`: YOLO11-seg trained on renders to segment bricks.
- Problems: (a) renders do not look like the real photos; research (TU Delft) shows synthetic-only transfer is poor (F1 ~0.41-0.45);
  (b) the renderer leaked a GPU framebuffer on every call, which nearly crashed the PC. Fix: `_fbo(S)` is cached with `functools.lru_cache`.
- **Decision (user, explicit): digital towers are no longer used to judge anything.** Everything is judged by eye on the real photos in `real/`.
  The renderer and `tower_seg.ipynb` stay in the repo but are low priority and were not run after the fix.

### 3.2 SAM-family segmentation: rejected
Research conclusion: SAM-like models fail on same-colour touching bricks (they merge them; the seam has almost no signal).
This is why the project moved to "heavy image processing + controlled conditions".

### 3.3 Image-processing pipeline (current approach)
Three stages, each a plain function of the image so any stage can be swapped:
1. **Foreground (tower mask)**
2. **Separation (one label per brick)**
3. **Recognition** (cut each piece by its mask, grey elsewhere, embed, match to gallery, read colour from mask)

## 4. `notebooks/pipeline.ipynb` cell by cell

Run order for real photos only: **imports cell -> cell 2 -> 2b -> 2c -> 2c-fix (`0f5ff2aa`) -> 2e or 2f**. Do **not** use Run All:
cell 1 re-renders digital scenes and the last cell kills the kernel.

| Cell | Purpose | Key decisions / reasoning |
|---|---|---|
| `p1` (markdown) | Header | Stale: still describes the abandoned digital benchmark. Should be rewritten. |
| `11943eb9` imports | `cv2, numpy, matplotlib` | Lets you skip cell 1. |
| `7e354746` cell 1 | Renders digital test scenes | **Abandoned.** Not needed for real photos. |
| `b5637dc1` cell 2 | `K3`, `foreground` (plain-background GrabCut), `edge_map`, `separate` | `edge_map` combines Canny on L, Canny on Lab a and b (tan/white/grey share L but not colour), and a black-hat on L (shadow in a seam), after a bilateral filter (smooths studs and print, keeps seams). `separate` = mask minus dilated edges gives one seed per brick face, then `cv2.watershed` grows seeds back to the mask. |
| `90ca6968` cell 2b | Real-photo foreground by GrabCut from rectangles (`RECT`), `cleanup`, `colorize`, saves outputs | The background is busy so a "border = background" assumption fails. `RECT` is one rectangle or a list (stepped shape: narrow roof, wide body). It ends above the base plate because the plate's printed picture otherwise splits into ~20 fake bricks. `cleanup` merges fragments under `MIN_AREA=300` px into the neighbour with the longest shared border. Holes under `HOLE=1500` px are filled (window bars); bigger holes (arches) stay open. |
| `37e04276` cell 2c | BiRefNet-lite foreground (ONNX), `seam_lines`, `stud_pitch`, `separate_v2` | BiRefNet (salient-object model) cut by the same `RECT` replaces GrabCut when `models/birefnet_lite.onnx` + `onnxruntime` exist (fp32, 1024x1024 input, ImageNet normalisation). `seam_lines` = Hough segments kept only if within 6 degrees of horizontal/vertical, added to the edge map. `stud_pitch` fits a comb to the vertical-seam profile to get pixels per stud so each piece's width can be printed in studs. |
| `0f5ff2aa` 2c-fix | Memory fix | ONNX session with the CPU memory arena off, 4 threads; foreground cached per photo so BiRefNet runs once. See section 6. |
| `e84e1aa0` 2d | Diagnostic: what does BiRefNet see | Raw probability vs `>0.5` vs GrabCut. Use it to decide whether the model or `RECT` is the problem. Re-runs the model (not cached). |
| `8f0999ef` 2e | Recognition on every piece >= `MIN_PIECE=400` px | Output is now a zoomed photo with a number per piece plus a colour-coded list (green >= 0.93, orange 0.85-0.93, red < 0.85) because overlaid text was unreadable when stacked. |
| `f2583862` 2f | **Separation variants, real photos only** | Switchable cues on top of the edge map, drawn side by side: `+seams`, `+color` (k-means on Lab, label-change boundaries; lightness weighted 0.6), `+merge` (merge neighbours with similar mean Lab and weak shared edge, `dE=12`, `thr=0.3`), and combinations. No ground truth, so judge by eye. |
| `a2064f19`, `0cdce91a` cells 3, 4 | Digital metrics (PQ/SQ/RQ, merges) and plots | **Abandoned with the digital benchmark.** Last digital numbers: PQ ~0.23 full / ~0.30 oracle foreground. `score` lives here, which is why the old 2f threw `NameError: score`. |
| `d969a12f` | Kills the kernel | Frees GPU/RAM after Run All. This is why Run All "crashes" the kernel by design. |

## 5. Other notebooks and scripts **(thin: described from memory of the session)**

- `notebooks/test_real.ipynb`, `notebooks/lookup.ipynb`: real-photo and lookup tests.
- `notebooks/blur.ipynb`: blur experiments; its data-fix cell `535a886b` (the 2449 / 4460b folder fix) is **still waiting for the user to run it**.
- `notebooks/towers.ipynb`: has a digital gap test (`ca86d351`), unrun, low value now.
- `notebooks/tower_seg.ipynb`: YOLO11-seg on renders (see 3.1). Render cell `f2b7e769` is resumable, uses `plate=True`, keyboard-band backgrounds; training cell `55a5d2b2` uses batch 8, workers 4, `cache=False`. Not run.
- `scripts/tower_labeler.py`, `labeling/tower_classes.txt`: labeling real towers. `scripts/exclude_photos.py`: drops bad photos. `labeling/labels/*.json`: per-image labels.
- `scripts/eval_lookup.py`: marked for deletion.
- `app.py`: app entry point. **(not reviewed this session)**
- `ldraw_render.py`: renderer (`render_scene`, `make_tower(plate=...)`, `flatten`).

## 6. Incidents and fixes

1. **PC near-crash running `tower_seg`**: GPU framebuffer leak in `render_scene` (a new 1280x1280 buffer per call). Fixed by caching `_fbo`; render loop made resumable; training batch reduced.
2. **PC near-crash running `pipeline.ipynb`**: cause **not confirmed**. The kernel died, so no process was left to inspect. Best guess: BiRefNet on CPU at 1024x1024 re-run in cells 2c, 2d, 2e, 2f in one kernel with onnxruntime's arena never releasing memory. Mitigation added (`0f5ff2aa`). If it recurs, watch `free -h` while running and note which cell spikes; next suspects are cell 2b's GrabCut and the k-means distance matrix in `color_cut`.
3. `TypeError` in `seam_lines`: OpenCV returns `(N,1,4)` or `(N,4)`. Fixed with `L.reshape(-1, 4)`.
4. **Notebook edit conflicts**: `pipeline.ipynb` open in the IDE plus a running kernel rewrites the file, so edits failed with "File has been modified since read". Workaround: close the tab or stop the kernel, Read then edit immediately.
5. BiRefNet confusion: it silently fell back to GrabCut when `onnxruntime` was missing from the kernel. Cell 2c now prints which foreground is in use. The user installed `onnxruntime` in `.venv`; the notebook must use that kernel.

## 7. Observations on real photos (no ground truth, by eye)

- GrabCut foreground leaks on `image.png` (wall/monitor around the roof); BiRefNet is the intended fix.
- Separation merges the dark-green roof, black grille and window; shatters the window frame and brown logs into slivers; merges tan arch pieces.
- Recognizer similarities ~0.78-0.90; fragmented pieces are mostly misnamed.
- Stud-pitch estimator is wrong (got 59/85/61.8/100/132 px against an expected ~32-35 and ~60): the comb score is biased toward large spacings. Hough at higher sensitivity over-splits the window grille and roof edge into slivers.

## 8. What could be improved (ranked by expected value)

1. **Capture conditions** (cheapest, biggest win): plain background, fixed camera on a stand, plate always in frame, a grey card for colour. Every research source and every failure above points here.
2. **Separation**: judge 2f's variants by eye first. Then fix the stud pitch with autocorrelation of the seam profile (not the comb fit), use mid-sensitivity Hough, drop line cues on tiny regions, and use pitch to snap seams to the stud grid. Per-depth pitch for perspective.
3. **Gallery gap**: add wall-face views from real labeled towers. Clean pieces being misnamed means the gallery only holds single-piece photos, not pieces as seen in a wall.
4. **Foreground**: confirm BiRefNet is running (cell 2d), replace hand-set `RECT` rectangles (per photo, manual) with something automatic such as plate detection.
5. **A real evaluation**: label a few real towers with `scripts/tower_labeler.py` and score separation/recognition on them. Right now every judgement is by eye.
6. **Few-shot fine-tuning on real towers** (the Brickognize recipe: renders plus a few real fine-tunes) once the pipeline is stable.
7. **Housekeeping**: rewrite the stale `p1` header, delete cells 1, 3, 4 and the kill cell (or move to a separate notebook), delete `scripts/eval_lookup.py`, split pipeline functions into a `.py` module so notebooks stop duplicating them.

## 9. Pending / owed

- User: run the data-fix cell in `blur.ipynb`; reshoot missing pieces (2449 and 4460b grey counts, `6178pr0002`, `3069b`, `63864`); build and label real towers.
- Refresh the graphify graph after the file additions (not done).
- Run 2f and send `output/<name>_separation_variants.png` so the best cue can be chosen.
- Rerun 2e after separation improves; read the numbered list.

## 10. Working agreements (from the user)

- Judge on real photos only. No more digital benchmarks.
- No terminal shenanigans: edit with the editor tools, no Bash writes, no temp files; the user does all downloads and installs.
- Experiments belong in notebooks.
- Ponytail style: smallest change, say what was skipped.

## 11. Honest limits of this document

- No code was run this session after the last edits (2e numbering, 2f variants, `0f5ff2aa`); they are untested.
- Memory crash cause is a hypothesis.
- Sections marked **(thin)** were not re-verified against the files.
