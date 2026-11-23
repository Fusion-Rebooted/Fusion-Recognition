# Reference: `lego_lookup.py`

The recogniser. Import from the repo root: `import lego_lookup as L`.
Images are **RGB uint8** numpy arrays; crops are **224×224**.

## Constants

| Name | Value | Meaning |
|---|---|---|
| `DEV` | `'cuda'` or `'cpu'` | device, chosen automatically |
| `GALLERY` | `data/gallery.npz` | default gallery file |
| `EMBED_PT` | `models/shape_embed.pt` | shape fingerprint weights |
| `BOX_PT` | `runs/detect/runs/box/weights/best.pt` | brick finder (YOLO) |
| `SHAPE_MIN_SIM` | 0.977 | open set: below this similarity → `unknown` (keeps 95% of known pieces) |
| `CLOSED_MIN_SIM` | 0.93 | closed set cut-off |
| `COLOR_MAX_DIST` | 1e9 | colour is never `unknown` yet (not calibrated) |

## Models and features

| Function | In → out |
|---|---|
| `load_embedder(path=EMBED_PT)` | → ResNet18 without its last layer. Falls back to ImageNet weights if the file is missing (much worse) |
| `embed(model, crops, bs=128)` | list of 224 crops → (N, 512) unit vectors. Grayscale, averaged with the mirror image |
| `gray3(rgb)` | RGB → grayscale copied into 3 channels (what the model sees) |
| `brick_lab(crop)` | 224 crop → median Lab colour (3,) of the brick pixels (GrabCut mask) |

## Finding bricks

| Function | In → out |
|---|---|
| `crops_from_photos(paths, detector=None)` | single-brick photo paths → list of `(path, crop)`; the most confident box per photo, photos with no brick are skipped |
| `crop_box(rgb, x0, y0, x1, y1)` | image + box → square 224 crop with 15% margin (the training format) |

## Gallery

| Function | What |
|---|---|
| `load_gallery(path=GALLERY)` | → dict `vec, lab, shape, color, src` ([format](data-files.md#gallerynpz-the-lookup-gallery-840-entries)); empty if the file doesn't exist |
| `save_gallery(g, path=GALLERY)` | writes it |
| `add_entries(g, vecs, labs, shape, color, srcs)` | → new gallery dict with rows appended (does not save) |
| `add_photos(paths, shape='', color='', model=None, path=GALLERY)` | **the way to add a piece**: finds, crops, embeds and saves. Returns how many photos were added |

## Recognition

```python
recognize(g, vecs, labs, k=5, shape_min_sim=None, color_max_dist=None, closed_set=False)
```

→ one `(shape, shape_sim, color, color_dist)` per input.

- **Shape**: the k most similar gallery entries vote, weighted by similarity. `shape_sim` is the best single similarity. Below the cut-off → `'unknown'`.
- **Colour**: the k nearest gallery colours (Lab distance) vote.
- `closed_set=True`: cut-off 0.93 instead of 0.977, and colour may only be one already seen with the predicted shape. More accurate on the known set, but rejects a new colour on a known shape.
- `shape_min_sim=-1`: never answer `unknown`, always give the best guess.
- Entries with `shape == ''` don't vote on shape; entries with `color == ''` don't vote on colour.
