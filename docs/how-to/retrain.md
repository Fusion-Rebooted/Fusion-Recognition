# How to retrain the models and rebuild the gallery

Each trained file is skipped if it already exists. **Delete the file to retrain it.** Needs an NVIDIA GPU (developed on an RTX 3060, 12 GB) and the data from [Get the data](get-data.md).

| Output | Delete it, then run | Time |
|---|---|---|
| `runs/detect/runs/box/weights/best.pt` (brick finder) | `notebooks/autolabel.ipynb` | ~20 min |
| `data/boxes.csv` | `autolabel.ipynb`, review in `scripts/review.sh`, rerun its last cell | review by hand |
| `cache1024/`, `crops224/` | `compare.ipynb`, first cells | a few min |
| `models/cnn_resnet18.pt`, `models/frcnn_r50.pt`, YOLO `cmp_yolo` | `compare.ipynb` | ~1.5 h together |
| `models/shape_embed.pt` (shape fingerprint) | `lookup.ipynb` section 1 | ~3 min |
| `data/gallery.npz` | `lookup.ipynb` section 2 (rebuilt on every run) | ~2 min |
| `models/shape_embed_render.pt` | `renders.ipynb` | ~5 min |
| YOLO `synth_yolo`, `synth_yolo_v2` | `synth.ipynb` | ~1.5 h |
| YOLO `tower_seg` | `tower_seg.ipynb` | depends on GPU |

## Order matters

```
build_dataset.py → autolabel → compare (makes crops224/) → lookup → everything else
```

`lookup.ipynb` reads `crops224/`, which `compare.ipynb` makes, so run `compare` at least up to its cache cell first.

## Important

- **Retraining `shape_embed.pt` invalidates the gallery.** Every fingerprint in `data/gallery.npz` comes from the old model, so rebuild the gallery (and re-add any pieces you added by hand) afterwards.
- **Changing the classes** (e.g. after fixing a mislabelled folder) makes the old CNN / Faster R-CNN / YOLO weights stale; `test_real.ipynb` will report them as "not loaded".
- The train/val/test split is fixed by photo session (`data/ranges.csv`), so results stay comparable between runs. See [decision 0003](../explanation/decisions/0003-split-by-session.md).
