# Reference: folders

Folders marked **git** are in the repository. The rest are downloaded ([Get the data](../how-to/get-data.md)) or made by the notebooks.

| Folder | Where | Size | What | Made by / used in |
|---|---|---|---|---|
| `data/` | git | small | labels, boxes, gallery ([details](data-files.md)) | many |
| `labeling/labels/` | git | small | hand-drawn boxes | `autolabel` |
| `notebooks/`, `scripts/`, `docs/` | git | small | code and docs | |
| `models/` | download (`shape_embed.pt` in git) | 0.5 GB | trained weights | `compare`, `lookup`, `renders` |
| `dataset/` | download | 12 GB | the photos, `dataset/<part>_<colour ID>_<element>/IMG_xxxx.JPG` (hardlinks to the originals) | `scripts/build_dataset.py`; most notebooks |
| `runs/` | download | 0.2 GB | YOLO training runs; the brick finder is `runs/detect/runs/box/weights/best.pt` | `autolabel`, `compare`, `synth`, `tower_seg` |
| `labeling/images/` | download | 1.5 GB | photos picked for hand boxing | `scripts/pick_label_subset.py` |
| `labeling/review/` | download | 0.4 GB | auto-boxed photos flagged for review | `autolabel` |
| `cache1024/` | download / rebuilt | 0.8 GB | 1024×768 copies of every photo | `compare` |
| `crops224/` | download / rebuilt | 71 MB | 224×224 brick crop of every photo | `compare` → `lookup`, `blur` |
| `yolo_box/`, `yolo41/`, `yolo_tower/` | download / rebuilt | 2.5 GB | YOLO-format training sets | `autolabel`, `compare`, `tower_seg` |
| `cutouts/`, `synth_v2/` | download / rebuilt | 0.5 GB | cut-out bricks, synthetic scenes | `synth` |
| `renders/` | download / rebuilt | 0.1 GB | 3D renders of every piece | `renders` |
| `ldraw/` | download, or library.ldraw.org | 0.6 GB | LDraw parts library | `ldraw_render.py` |
| `real/` | download | 16 MB | real tower photos (the test images) | `test_real`, `pipeline`, `synth`, `tower_seg` |
| `output/`, `eda/` | rebuilt | small | saved figures and stats | various |
| `photos/` | local | | photos added through `app.py` | `app.py` |
| `sam_b.pt`, `yolo11s.pt` | auto-downloaded | 0.4 GB | stock Ultralytics weights | `test_real`, YOLO notebooks |
