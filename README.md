# LEGO Fusion Battle Towers: brick recognition

Uni project: find and identify the bricks of the LEGO Fusion Battle Towers set in photos.
Three models are compared: a CNN (ResNet18), Faster R-CNN and YOLO. Then the best one is pushed toward real built towers.

## Data
- 6491 photos of 41 brick types (one brick per photo, on a table). Not in the repo (12 GB).
- `data/element_map.csv`: LEGO element ID → part number, name, color.
- `data/ranges.csv`: which photo numbers (IMG_xxxx) show which element.
- `data/labels.csv`: every photo with its element, shape (part_num) and official LEGO color ID.
- `data/boxes.csv`: one bounding box per photo (809 drawn by hand, the rest by a model and then reviewed).
- `labeling/labels/`: the hand-drawn boxes (X-AnyLabeling JSON).
- `data/gallery.npz`: lookup reference sheet (shape fingerprint + Lab color per reference photo), used by `lego_lookup.py`
- `data/exif_orientation_before.csv`: original EXIF orientation tags, which were stripped from the photos.

## Notebooks (`notebooks/`, run in this order)
| Notebook | What it does |
|---|---|
| `eda.ipynb` | Class balance, brightness/blur, color separability |
| `autolabel.ipynb` | Trains a 1-class brick detector on the hand boxes, boxes the remaining photos, flags doubtful ones for review |
| `compare.ipynb` | Trains and compares CNN, Faster R-CNN and YOLO on the same 70/15/15 split |
| `test_real.ipynb` | Runs the models on real tower photos (`real/`), tiling, and why they fail |
| `synth.ipynb` | Builds synthetic tower scenes from brick cutouts and trains YOLO on them |
| `lookup.ipynb` | Shape + color by gallery lookup: trains the color-blind fingerprint, builds `data/gallery.npz`, tests new/unknown pieces |

## Renders (LDraw)
`ldraw_render.py` draws any LDraw part in any color from any angle on the GPU (about 1 ms per image), headless. Needs the LDraw library in `ldraw/`:
```
curl -LO https://library.ldraw.org/library/updates/complete.zip && unzip complete.zip && rm complete.zip
```
`data/lego_colors.csv` maps LEGO color IDs to LDraw color codes.

## App
`.venv/bin/python app.py` → http://127.0.0.1:7860. Upload ~20 photos of one new brick; Brickognize (brickognize.com) suggests the part number and color, you confirm, and it is added to `data/gallery.npz`. Photos are kept in `photos/<part>_<color ID>/`. Colors live in `data/lego_colors.csv`.

## Scripts
- `scripts/add_piece.py`: teach a new piece or color, no training: `python scripts/add_piece.py photos/x/ --shape 3001 --color 21`
- `scripts/build_dataset.py`: hardlinks photos into `dataset/<part>_<color ID>_<element>/` and writes `data/labels.csv`
- `scripts/pick_label_subset.py`: picks 20 photos per brick for hand-labeling
- `scripts/label.sh`, `scripts/review.sh`: open X-AnyLabeling on the labeling and review sets

Run scripts from the repo root. Notebooks switch to the repo root themselves.

## Setup
```
uv venv --python 3.12 .venv && uv pip install -r requirements.txt   # models
uv venv .venv-label && uv pip install --python .venv-label x-anylabeling-cvhub[cpu]   # labeling tool
```
