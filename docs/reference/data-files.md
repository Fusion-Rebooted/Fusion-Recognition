# Reference: data files

All in `data/`, all in git. Paths inside the files are relative to the repo root. Rows as of the last commit.

## `element_map.csv`: the pieces of set 21205 (45 rows)

| Column | Example | Meaning |
|---|---|---|
| `element_id` | `4113915` | LEGO element ID = one shape in one colour. The unit everything is labelled by |
| `part_num` | `3005` | Shape, any colour |
| `part_name` | `Brick 1 x 1` | Human name |
| `color` | `Tan` | BrickLink colour name |

## `lego_colors.csv`: colours of the set (8 rows)

| Column | Example | Meaning |
|---|---|---|
| `lego_id` | `5` | Official LEGO colour ID, used everywhere as "color" |
| `lego_name` | `Brick Yellow` | Official LEGO name |
| `bricklink_name` | `Tan` | BrickLink name (used by `element_map.csv` and Brickognize) |
| `ldraw_code` | `19` | LDraw colour code (renders) |
| `rgb` | `#D7BA8C` | Display colour |

## `labels.csv`: every photo (6492 rows)

| Column | Example |
|---|---|
| `path` | `dataset/4286_192_4211197/IMG_0001.JPG` |
| `element_id` | `4211197` |
| `part_num` | `4286` |
| `part_name` | `Brick Sloped 33° 3 x 1` |
| `lego_color_id` | `192` |
| `lego_color_name` | `Reddish Brown` |

Written by `scripts/build_dataset.py`.

## `boxes.csv`: one brick box per photo (6491 rows)

| Column | Meaning |
|---|---|
| `path` | photo |
| `element_id` | piece in the box |
| `x0, y0, x1, y1` | box corners in pixels of the full 4032×3024 photo |
| `source` | `manual` (hand-drawn), `auto 0.87` (detector + confidence), `reviewed` (detector, corrected by hand) |

Written by `notebooks/autolabel.ipynb`.

## `ranges.csv`: photo sessions (45 rows)

`start, end, element_id`: photos IMG_`start`..`end` show this element. Used to split each session in time into train / val / test.

## `blur.csv`, `blurry.csv`, `excluded.csv`

| File | Columns | Meaning |
|---|---|---|
| `blur.csv` | `path, element_id, blur` | Sharpness (Laplacian variance inside the box). Delete to recompute |
| `blurry.csv` | `path, element_id, rel` | Photos below the cut-off; `rel` = sharpness / piece median |
| `excluded.csv` | `path` | Photos hidden from training by `scripts/exclude_photos.py` |

## `exif_orientation_before.csv`

`path, orientation`: the EXIF rotation tag each photo had before it was stripped (all photos are now stored landscape).

## `gallery.npz`: the lookup gallery (840 entries)

A NumPy archive, one row per reference photo. Load with `L.load_gallery()`.

| Key | Shape | Meaning |
|---|---|---|
| `vec` | (N, 512) float32 | shape fingerprint, unit length |
| `lab` | (N, 3) float32 | median Lab colour of the brick |
| `shape` | (N,) str | part number, or `''` for a colour-only entry |
| `color` | (N,) str | LEGO colour ID, or `''` for a shape-only entry |
| `src` | (N,) str | the photo it came from |

## `labeling/`

| File | Meaning |
|---|---|
| `labels/<element>_<IMG>.json` | 809 hand-drawn boxes (X-AnyLabeling format) |
| `classes.txt` | label list for single-brick boxing (`brick`) |
| `tower_classes.txt` | 45 tower classes, `NN_<part>_<colour>`, same order as `element_map.csv` |
| `brick_sheet.png` | picture of every class number, for labelling |
