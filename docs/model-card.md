# Model card: brick lookup (`shape_embed.pt` + `gallery.npz`)

Format after Mitchell et al., *Model Cards for Model Reporting* (2019).

## Details
- **What:** a shape fingerprint (ResNet18, ImageNet-pretrained, fine-tuned on grayscale crops of 23 shapes, last layer removed → 512-d vector) plus a gallery of reference photos with Lab colours. Recognition = k-nearest-neighbour vote (k = 5).
- **Files:** `models/shape_embed.pt` (43 MB), `data/gallery.npz` (840 entries, 20 per element). Code: `lego_lookup.py`.
- **Brick finder used with it:** one-class YOLO11s, `runs/detect/runs/box/weights/best.pt`.
- **Trained:** 10 epochs, AdamW, OneCycle, label smoothing; ~3 min on an RTX 3060 (`notebooks/lookup.ipynb`).

## Intended use
- **For:** naming a single LEGO piece of set 21205 on a plain surface; adding new pieces by photo.
- **Not for:** pieces in built towers (see below), crowded scenes, other photo conditions, safety-relevant decisions.

## Data
- **Training:** the train split (70%) of the [single-brick dataset](datasheet.md), 23 of 26 shapes. 3 shapes held out to test generalisation.
- **Gallery:** 20 evenly spaced train photos per element, including the 3 held-out shapes.
- **Test:** 940 photos, the last 15% of each photo session ([why](explanation/decisions/0003-split-by-session.md)).

## Results (test split)

| Mode | Shape | Colour | Both |
|---|---|---|---|
| Open set (cut-off 0.977) | 92.8% | 96.7% | 90.6% |
| Closed set (cut-off 0.93, colours limited per shape) | 98.3% | 99.7% | 98.3% |
| Shapes never trained on, open set | 81.0% | | |
| *Baseline: same pipeline, untrained ImageNet fingerprint* | *22.0%* | *96.7%* | *21.7%* |

Unknown pieces (open set): at the 0.977 cut-off 95% of known pieces are kept and 100% of the held-out pieces are rejected.

For comparison, fixed classifiers on the same split (`compare.ipynb`): ResNet18 on the true crop, YOLO11s, Faster R-CNN all 99.1% element, 100% colour.

## Limitations
- **Real towers:** the lookup does not name bricks in tower photos reliably; best matches stay below 0.93. On rendered towers shape accuracy is ~20%. See [why towers fail](explanation/why-towers-fail.md).
- **Confusion:** 3040b vs 4286 (reddish-brown slopes), 9 test errors that more reference photos do not fix.
- **Colour** relies on a GrabCut mask of the brick; shadows and neighbours shift it. `COLOR_MAX_DIST` is not calibrated, so colour is never `unknown`.
- **One condition:** one camera, table and light; expect lower accuracy elsewhere.
- Retraining the fingerprint invalidates the gallery; rebuild it afterwards.

## Ethics
No people or personal data. Risk is limited to misidentified toy parts.
