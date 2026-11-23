# How to label photos

Labelling uses [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) in its own environment:

```bash
uv venv .venv-label && uv pip install --python .venv-label x-anylabeling-cvhub[cpu]
```

## Single-brick photos (bounding boxes)

1. `python scripts/pick_label_subset.py`: links 20 photos per piece into `labeling/images/`.
2. `scripts/label.sh`: draw one box per brick, label `brick`. Saved to `labeling/labels/<element>_<IMG>.json`.
3. Run `notebooks/autolabel.ipynb`: trains a detector on your boxes and boxes every other photo.
4. `scripts/review.sh`: fix the flagged photos in `labeling/review/`.
5. Rerun the last cell of `autolabel.ipynb` to write `data/boxes.csv`.

## Tower photos (box + piece class)

1. Run the last labelling cell of `notebooks/test_real.ipynb` once: it makes `labeling/thumbs/` and `labeling/brick_sheet.png`.
2. Put the tower photos in `labeling/towers/`.
3. `scripts/label_towers.sh`: drag a box, click the piece's picture on the right.
   - Right / Left: next / previous photo. Delete: remove the selected box.
   - Boxes without a class are not saved.
4. Labels go to `labeling/tower_labels/<photo>.json` (same format as X-AnyLabeling). Class numbers are the lines of `labeling/tower_classes.txt`.

## Hide bad photos (without deleting)

`python scripts/exclude_photos.py`: click a photo to toggle it (red = excluded). Saved in `data/excluded.csv`; `lookup.ipynb` leaves these out. Blurriest photos are shown first if `data/blur.csv` exists.
