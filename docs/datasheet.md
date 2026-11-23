# Datasheet: LEGO 21205 single-brick photos

Format after Gebru et al., *Datasheets for Datasets* (2021).

## Motivation
- **Why was it made?** To train and test recognition of the pieces of LEGO set 21205 (Fusion Battle Towers), as the first step toward reading built towers.
- **By whom?** The project team (university project), 2026.

## Composition
- **What is in it?** 6492 photos, each of **one** LEGO piece lying on a table, plus a bounding box per photo.
- **Classes:** 41 element IDs photographed (25 shapes × 8 colours) of the set's 45. **Missing:** 3069b, 63864 and 6178pr0002 (Light Bluish Gray) have no photos; 4490 has only 71.
- **Per class:** 71–325 photos (median 128).
- **Image:** 4032×3024 JPEG, one phone camera; EXIF rotation stripped (originals listed in `data/exif_orientation_before.csv`).
- **Brick size:** median 12% of the frame (5th–95th percentile: 1.4–31%).
- **Labels:** `data/labels.csv` (piece per photo), `data/boxes.csv` (box per photo).
- **Separate test images:** a handful of real tower photos in `real/`, unlabelled.
- **Personal data:** none.

## Collection
- Each piece was photographed in one continuous session from many angles (`data/ranges.csv` lists the photo numbers per piece).
- Same table, indoor lighting. Brightness is very even (5th percentile 140, median 146 on 0–255).
- Photos are sorted by `scripts/build_dataset.py` into `dataset/<part>_<colour ID>_<element>/` as hardlinks; originals untouched.

## Labelling
- 809 boxes drawn by hand in X-AnyLabeling (20 per piece).
- A one-class YOLO detector trained on those boxed the remaining 5683 photos; 209 doubtful ones (mostly the two-part window 4541873) were checked by hand. The `source` column says which is which.
- The window frame (4530590) and doors (4541873) are labelled as one box.

## Known issues
- **Fixed:** IMG 5861–6020 were filed as 4460b but show 2449; moved and relabelled (`blur.ipynb`).
- **Blur:** 62 photos below half their piece's median sharpness, listed in `data/blurry.csv` and excluded via `data/excluded.csv`.
- **One condition only:** one camera, one table, one light, so models trained on it do not transfer to other settings, especially towers.
- The folder audit in `blur.ipynb` flags a few IMG ranges worth checking by eye.

## Uses
- **Suitable:** recognising single pieces of this set on a plain surface; building a lookup gallery; benchmarking classifiers.
- **Not suitable on its own:** recognising pieces in built models, other sets, or other photo conditions.

## Distribution
- Not public. Hosted on the team server ([Get the data](how-to/get-data.md)).
- LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project. Part numbers are from BrickLink/Rebrickable; 3D parts from the LDraw library (CC BY 2.0).
