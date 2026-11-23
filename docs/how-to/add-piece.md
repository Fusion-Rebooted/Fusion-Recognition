# How to add a new piece or colour

No training needed. You add reference photos to the gallery (`data/gallery.npz`).

## 1. Take the photos

- **~20 photos** of the piece **alone**, on a plain table.
- Different sides and angles: top, sides, upside down, rotated.
- Sharp, evenly lit. The piece should fill roughly a tenth to a third of the frame.
- Put them in one folder, e.g. `photos/3001_21/`.

## 2. Find the IDs

- **Part number** (the shape): on [BrickLink](https://www.bricklink.com) or [Rebrickable](https://rebrickable.com), e.g. `3001` = Brick 2 x 4.
- **LEGO colour ID**: the `lego_id` column of `data/lego_colors.csv`, e.g. `21` = Bright Red. A colour not in that file must be added there first.

## 3a. Command line

```bash
python scripts/add_piece.py photos/3001_21/ --shape 3001 --color 21
```

Prints `added N of M photos`. Photos where no brick was found are skipped.

## 3b. Web app (if you don't know the part number)

```bash
.venv/bin/python app.py      # opens http://127.0.0.1:7860
```

Upload the photos; [Brickognize](https://brickognize.com) suggests part and colour, you confirm, the app adds them.

## 3c. From Python

```python
import glob, lego_lookup as L
L.add_photos(glob.glob('photos/3001_21/*.jpg'), shape='3001', color='21')
```

## Special cases

| Case | What to pass |
|---|---|
| Known shape, **new colour** | same command, the new `--color` |
| Shape only (e.g. renders, colour unknown) | `--shape` only |
| Colour only | `--color` only |
| Re-adding the same photo | it replaces the old entry, no duplicates |

## 4. Check and share

Recognise one of the new photos ([tutorial](../tutorial.md) step 3); it should come back with your part and colour.
Then commit `data/gallery.npz` so others get the piece too.

**Note:** `closed_set=True` only answers colours already seen with that shape. While you are still adding new colours, call `recognize` with `closed_set=False`.
