# Tutorial: recognise your first brick

About 10 minutes. You will install the project, download the minimum data, and name a brick from a photo.

## 1. Install

You need Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is optional for this tutorial.

```bash
git clone https://github.com/Fusion-Rebooted/Fusion-Recognition.git
cd Fusion-Recognition
uv venv --python 3.12 .venv && uv pip install -r requirements.txt
```

The repo already contains the shape model (`models/shape_embed.pt`) and the gallery (`data/gallery.npz`).

## 2. Get the brick finder and a few photos

The box detector lives in `runs/` and the photos in `dataset/`; both are downloaded separately ([Get the data](how-to/get-data.md)). For this tutorial you only need:

```bash
wget -r -np -nH --cut-dirs=1 -c -R "index.html*" <DATA_URL>/runs/detect/runs/box/
wget -r -np -nH --cut-dirs=1 -c -R "index.html*" <DATA_URL>/dataset/3005_5_4113915/
```

No access? Use your own photo of one brick lying on a plain table instead.

## 3. Recognise it

```bash
.venv/bin/python
```
```python
import lego_lookup as L, numpy as np

emb = L.load_embedder()        # the shape fingerprint model
gal = L.load_gallery()         # the reference photos: 840 entries, 20 per piece

# find the brick in the photo and cut out a 224x224 square around it
[(path, crop)] = L.crops_from_photos(['dataset/3005_5_4113915/IMG_1074.JPG'])

vec = L.embed(emb, [crop])                 # shape fingerprint: a 512-number vector
lab = np.array([L.brick_lab(crop)])        # colour: median Lab colour of the brick pixels

shape, sim, color, dist = L.recognize(gal, vec, lab, closed_set=True)[0]
print(shape, color, round(sim, 3))
```

Output:

```
3005 5 0.997
```

## 4. Read the answer

| Value | Meaning |
|---|---|
| `3005` | part number: Brick 1 x 1 (look it up in `data/element_map.csv`) |
| `5` | LEGO colour ID: Brick Yellow, called "Tan" by BrickLink (`data/lego_colors.csv`) |
| `0.997` | similarity to the closest reference photo. Above ~0.93 is a confident match; below, `recognize` answers `unknown` |

## 5. What just happened

1. A YOLO detector found the brick and cropped it.
2. A ResNet18 that only sees **grayscale** turned the crop into a fingerprint, so it judges shape, not colour.
3. The fingerprint was compared with every reference photo in the gallery; the 5 closest voted on the shape.
4. Separately, the brick's colour was compared with the reference colours; the 5 closest voted on the colour.

Why it is built this way: [how recognition works](explanation/how-recognition-works.md).

## Next

- Teach it a piece it doesn't know: [Add a piece](how-to/add-piece.md)
- See how accurate it is: `notebooks/lookup.ipynb` and the [model card](model-card.md)
