# Why real towers are hard

Every model here scores 98–99% on bricks photographed alone. On a photo of a built tower, no method so far names bricks reliably. This page explains why and what has been tried.

## The gap between training and use

| | Training photos | Tower photos |
|---|---|---|
| Bricks per photo | 1 | 20–50 |
| Brick size in frame | ~12% (median) | a few % |
| Background | plain table | desk, keyboard, printed base plate |
| What is visible | the whole brick, any side | one face; top, studs and sides hidden |
| Neighbours | none | touching, often the same colour |

A model learns what it is shown. It never saw small, touching, half-hidden bricks, and never saw a busy background, so base-plate studs and desk corners get named as bricks with high confidence.

## What was tried

| Approach | Notebook | Outcome |
|---|---|---|
| Run detectors on the whole photo | `test_real` | ~5 boxes, mostly junk |
| Zoomed tiles + CNN confidence filter | `test_real` | ~13 boxes, about half real bricks |
| SAM "segment everything" + lookup | `test_real` | good outlines, but same-coloured neighbours merge; no confident names (similarity ≤ 0.91) |
| Train YOLO on pasted fake towers | `synth` | judged by eye on real photos |
| Render 3D towers, test the lookup on them | `towers` | shape only ~20% right; not a resolution problem (1600 px barely helps) |
| Train a segmenter on rendered towers | `tower_seg` | separates bricks better than SAM (median similarity 0.85 vs 0.80), naming still uncertain |
| Classic image processing: tower mask → watershed | `pipeline` | tower mask OK (IoU ~0.7, 0.9 with BiRefNet); brick split weak (PQ ~0.3, ~5 merges per tower) |

## The pattern

Two separate problems:

1. **Where** is each brick? Getting better: rendered-tower segmentation and BiRefNet + watershed both find most bricks.
2. **What** is each brick? Still the bottleneck. The gallery holds whole bricks on a table; a tower shows one face of a brick among neighbours, so matches stay below the confident level.

## What would likely help

- Gallery photos that look like towers: bricks photographed **inside** small builds, front faces only.
- A labelled set of real tower photos (the tower labeller exists: `scripts/label_towers.sh`) to measure progress with numbers instead of by eye.
- Controlled photos: plain background, camera on a stand, which `pipeline.ipynb` is built for (`real/plain/`).

Full history: [DEVELOPMENT_LOG.md](../DEVELOPMENT_LOG.md).
