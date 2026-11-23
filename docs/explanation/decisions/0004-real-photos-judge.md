# 0004: Real photos are the only judge for towers, not renders

**Status:** Accepted

## Context
Rendered 3D towers (`ldraw_render.py`, `towers.ipynb`, `tower_seg.ipynb`) give exact labels for free, so it is tempting to score tower methods on them. But renders look different from photos: the lookup gets ~20% of shapes right on renders, and published work on synthetic-only LEGO training reports poor transfer (F1 ~0.4).

## Decision
Renders may be used for **training**, never to decide whether a method is good. Methods for towers are judged on the real photos in `real/`, by eye until a labelled set of real tower photos exists.

## Consequences
- ✅ No false confidence from scores on renders.
- ❌ Judging by eye is slow and not a number. Labelling real tower photos (`scripts/label_towers.sh`) is the fix.
- The renderer and `tower_seg.ipynb` stay in the repo as training-data tools.
