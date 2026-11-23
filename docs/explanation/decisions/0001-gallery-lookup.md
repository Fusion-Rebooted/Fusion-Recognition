# 0001: Recognise by gallery lookup, not a fixed classifier

**Status:** Accepted

## Context
Classifiers (CNN, YOLO, Faster R-CNN) reached 99.1% on single bricks (`compare.ipynb`), but each has the 41 classes built in. The set will grow, and retraining for every new piece is slow and needs all old photos.

## Decision
Name bricks by nearest-neighbour lookup in a gallery of reference photos (`data/gallery.npz`), using a learned shape fingerprint plus a measured colour. Implemented in `lego_lookup.py`.

## Consequences
- ✅ New piece or colour = ~20 photos, no training (`scripts/add_piece.py`, `app.py`).
- ✅ Unknown pieces can be rejected (`unknown`) instead of misnamed.
- ✅ Accuracy on the known set matches the classifiers (98.3% closed set).
- ❌ Shapes the fingerprint never trained on are weaker (81%).
- ❌ Retraining the fingerprint means rebuilding the whole gallery.
