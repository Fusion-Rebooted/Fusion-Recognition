#!/bin/sh
cd "$(dirname "$0")/.."   # run from repo root wherever called from
# Tower labeler: drag a box, click the piece's picture. Needs labeling/thumbs/ (last cell of notebooks/test_real.ipynb).
# Saves the same JSON as X-AnyLabeling, so you can still open labeling/tower_labels/ there to fix boxes:
#   .venv-label/bin/xanylabeling --filename labeling/towers --output labeling/tower_labels --labels labeling/tower_classes.txt --nodata --autosave --no-auto-update-check
exec .venv/bin/python scripts/tower_labeler.py
