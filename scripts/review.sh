#!/bin/sh
cd "$(dirname "$0")/.."   # run from repo root wherever called from
# Opens auto-labeled photos flagged for review (written by autolabel.ipynb step 5).
cd "$(dirname "$0")"
exec .venv-label/bin/xanylabeling --filename labeling/review/images --output labeling/review/labels --labels brick,4530590,4541873 --nodata --autosave --no-auto-update-check
