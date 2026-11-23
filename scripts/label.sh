#!/bin/sh
cd "$(dirname "$0")/.."   # run from repo root wherever called from
# Opens the boxing subset in X-AnyLabeling. Boxes saved as JSON next to labeling/labels/.
cd "$(dirname "$0")"
exec .venv-label/bin/xanylabeling --filename labeling/images --output labeling/labels --labels brick --nodata --autosave --no-auto-update-check
