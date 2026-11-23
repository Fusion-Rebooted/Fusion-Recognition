# 0003: Split train/val/test in time inside each photo session

**Status:** Accepted

## Context
Each piece was shot in one continuous session (`data/ranges.csv`); consecutive frames are near-duplicates. A random split would put almost identical photos in train and test and inflate the scores.

## Decision
Inside every session, sort by photo number: first 70% → train, next 15% → val, last 15% → test. All notebooks use the same rule, so every result is on the same test photos.

## Consequences
- ✅ Test photos are never near-copies of training photos; results across notebooks are comparable.
- ✅ Every class is in every split.
- ❌ Still same camera, table and lighting in all splits, so the scores say nothing about other conditions (e.g. towers).
