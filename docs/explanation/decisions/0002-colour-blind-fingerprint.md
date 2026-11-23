# 0002: Shape fingerprint sees grayscale only; colour measured separately

**Status:** Accepted

## Context
The same shape comes in several colours. A model that sees colour learns each shape+colour pair separately, needs photos of every combination, and can't handle a known shape in a new colour.

## Decision
The ResNet18 fingerprint is trained and run on **grayscale** crops (`L.gray3`), with colour jitter limited to brightness/contrast. Colour is the median Lab value of the brick pixels (`L.brick_lab`), matched separately.

## Consequences
- ✅ One shape = one class, whatever the colour; a new colour on a known shape needs no new shape data.
- ✅ Colour is cheap and accurate on table photos (96.7%, 99.7% in closed set).
- ❌ Colour depends on a clean brick mask; on towers (shadows, neighbours) it drops.
- ❌ Shapes that differ only in colour-coded detail can't be told apart by shape alone (not an issue in this set).
