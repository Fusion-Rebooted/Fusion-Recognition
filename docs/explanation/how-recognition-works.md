# How recognition works

## The problem

Set 21205 has 45 kinds of piece (element IDs): 25 shapes in 8 colours. The goal is to name each piece in a photo. Later the set may grow, so new pieces must be easy to add.

## First attempt: a normal classifier

`notebooks/compare.ipynb` trained three standard models (ResNet18 classifier, YOLO, Faster R-CNN) with one output per element. On single-brick photos all three hit **99.1%**. So why not stop there?

Because a classifier has its classes baked in. A new piece means new output neurons and retraining on every photo. And colour and shape are tangled together: the model has to learn "brick 1×1" separately for each colour.

## The design used: look it up

Split the question in two and answer each by comparison with known examples.

**Shape: a colour-blind fingerprint.**
A ResNet18 is trained to tell the 25 *shapes* apart, but it only ever sees **grayscale** crops, so it cannot use colour. After training its last layer is removed; what remains turns any crop into a 512-number vector, a *fingerprint*. Crops of the same shape give vectors pointing the same way (high cosine similarity), even for shapes it never trained on.

**Colour: just measure it.**
The colour is the median Lab value of the brick pixels (GrabCut separates brick from table). Lab is used because distances in it roughly match how different colours look.

**The gallery: the memory.**
`data/gallery.npz` stores fingerprint + colour + label for ~20 reference photos per piece. To recognise a brick, compute its fingerprint and colour, find the 5 closest gallery entries for each, and let them vote.

**Adding a piece** = putting its ~20 photos in the gallery. The fingerprint model never changes.

## Does it work?

From `notebooks/lookup.ipynb`, test photos never seen in training:

| Setting | Shape | Colour | Both |
|---|---|---|---|
| Plain ImageNet fingerprint (no training) | 22% | 96.7% | 21.7% |
| Trained colour-blind fingerprint, open set | 92.8% | 96.7% | 90.6% |
| Same, closed set | **98.3%** | **99.7%** | **98.3%** |
| 3 shapes it **never trained on** (only in the gallery) | 81% | | |

Training the fingerprint is what makes it work (22% → 93%). It generalises to new shapes reasonably (81%) without retraining.

## Open vs closed set

- **Open set** (default): answer `unknown` if the best similarity is below 0.977. Catches 100% of unknown pieces in the test, at the cost of rejecting ~5% of known ones.
- **Closed set**: we know only these 45 pieces exist, so use a looser cut-off (0.93) and only allow colours seen with the predicted shape. More accurate, but would refuse a genuinely new colour.

## Known weaknesses

- **3040b vs 4286** (two reddish-brown slopes) are confused from some angles; more reference photos don't fix it.
- The gallery holds **table photos**. Bricks seen inside a tower look different (only one face, neighbours around), so similarity drops below the confident range. See [why towers fail](why-towers-fail.md).

Decisions behind this: [0001 gallery lookup](decisions/0001-gallery-lookup.md), [0002 colour-blind fingerprint](decisions/0002-colour-blind-fingerprint.md).
