# Brick recognition by lookup: a photo of a brick -> (shape part_num, LEGO color ID), no retraining for new pieces.
# The gallery (data/gallery.npz) holds one entry per reference photo: shape fingerprint + Lab color + labels.
# New piece / new color = add reference photos to the gallery (add_photos), nothing else.
import os, cv2, numpy as np, torch, torchvision

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GALLERY = 'data/gallery.npz'
EMBED_PT = 'models/shape_embed.pt'   # color-blind ResNet18 trained on shapes; falls back to plain ImageNet weights
BOX_PT = 'runs/detect/runs/box/weights/best.pt'
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def load_embedder(path=EMBED_PT):
    m = torchvision.models.resnet18(weights=None if os.path.exists(path) else 'DEFAULT')
    m.fc = torch.nn.Identity()
    if os.path.exists(path): m.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    return m.to(DEV).eval()

def gray3(rgb):   # color-blind input: the shape fingerprint must not see color
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return np.dstack([g, g, g])

@torch.no_grad()
def embed(model, crops, bs=128):
    """crops: list of 224x224 RGB uint8 -> (N, 512) unit vectors (averaged with the mirror image)."""
    out = []
    for i in range(0, len(crops), bs):
        x = torch.stack([(torch.from_numpy(gray3(c)).permute(2, 0, 1).float() / 255 - MEAN) / STD for c in crops[i:i + bs]]).to(DEV)
        v = model(x) + model(x.flip(3))
        out.append(torch.nn.functional.normalize(v, dim=1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 512), np.float32)

def brick_lab(crop):
    """Median Lab color of the brick pixels in a 224 crop (brick fills the middle ~87%, see crop_box)."""
    m = 15   # crop_box adds a 15% margin -> brick starts ~15 px in
    mask = np.zeros(crop.shape[:2], np.uint8)
    try: cv2.grabCut(crop[..., ::-1].copy(), mask, (m, m, 224 - 2 * m, 224 - 2 * m), np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error: pass
    fg = np.isin(mask, (1, 3))
    if fg.sum() < 200: fg = np.zeros_like(fg); fg[60:164, 60:164] = True   # GrabCut failed: use the centre
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)[fg].astype(np.float32)
    return np.median(lab, 0)

def crop_box(rgb, x0, y0, x1, y1):
    """Square crop around a box + 15% margin, 224x224 (same as crops224/ used in training)."""
    s = max(x1 - x0, y1 - y0) * 1.15
    return cv2.resize(cv2.getRectSubPix(rgb, (int(s), int(s)), ((x0 + x1) / 2, (y0 + y1) / 2)), (224, 224), interpolation=cv2.INTER_AREA)

def crops_from_photos(paths, detector=None):
    """Single-brick photos -> one crop each (the most confident box). Photos with no brick found are skipped."""
    from ultralytics import YOLO
    det = detector or YOLO(BOX_PT)
    out = []
    for p in paths:
        bgr = cv2.imread(p)
        if bgr is None: continue
        f = 1024 / max(bgr.shape[:2]); small = cv2.resize(bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        b = det.predict(small, imgsz=1024, conf=0.25, verbose=False)[0].boxes
        if len(b): out.append((p, crop_box(small[..., ::-1].copy(), *b.xyxy[b.conf.argmax()].tolist())))
    return out

# ---- gallery ----
def load_gallery(path=GALLERY):
    if not os.path.exists(path): return dict(vec=np.zeros((0, 512), np.float32), lab=np.zeros((0, 3), np.float32), shape=np.array([], str), color=np.array([], str), src=np.array([], str))
    g = np.load(path); return {k: g[k] for k in g.files}

def save_gallery(g, path=GALLERY):
    np.savez_compressed(path, **g)

def add_entries(g, vecs, labs, shape, color, srcs):
    n = len(vecs)
    return dict(vec=np.concatenate([g['vec'], vecs]).astype(np.float32), lab=np.concatenate([g['lab'], labs]).astype(np.float32),
                shape=np.concatenate([g['shape'], [shape] * n]), color=np.concatenate([g['color'], [str(color)] * n]),
                src=np.concatenate([g['src'], srcs]))

def add_photos(paths, shape='', color='', model=None, path=GALLERY):
    """THE way to teach a new piece or color: ~20 single-brick photos -> gallery entries. Returns how many were added.
    Give only `shape` (color unknown/irrelevant, e.g. renders) or only `color` (any shape, e.g. a new color):
    an entry with an empty label is skipped by that vote in recognize()."""
    assert shape or color, 'give a shape, a color, or both'
    shape, color = str(shape), str(color)
    model = model or load_embedder()
    found = crops_from_photos(paths)
    assert found, 'no brick found in any photo'
    g = load_gallery(path)
    keep = ~((g['shape'] == shape) & (g['color'] == str(color)) & np.isin(g['src'], [p for p, _ in found]))   # re-adding a photo replaces it
    g = {k: v[keep] for k, v in g.items()}
    g = add_entries(g, embed(model, [c for _, c in found]), np.array([brick_lab(c) for _, c in found]), shape, color, [p for p, _ in found])
    save_gallery(g, path)
    return len(found)

# ---- recognition ----
SHAPE_MIN_SIM = 0.977  # below this best-match similarity -> "unknown"; from notebooks/lookup.ipynb step 4 (keeps 95% of known pieces)
CLOSED_MIN_SIM = 0.93  # closed_set mode: both-right 0.989 on the test split, but only 31% of unseen shapes are caught (notebooks/lookup.ipynb 4b)
COLOR_MAX_DIST = 1e9  # shortcut: never says "unknown" color; calibrate once photos of a color NOT in the gallery exist

def recognize(g, vecs, labs, k=5, shape_min_sim=None, color_max_dist=None, closed_set=False):
    """vecs (N,512), labs (N,3) -> list of (shape, shape_sim, color, color_dist). k-nearest vote, 'unknown' if too far.
    closed_set=True (fixed set of known pieces): looser shape cutoff, and color voted only among colors seen with the predicted shape.
    It rejects a new color on a known shape, so keep it off while the set is still growing."""
    smin = (CLOSED_MIN_SIM if closed_set else SHAPE_MIN_SIM) if shape_min_sim is None else shape_min_sim
    cmax = COLOR_MAX_DIST if color_max_dist is None else color_max_dist
    sims = np.where(g['shape'] != '', vecs @ g['vec'].T, -np.inf)   # color-only entries (shape '') can't vote on shape
    dl = np.where(g['color'] != '', np.linalg.norm(labs[:, None] - g['lab'][None], axis=2), np.inf)   # shape-only entries can't vote on color
    out = []
    for i in range(len(vecs)):
        nn = np.argsort(-sims[i])[:k]; nn = nn[np.isfinite(sims[i][nn])]
        shape, s = ('unknown', -1.0) if not len(nn) else (max(set(g['shape'][nn]), key=lambda t: sims[i][nn][g['shape'][nn] == t].sum()), float(sims[i][nn[0]]))
        di = dl[i]
        if closed_set and len(nn): di = np.where(np.isin(g['color'], g['color'][g['shape'] == shape]), di, np.inf)   # only colors seen with this shape
        nc = np.argsort(di)[:k]; nc = nc[np.isfinite(di[nc])]
        color, d = ('unknown', float('inf')) if not len(nc) else (max(set(g['color'][nc]), key=list(g['color'][nc]).count), float(dl[i][nc[0]]))
        out.append((shape if s >= smin else 'unknown', s, color if d <= cmax else 'unknown', d))
    return out
