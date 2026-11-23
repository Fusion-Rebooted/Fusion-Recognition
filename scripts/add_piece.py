# Teach the lookup a new piece (or a known shape in a new color): ~20 single-brick photos -> data/gallery.npz.
# Usage (from repo root): python scripts/add_piece.py photos/my_piece/ --shape 3001 --color 21
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lego_lookup as L

ap = argparse.ArgumentParser()
ap.add_argument('folder'); ap.add_argument('--shape', required=True, help='part number, e.g. 3001')
ap.add_argument('--color', required=True, help='official LEGO color ID, e.g. 21')
a = ap.parse_args()
paths = sorted(p for p in glob.glob(os.path.join(a.folder, '*')) if p.lower().endswith(('.jpg', '.jpeg', '.png')))
if not paths: sys.exit(f'no jpg/png photos in {a.folder}')
n = L.add_photos(paths, a.shape, a.color)
print(f'added {n} of {len(paths)} photos as shape {a.shape}, color {a.color} -> {L.GALLERY}')
