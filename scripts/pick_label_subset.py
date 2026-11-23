# Hardlinks 20 evenly spaced photos per element into labeling/images/ for manual boxing.
import csv, os
from collections import defaultdict
N = 20
by = defaultdict(list)
for r in csv.DictReader(open('data/labels.csv')):
    by[r['element_id']].append(r['path'])
os.makedirs('labeling/images', exist_ok=True)
total = 0
for e, paths in by.items():
    paths.sort()
    for p in [paths[i * len(paths) // N] for i in range(N)]:
        dst = f'labeling/images/{e}_{os.path.basename(p)}'  # element in name: class recoverable from file alone
        if not os.path.exists(dst):
            os.link(p, dst)
        total += 1
print(total, 'images in labeling/images')
