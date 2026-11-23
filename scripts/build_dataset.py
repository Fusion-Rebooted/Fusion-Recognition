# Hardlinks every photo into dataset/<part>_<LEGO color ID>_<element>/ and writes data/labels.csv.
# Originals are never moved or modified.
import csv, os, glob

# Rebrickable color name -> (official LEGO color ID, LEGO color name)
LEGO_COLOR = {
    'White': (1, 'White'), 'Black': (26, 'Black'), 'Tan': (5, 'Brick Yellow'),
    'Dark Green': (141, 'Earth Green'), 'Sand Green': (151, 'Sand Green'),
    'Reddish Brown': (192, 'Reddish Brown'), 'Light Bluish Gray': (194, 'Medium Stone Grey'),
    'Pearl Gold': (297, 'Warm Gold'),
}

em = {r['element_id']: r for r in csv.DictReader(open('data/element_map.csv'))}
ranges = [(int(r['start']), int(r['end']), r['element_id']) for r in csv.DictReader(open('data/ranges.csv'))]

# phone dump first, then curated dataset overrides it for shared filenames
src = {}
for f in glob.glob('photos_from_phone/*/IMG_*.JPG') + glob.glob('The Lego Dataset/**/IMG_*.JPG', recursive=True):
    src[os.path.basename(f)] = f

rows, unlabeled = [], []
for name, f in sorted(src.items()):
    n = int(name[4:8])
    hits = [e for a, b, e in ranges if a <= n <= b]
    assert len(hits) <= 1, f'{name} in overlapping ranges {hits}'
    if not hits:
        unlabeled.append(name); continue
    e = em[hits[0]]
    cid, cname = LEGO_COLOR[e['color']]
    d = f"dataset/{e['part_num']}_{cid}_{hits[0]}"
    os.makedirs(d, exist_ok=True)
    if not os.path.exists(f'{d}/{name}'):
        os.link(f, f'{d}/{name}')
    rows.append([f'{d}/{name}', hits[0], e['part_num'], e['part_name'], cid, cname])

with open('data/labels.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'element_id', 'part_num', 'part_name', 'lego_color_id', 'lego_color_name'])
    w.writerows(rows)

print(len(rows), 'labeled,', len(unlabeled), 'unlabeled:', unlabeled[:20])
