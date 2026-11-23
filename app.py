# Add new pieces to the lookup gallery from your own photos. Brickognize names the part, you confirm, the app stores it.
# Run from the repo root: .venv/bin/python app.py  -> opens http://127.0.0.1:7860
import csv, io, os, shutil
from collections import Counter
import cv2, gradio as gr, numpy as np, requests
import lego_lookup as L

COLORS = 'data/lego_colors.csv'   # LEGO color ID <-> name <-> BrickLink name (Brickognize speaks BrickLink)
API = 'https://api.brickognize.com/predict/parts/'

def colors():
    return list(csv.DictReader(open(COLORS)))

model = L.load_embedder()
from ultralytics import YOLO
det = YOLO(L.BOX_PT)

def brickognize(crops):
    """Up to 3 crops of the same brick -> (parts [(id, name, score, img_url)], colors [(bricklink name, score)])."""
    files = [('query_image', (f'{i}.jpg', cv2.imencode('.jpg', c[..., ::-1])[1].tobytes(), 'image/jpeg')) for i, c in enumerate(crops[:3])]
    r = requests.post(API, params={'predict_color': 'true', 'top_k_items': 3}, files=files, timeout=30)
    r.raise_for_status(); j = r.json()
    return ([(i['id'], i['name'], i['score'], i['img_url']) for i in j.get('items', [])],
            [(c['name'], c['score']) for c in j.get('colors', [])])

def our_shape_id(bl_id, known):
    """BrickLink '3040' is Rebrickable/our '3040b': reuse our id when only a letter suffix differs."""
    m = [s for s in known if s == bl_id or (s.startswith(bl_id) and s[len(bl_id):].isalpha())]
    return m[0] if m else bl_id

def identify(files):
    if not files: return [], 'Upload photos first.', '', '', '', None
    paths = [f if isinstance(f, str) else f.name for f in files]
    found = L.crops_from_photos(paths, det)
    if not found: return [], 'No brick found in any photo. One brick per photo, on a plain table.', '', '', '', None
    crops = [c for _, c in found]
    g = L.load_gallery()
    res = L.recognize(g, L.embed(model, crops), np.array([L.brick_lab(c) for c in crops]))
    shape = Counter(r[0] for r in res).most_common(1)[0][0]
    ours_color = Counter(r[2] for r in res).most_common(1)[0][0]
    msg = [f'**Gallery says:** shape `{shape}`, color `{ours_color}` (best similarity {max(r[1] for r in res):.3f})']
    shape_guess, color_guess = (shape if shape != 'unknown' else ''), ours_color
    try:
        parts, cols = brickognize(crops)
        msg.append('**Brickognize says:**')
        msg += [f'- `{i}` {n} ({s:.0%}) [image]({u})' for i, n, s, u in parts] or ['- no part match']
        if parts: shape_guess = our_shape_id(parts[0][0], set(g['shape']))
        if cols:
            msg.append('- color: ' + ', '.join(f'{n} ({s:.0%})' for n, s in cols[:3]))
            hit = [c for c in colors() if c['bricklink_name'] == cols[0][0]]
            color_guess = hit[0]['lego_id'] if hit else ''
            if not hit: msg.append(f'- **New color** "{cols[0][0]}": type its LEGO color ID and name below.')
    except Exception as e:   # API down or offline: the gallery answer + manual entry still work
        msg.append(f'Brickognize unavailable ({e}). Type the part number yourself.')
    name = next((c['lego_name'] for c in colors() if c['lego_id'] == color_guess), '')
    msg.append(f'\nCheck the fields, then **Add to gallery**. ({len(found)} of {len(paths)} photos had a brick.)')
    return [c for c in crops], '\n'.join(msg), shape_guess, color_guess, name, paths

def add(paths, shape, color_id, color_name):
    shape, color_id, color_name = shape.strip(), color_id.strip(), color_name.strip()
    if not paths: return 'Identify first.'
    if not shape or not color_id.isdigit(): return 'Need a part number and a numeric LEGO color ID.'
    known = {c['lego_id'] for c in colors()}
    if color_id not in known:
        if not color_name: return f'Color {color_id} is new: give it a name too.'
        with open(COLORS, 'a', newline='') as f: csv.writer(f).writerow([color_id, color_name, color_name])
    keep = f'photos/{shape}_{color_id}'   # keep the originals: future training data
    os.makedirs(keep, exist_ok=True)
    kept = []
    for p in paths:
        dst = os.path.join(keep, os.path.basename(p)); shutil.copy(p, dst); kept.append(dst)
    n = L.add_photos(kept, shape, color_id, model)
    return f'Added {n} photos as `{shape}` / color `{color_id}`. Saved in `{keep}/`. Commit `data/gallery.npz` to share.'

with gr.Blocks(title='LEGO piece adder') as app:
    gr.Markdown('## Add a LEGO piece\nUpload ~20 photos of **one** brick, alone on a plain table, from different sides.')
    files = gr.File(file_count='multiple', file_types=['image'], label='Photos')
    go = gr.Button('Identify', variant='primary')
    crops = gr.Gallery(label='What the detector cropped', columns=8, height=200)
    info = gr.Markdown()
    with gr.Row():
        shape = gr.Textbox(label='Part number')
        cid = gr.Textbox(label='LEGO color ID')
        cname = gr.Textbox(label='Color name (only for a new color)')
    paths = gr.State()
    addb = gr.Button('Add to gallery')
    status = gr.Markdown()
    go.click(identify, files, [crops, info, shape, cid, cname, paths])
    addb.click(add, [paths, shape, cid, cname], status)

if __name__ == '__main__':
    app.launch()
