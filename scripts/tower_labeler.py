# Tower box labeler: drag a box on the photo, then click the piece's picture on the right to give it its class.
# Saves X-AnyLabeling/labelme JSON (labeling/tower_labels/<photo>.json), so the files still open in X-AnyLabeling.
# Needs labeling/thumbs/NN.png (written by the last cell of notebooks/test_real.ipynb) and labeling/tower_classes.txt.
# Mouse: drag = new box | click inside a box = select it | picture = set class of the selected box.
# Keys: Right/Left = next/previous photo | Delete = remove the selected box | boxes without a class are not saved.
import glob, json, os, tkinter as tk
from PIL import Image, ImageOps, ImageTk

IMG_DIR, OUT, COLS, TH, CW, CH = 'labeling/towers', 'labeling/tower_labels', 6, 80, 1100, 880
classes = open('labeling/tower_classes.txt').read().split()
photos = sorted(f for f in glob.glob(IMG_DIR + '/*') if f.lower().endswith(('.jpg', '.jpeg', '.png')))
assert photos, f'put tower photos in {IMG_DIR}/'
os.makedirs(OUT, exist_ok=True)

class App:
    def __init__(s, root):
        s.root, s.i, s.boxes, s.sel, s.p0 = root, 0, [], None, None   # boxes: [x0, y0, x1, y1, class] in photo pixels
        s.cv = tk.Canvas(root, width=CW, height=CH, bg='#222', cursor='cross'); s.cv.pack(side='left')
        side = tk.Frame(root); side.pack(side='left', fill='y'); s.thumbs = []
        for n, c in enumerate(classes):
            f = f'labeling/thumbs/{c[:2]}.png'
            s.thumbs.append(ImageTk.PhotoImage(Image.open(f).convert('RGB').resize((TH, TH)) if os.path.exists(f) else Image.new('RGB', (TH, TH), 'gray')))
            tk.Button(side, image=s.thumbs[-1], text=c[:2], compound='top', command=lambda c=c: s.assign(c)).grid(row=n // COLS, column=n % COLS)
        s.cv.bind('<ButtonPress-1>', s.press); s.cv.bind('<B1-Motion>', s.drag); s.cv.bind('<ButtonRelease-1>', s.release)
        root.bind('<Right>', lambda e: s.go(1)); root.bind('<Left>', lambda e: s.go(-1)); root.bind('<Delete>', s.delete)
        s.load(0)

    def path(s): return f'{OUT}/{os.path.splitext(os.path.basename(photos[s.i]))[0]}.json'

    def load(s, i):
        s.i, s.sel = i, None
        s.img = ImageOps.exif_transpose(Image.open(photos[i])).convert('RGB')   # same orientation X-AnyLabeling shows
        s.k = min(CW / s.img.width, CH / s.img.height)
        s.tk = ImageTk.PhotoImage(s.img.resize((int(s.img.width * s.k), int(s.img.height * s.k))))
        s.boxes = []
        if os.path.exists(s.path()):
            for sh in json.load(open(s.path()))['shapes']:
                xs, ys = [p[0] for p in sh['points']], [p[1] for p in sh['points']]
                s.boxes.append([min(xs), min(ys), max(xs), max(ys), sh['label']])
        s.redraw()

    def save(s):
        shapes = [dict(label=c, points=[[x0, y0], [x1, y1]], group_id=None, description='', shape_type='rectangle', flags={}) for x0, y0, x1, y1, c in s.boxes if c]
        json.dump(dict(version='2.0', flags={}, shapes=shapes, imagePath=os.path.relpath(photos[s.i], OUT), imageData=None,
                       imageHeight=s.img.height, imageWidth=s.img.width), open(s.path(), 'w'), indent=1)

    def redraw(s):
        s.cv.delete('all'); s.cv.create_image(0, 0, image=s.tk, anchor='nw')
        for j, (x0, y0, x1, y1, c) in enumerate(s.boxes):
            col = 'yellow' if j == s.sel else 'lime' if c else 'red'
            s.cv.create_rectangle(x0 * s.k, y0 * s.k, x1 * s.k, y1 * s.k, outline=col, width=2)
            if c: s.cv.create_text(x0 * s.k + 3, y0 * s.k + 2, text=c[:2], fill=col, anchor='nw', font=('Arial', 12, 'bold'))
        s.root.title(f'{s.i + 1}/{len(photos)} {os.path.basename(photos[s.i])} | {sum(1 for b in s.boxes if b[4])} labeled, {sum(1 for b in s.boxes if not b[4])} without class')

    def press(s, e): s.p0 = (e.x, e.y)
    def drag(s, e):
        s.cv.delete('rubber'); s.cv.create_rectangle(*s.p0, e.x, e.y, outline='cyan', tags='rubber')

    def release(s, e):
        s.cv.delete('rubber'); (x0, y0), (x1, y1) = s.p0, (e.x, e.y)
        if abs(x1 - x0) < 6 or abs(y1 - y0) < 6:   # a click: select the smallest box under it
            hit = [j for j, b in enumerate(s.boxes) if b[0] * s.k <= e.x <= b[2] * s.k and b[1] * s.k <= e.y <= b[3] * s.k]
            s.sel = min(hit, key=lambda j: (s.boxes[j][2] - s.boxes[j][0]) * (s.boxes[j][3] - s.boxes[j][1])) if hit else None
        else:
            s.boxes.append([max(min(x0, x1), 0) / s.k, max(min(y0, y1), 0) / s.k, min(max(x0, x1), s.img.width * s.k) / s.k, min(max(y0, y1), s.img.height * s.k) / s.k, ''])
            s.sel = len(s.boxes) - 1
        s.redraw()

    def assign(s, c):
        if s.sel is not None: s.boxes[s.sel][4] = c; s.save(); s.redraw()

    def delete(s, e):
        if s.sel is not None: del s.boxes[s.sel]; s.sel = None; s.save(); s.redraw()

    def go(s, d):
        s.save(); s.load((s.i + d) % len(photos))

root = tk.Tk(); App(root); root.mainloop()
