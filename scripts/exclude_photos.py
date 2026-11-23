# Exclude photos from training WITHOUT deleting them: pick a folder, click a photo to toggle it (red = excluded).
# Saved in data/excluded.csv (column: path). notebooks/lookup.ipynb (first cell) drops these paths from train/val/test/gallery.
# Photos are shown blurriest first when data/blur.csv exists (made by notebooks/blur.ipynb). Run: .venv/bin/python scripts/exclude_photos.py
import os, tkinter as tk
import pandas as pd
from PIL import Image, ImageDraw, ImageTk

EX, COLS, ROWS, TH = 'data/excluded.csv', 6, 4, 170
lab = pd.read_csv('data/labels.csv', dtype=str); lab['folder'] = lab.path.str.rsplit('/', n=1).str[0]
rel = {}
if os.path.exists('data/blur.csv'):
    b = pd.read_csv('data/blur.csv', dtype={'element_id': str}).dropna()
    rel = dict(zip(b.path, b.blur / b.groupby('element_id').blur.transform('median')))
excluded = set(pd.read_csv(EX).path) if os.path.exists(EX) else set()

def save(): pd.DataFrame({'path': sorted(excluded)}).to_csv(EX, index=False)

class App:
    def __init__(s, root):
        s.root, s.page, s.ps, s.imgs = root, 0, [], [None] * (COLS * ROWS)
        s.folders = sorted(lab.folder.unique()); s.var = tk.StringVar(value=s.folders[0])
        bar = tk.Frame(root); bar.pack(fill='x')
        tk.OptionMenu(bar, s.var, *s.folders, command=lambda _: s.show(0)).pack(side='left')
        tk.Button(bar, text='< prev', command=lambda: s.show(s.page - 1)).pack(side='left')
        tk.Button(bar, text='next >', command=lambda: s.show(s.page + 1)).pack(side='left')
        tk.Button(bar, text='exclude everything in data/blurry.csv', command=s.add_blurry).pack(side='left', padx=10)
        s.info = tk.Label(bar); s.info.pack(side='left')
        grid = tk.Frame(root); grid.pack(); s.cells = []
        for i in range(COLS * ROWS):
            l = tk.Label(grid); l.grid(row=i // COLS, column=i % COLS, padx=2, pady=2)
            l.bind('<Button-1>', lambda e, i=i: s.toggle(i)); s.cells.append(l)
        root.bind('<Right>', lambda e: s.show(s.page + 1)); root.bind('<Left>', lambda e: s.show(s.page - 1))
        s.show(0)

    def show(s, page):
        ps = sorted(lab.path[lab.folder == s.var.get()], key=lambda p: (rel.get(p, 9), p))   # blurriest first
        s.npages = max(1, -(-len(ps) // (COLS * ROWS))); s.page = min(max(page, 0), s.npages - 1); s.all = ps
        s.ps = ps[s.page * COLS * ROWS:(s.page + 1) * COLS * ROWS]
        for i in range(COLS * ROWS): s.draw(i)

    def draw(s, i):
        if i >= len(s.ps): s.cells[i].config(image=''); s.imgs[i] = None; s.update_info(); return
        p = s.ps[i]; f = 'crops224/' + p
        im = Image.open(f).convert('RGB').resize((TH, TH)) if os.path.exists(f) else Image.new('RGB', (TH, TH), 'gray')
        d = ImageDraw.Draw(im); d.text((6, 4), f'{os.path.basename(p)[4:8]}  sharp {rel.get(p, float("nan")):.2f}', fill='yellow')
        if p in excluded: d.rectangle([0, 0, TH - 1, TH - 1], outline='red', width=8); d.line([0, 0, TH, TH], fill='red', width=4); d.line([0, TH, TH, 0], fill='red', width=4)
        s.imgs[i] = ImageTk.PhotoImage(im); s.cells[i].config(image=s.imgs[i]); s.update_info()

    def update_info(s):
        s.info.config(text=f'page {s.page + 1}/{s.npages} | {sum(p in excluded for p in s.all)} of {len(s.all)} excluded in this folder | {len(excluded)} in total')

    def toggle(s, i):
        if i < len(s.ps): excluded.symmetric_difference_update({s.ps[i]}); save(); s.draw(i)

    def add_blurry(s):
        if os.path.exists('data/blurry.csv'): excluded.update(pd.read_csv('data/blurry.csv').path); save(); s.show(s.page)

root = tk.Tk(); root.title('exclude photos from training'); App(root); root.mainloop()
