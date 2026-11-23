# Headless LDraw renderer: part number + LDraw color + random view -> RGBA image (transparent background).
# Reads the plain-text LDraw files in ldraw/ (download: https://library.ldraw.org/library/updates/complete.zip) and draws on the GPU.
import os, re, functools, numpy as np, cv2, moderngl

LD = 'ldraw'

@functools.lru_cache(None)
def _index():
    """lowercase relative name -> real path, for parts/ and p/ (LDraw names are case-insensitive, use backslashes)."""
    idx = {}
    for top in ('p', 'parts'):   # parts/ wins over p/ on a name clash
        root = os.path.join(LD, top)
        for d, _, fs in os.walk(root):
            for f in fs: idx[os.path.relpath(os.path.join(d, f), root).lower()] = os.path.join(d, f)
    return idx

@functools.lru_cache(None)
def colors():
    """LDraw color code -> (r, g, b) 0..1, from LDConfig.ldr."""
    out = {}
    for l in open(os.path.join(LD, 'LDConfig.ldr'), encoding='utf-8', errors='replace'):
        m = re.search(r'!COLOUR\s+\S+\s+CODE\s+(\d+)\s+VALUE\s+#(\w{6})', l)
        if m: out[int(m.group(1))] = tuple(int(m.group(2)[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return out

@functools.lru_cache(None)
def flatten(name):
    """All triangles of a part in its own coordinates: (N,3,3) float32 and (N,) color codes (16 = 'the part's color')."""
    path = _index().get(name.lower().replace('\\', '/'))
    if path is None: raise FileNotFoundError(f'LDraw file not found: {name}')
    tris, cols = [], []
    for l in open(path, encoding='utf-8', errors='replace'):
        t = l.split()
        if not t: continue
        if t[0] == '1' and len(t) >= 15:
            c = int(t[1]); v = np.array(t[2:14], np.float32)
            M, off = v[3:12].reshape(3, 3), v[:3]
            st, sc = flatten(' '.join(t[14:]))
            if len(st):
                tris.append(st @ M.T + off)
                cols.append(np.where(sc == 16, c, sc))
        elif t[0] in ('3', '4') and len(t) >= (11 if t[0] == '3' else 14):
            c = int(t[1]); p = np.array(t[2:], np.float32).reshape(-1, 3)
            for q in ([p[[0, 1, 2]]] if t[0] == '3' else [p[[0, 1, 2]], p[[0, 2, 3]]]):
                tris.append(q[None]); cols.append(np.array([c]))
    if not tris: return np.zeros((0, 3, 3), np.float32), np.zeros(0, int)
    return np.concatenate(tris).astype(np.float32), np.concatenate(cols)

VS = '''#version 330
in vec3 pos; in vec3 nrm; in vec3 col;
uniform mat4 mvp; uniform mat3 rot;
out vec3 vn; out vec3 vc;
void main() { gl_Position = mvp * vec4(pos, 1.0); vn = rot * nrm; vc = col; }'''
FS = '''#version 330
in vec3 vn; in vec3 vc; out vec4 o;
uniform vec3 light; uniform float amb; uniform float spec;
void main() {
    vec3 n = normalize(vn); if (n.z < 0.0) n = -n;          // LDraw winding is not always reliable: face the camera
    float d = max(dot(n, light), 0.0);
    float s = pow(max(dot(n, normalize(light + vec3(0, 0, 1))), 0.0), 40.0) * spec;   // plastic shine
    o = vec4(vc * (amb + (1.0 - amb) * d) + s, 1.0);
}'''

@functools.lru_cache(None)
def _gl():
    ctx = moderngl.create_standalone_context(backend='egl')
    ctx.enable(moderngl.DEPTH_TEST)
    return ctx, ctx.program(vertex_shader=VS, fragment_shader=FS)

@functools.lru_cache(None)
def _mesh(part):
    tris, cols = flatten(part + '.dat')
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    keep = np.linalg.norm(n, axis=1) > 1e-9
    tris, cols, n = tris[keep], cols[keep], n[keep] / np.linalg.norm(n[keep], axis=1, keepdims=True)
    c = (tris.reshape(-1, 3).max(0) + tris.reshape(-1, 3).min(0)) / 2
    tris = tris - c
    return tris, cols, n, float(np.linalg.norm(tris.reshape(-1, 3), axis=1).max())

def random_rotation(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], np.float32)

def render(part, color, size=224, rng=None, rot=None):
    """part: LDraw part number ('3004'); color: LDraw color code (see data/lego_colors.csv ldraw_code).
    Returns (size,size,4) RGBA uint8 with transparent background; the part fills ~85% of the image."""
    rng = rng or np.random.default_rng()
    ctx, prog = _gl()
    tris, cols, nrm, r = _mesh(part)
    pal = colors(); base = pal.get(int(color), (0.5, 0.5, 0.5))
    rgb = np.array([base if c in (16, 24) else pal.get(int(c), base) for c in cols], np.float32)
    rgb = np.clip(rgb * rng.uniform(0.85, 1.15), 0, 1)   # a little lighting/white-balance variation
    R = random_rotation(rng) if rot is None else np.asarray(rot, np.float32)
    S = 2 * size   # render 2x, shrink = anti-aliasing
    s = 0.85 / r   # orthographic: bounding sphere fills 85% of the frame
    mvp = np.eye(4, dtype=np.float32); mvp[:3, :3] = R * s; mvp[1] *= -1   # LDraw: -Y is up
    mvp[2] *= -0.5 / 0.85   # keep depth inside the clip range
    data = np.concatenate([tris.reshape(-1, 3), np.repeat(nrm, 3, 0) * [1, -1, 1], np.repeat(rgb, 3, 0)], 1).astype('f4')
    vbo = ctx.buffer(data.tobytes())
    vao = ctx.vertex_array(prog, [(vbo, '3f 3f 3f', 'pos', 'nrm', 'col')])
    fbo = ctx.simple_framebuffer((S, S), components=4); fbo.use(); fbo.clear(0, 0, 0, 0)
    prog['mvp'].write(mvp.T.copy().tobytes())
    rn = R.copy(); rn[2] *= -1   # normals into camera space (z towards the camera)
    prog['rot'].write(rn.T.copy().tobytes())
    l = rng.normal(size=3); l[2] = abs(l[2]) + 0.5; prog['light'].value = tuple(l / np.linalg.norm(l))
    prog['amb'].value = float(rng.uniform(0.3, 0.5)); prog['spec'].value = float(rng.uniform(0.05, 0.3))
    vao.render(moderngl.TRIANGLES)
    img = np.frombuffer(fbo.read(components=4), np.uint8).reshape(S, S, 4)[::-1]
    vao.release(); vbo.release(); fbo.release()
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

# ---- towers: valid random stacks on a 16 x 16 stud grid, rendered with an exact per-brick id mask, exported as .ldr ----
GRID = 16   # the real towers are 16 x 16 studs

@functools.lru_cache(None)
def _dims(part):
    """(bbox centre in part coordinates, bbox size in LDU). Same filtered triangles as _mesh, so centring matches the render."""
    t, _ = flatten(part + '.dat')
    t = t[np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1) > 1e-9].reshape(-1, 3)
    lo, hi = t.min(0), t.max(0)
    return (lo + hi) / 2, hi - lo

@functools.lru_cache(None)
def _rgb(part, color):
    pal = colors(); base = pal.get(int(color), (.5, .5, .5))
    return np.array([base if c in (16, 24) else pal.get(int(c), base) for c in _mesh(part)[1]], np.float32)

HEIGHT = 16 * 24   # LDU: 16 bricks tall

def _profile(width, rng):
    """Target height (LDU) per stud column: flat wall, tall middle, steps, or two low wings."""
    base = int(rng.integers(6, 17)) * 24; x = np.arange(width); kind = rng.choice(['flat', 'middle', 'steps', 'wings'])
    if kind == 'flat': h = np.full(width, base)
    elif kind == 'middle': h = base * (0.5 + 0.5 * (1 - np.abs(x - (width - 1) / 2) / (width / 2)))
    elif kind == 'steps': h = np.repeat(rng.uniform(.4, 1, -(-width // 3)), 3)[:width] * base
    else: h = np.where((x < width // 4) | (x >= width - width // 4), 0.45 * base, base)
    return np.maximum(h, 48).astype(int)

def make_tower(parts, rng, width=None, height=None, plate=False):
    """parts: list of (part, ldraw_color). The real towers are flat walls: ONE stud thick, `width` studs wide (random 6-16 if None), stacked bottom-up,
    with a random height profile (`height`: LDU per column or a number). Each tower prefers 1-3 colors, like the real ones.
    Returns bricks = dicts(part, color, i, j, rot, w, d, base, h). Always fills the lowest gap with a random piece that fits it (so no overlaps, no floating).
    Pieces thicker than 1 stud (e.g. the 6x12 plate) are skipped. shortcut: footprint = bounding box (slopes/arches count as full blocks) and the
    top edge is ragged (tall pieces overshoot); fine for occlusion data, upgrade for exact layouts."""
    width = int(rng.integers(6, (12 if plate else GRID) + 1)) if width is None else width   # plate=True: the 12-stud base plate limits the width
    hv = _profile(width, rng) if height is None else np.broadcast_to(np.asarray(height), (width,))
    allc = sorted({int(c) for _, c in parts}); pal = set(rng.choice(allc, size=int(rng.integers(1, 4)), replace=False).tolist())
    hm, out, choices = np.zeros(width, int), [], []   # hm: height of the top surface per stud column, LDU
    for part, col in parts:
        ext = _dims(part)[1]; sx, sz = int(np.ceil(ext[0] / 20 - .1)), int(np.ceil(ext[2] / 20 - .1))
        for rot in range(4):
            w, d = (sx, sz) if rot % 2 == 0 else (sz, sx)   # w along the wall, d = thickness
            if d == 1: choices.append((part, int(col), rot, w, max(8, int(ext[1] // 8 * 8))))   # body height: the 4 LDU stud on top is not counted
    while (hm < hv).any():
        x = int(np.where(hm < hv, hm, 10 ** 9).argmin()); run = 1   # lowest column that still needs bricks
        while x + run < width and hm[x + run] == hm[x]: run += 1
        fits = [c for c in choices if c[3] <= run]; assert fits, f'no 1-stud-thick piece fits a gap of {run} studs'
        mine = [c for c in fits if c[1] in pal]; pool = mine if mine and rng.random() < 0.8 else fits   # mostly the tower's own colors
        part, col, rot, w, h = pool[rng.integers(len(pool))]; base = int(hm[x]); hm[x:x + w] = base + h
        out.append(dict(part=part, color=col, i=x, j=0, rot=rot, w=w, d=1, base=base, h=h))
    for a, A in enumerate(out):   # self-check: no two bricks share space
        for B in out[:a]:
            if A['i'] < B['i'] + B['w'] and B['i'] < A['i'] + A['w'] and A['j'] < B['j'] + B['d'] and B['j'] < A['j'] + A['d']:
                assert A['base'] >= B['base'] + B['h'] or B['base'] >= A['base'] + A['h'], 'bricks overlap'
    if plate:   # the real towers stand one row in from the back edge of the 6x12 base plate (LDraw has no printed 6178pr0002: the plain 3028 plate stands in, labeled as it)
        i0 = int(rng.integers(0, 12 - width + 1))
        for b in out: b['i'] += i0; b['j'] = 1
        out.insert(0, dict(part='3028', color=71, i=0, j=0, rot=0, w=12, d=6, base=-8, h=8))
    return out

def _place(b):
    """Yaw matrix (90 degree steps) and the part-centre position of a brick, in LDraw coordinates (-Y is up)."""
    th = b['rot'] * np.pi / 2; c, s = round(np.cos(th)), round(np.sin(th))
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return Ry, np.array([(b['i'] + b['w'] / 2) * 20, -b['base'] - _dims(b['part'])[1][1] / 2, (b['j'] + b['d'] / 2) * 20], np.float32)

@functools.lru_cache(None)
def _fbo(S):
    return _gl()[0].simple_framebuffer((S, S), components=4)

def render_scene(bricks, size=512, az=None, el=None, rng=None, zoom=None, fov=None):
    """-> (rgba (size,size,4) uint8, ids (size,size) int32). ids: 0 = background, k + 1 = bricks[k] (exact, hidden parts excluded).
    Orthographic camera; az = turn around the tower (0 = straight at the front), el = looking down (radians; random near the front if None)."""
    rng = rng or np.random.default_rng(); ctx, prog = _gl()
    T, C, I = [], [], []
    for k, b in enumerate(bricks):
        Ry, ctr = _place(b); tris = _mesh(b['part'])[0]
        T.append(tris @ Ry.T + ctr); C.append(_rgb(b['part'], b['color'])); I.append(np.full(len(tris), k + 1))
    T, C, I = np.concatenate(T), np.concatenate(C), np.concatenate(I)
    T -= (T.reshape(-1, 3).max(0) + T.reshape(-1, 3).min(0)) / 2
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    r = float(np.linalg.norm(T.reshape(-1, 3), axis=1).max())
    az = rng.uniform(-0.4, 0.4) if az is None else az; el = rng.uniform(0.0, 0.9) if el is None else el   # like the photos: nearly from the front, level to ~50 degrees above
    fov = rng.uniform(45, 70) if fov is None else fov   # phone camera, degrees; 0 = orthographic
    ca, sa, ce, se = np.cos(az), np.sin(az), np.cos(el), np.sin(el)
    R = np.array([[1, 0, 0], [0, ce, se], [0, -se, ce]], np.float32) @ np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], np.float32)   # tilt down, then turn
    zoom = rng.uniform(0.35, 0.75) if zoom is None else zoom   # fraction of the frame the object fills; smaller + off-centre = the tower sits anywhere
    sx, sy = rng.uniform(-1, 1, 2) * (1 - 0.9 * zoom) * 0.6
    if fov:   # perspective: camera on +z looking at the centre; clip = (f*x', -f*y', depth, w) with w = distance, so far things shrink
        f = 1 / np.tan(np.radians(fov) / 2); d = f * r / (0.9 * zoom); near, far = d - 1.2 * r, d + 1.2 * r
        A, B = (far + near) / (far - near), -2 * far * near / (far - near)
        mvp = np.zeros((4, 4), np.float32); mvp[0, :3], mvp[1, :3] = f * R[0], -f * R[1]
        mvp[2, :3], mvp[2, 3] = -A * R[2], A * d + B; mvp[3, :3], mvp[3, 3] = -R[2], d
        mvp[0] += sx * mvp[3]; mvp[1] += sy * mvp[3]   # shift in the picture
    else:
        s = 0.9 * zoom / r; mvp = np.eye(4, dtype=np.float32); mvp[:3, :3] = R * s; mvp[1] *= -1; mvp[2] *= -0.5 / 0.9; mvp[:2, 3] = sx, sy
    rn = R.copy(); rn[2] *= -1
    S = 2 * size
    def draw(rgb, amb, spec, light):
        data = np.concatenate([T.reshape(-1, 3), np.repeat(n, 3, 0) * [1, -1, 1], np.repeat(rgb, 3, 0)], 1).astype('f4')
        vbo = ctx.buffer(data.tobytes()); vao = ctx.vertex_array(prog, [(vbo, '3f 3f 3f', 'pos', 'nrm', 'col')])
        fbo = _fbo(S); fbo.use(); fbo.clear(0, 0, 0, 0)   # one cached framebuffer per size: creating one per call leaked GPU memory
        prog['mvp'].write(mvp.T.copy().tobytes()); prog['rot'].write(rn.T.copy().tobytes())
        prog['light'].value = light; prog['amb'].value = amb; prog['spec'].value = spec
        vao.render(moderngl.TRIANGLES)
        img = np.frombuffer(fbo.read(components=4), np.uint8).reshape(S, S, 4)[::-1].copy()
        vao.release(); vbo.release(); return img
    l = rng.normal(size=3); l[2] = abs(l[2]) + 0.5
    rgba = draw(np.clip(C * rng.uniform(0.85, 1.15), 0, 1), float(rng.uniform(0.3, 0.5)), float(rng.uniform(0.05, 0.3)), tuple(l / np.linalg.norm(l)))
    idc = (np.stack([I % 256, (I // 256) % 256, I // 65536], 1) / 255).astype(np.float32)   # flat colour = brick id (amb 1, no shine: exact)
    ids = draw(idc, 1.0, 0.0, (0.0, 0.0, 1.0))[::2, ::2]
    return cv2.resize(rgba, (size, size), interpolation=cv2.INTER_AREA), ids[..., 0].astype(np.int32) + 256 * ids[..., 1].astype(np.int32)

def to_ldr(bricks, path):
    """Writes the tower as an LDraw file (opens in LDView / Studio / BrickLink). Positions are exact: part origin = centre - R @ bbox centre."""
    lines = ['0 tower']
    for b in bricks:
        Ry, ctr = _place(b); x, y, z = ctr - Ry @ _dims(b['part'])[0]
        lines.append(f"1 {b['color']} {x:g} {y:g} {z:g} " + ' '.join(f'{v:g}' for v in Ry.ravel()) + f" {b['part']}.dat")
    open(path, 'w').write('\n'.join(lines) + '\n')
