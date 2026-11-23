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
