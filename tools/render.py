"""Renders the profile's films: a particle engine with a perspective camera,
depth of field, bloom and lens dispersion, encoded as looping animated WebP.

    pip install numpy pillow
    python tools/render.py            # everything
    python tools/render.py hero 0.6   # one still frame at loop position 0.6
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "tools" / "fonts" / "Unbounded-VF.ttf"
OUT = ROOT / "assets"
FPS = 25
TAU = np.pi * 2


# ─── helpers ────────────────────────────────────────────────────────────────


def font(size, wght):
    f = ImageFont.truetype(str(FONT), size)
    f.set_variation_by_axes([wght])
    return f


def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def splat(w, h, x, y, v):
    """Bilinear point splat into a w×h float buffer."""
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    m = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
    x0, y0, fx, fy, v = x0[m], y0[m], fx[m], fy[m], v[m]
    i = y0 * w + x0
    n = w * h
    buf = (np.bincount(i, v * (1 - fx) * (1 - fy), n) + np.bincount(i + 1, v * fx * (1 - fy), n)
           + np.bincount(i + w, v * (1 - fx) * fy, n) + np.bincount(i + w + 1, v * fx * fy, n))
    return buf.reshape(h, w)


def box(a, r, axis):
    if r < 1:
        return a
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis)
    hi = np.take(c, range(2 * r + 1, c.shape[axis]), axis=axis)
    lo = np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis=axis)
    return (hi - lo) / (2 * r + 1)


def blur(a, r):
    for _ in range(3):
        a = box(box(a, r, 0), r, 1)
    return a


def down(a, k):
    h, w = a.shape
    return a[: h // k * k, : w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def up(a, w, h):
    return np.asarray(Image.fromarray(a.astype(np.float32), "F").resize((w, h), Image.BICUBIC))


def zoom(a, s):
    """Scale a float image about its centre (for lens dispersion)."""
    h, w = a.shape
    img = Image.fromarray(a.astype(np.float32), "F")
    nw, nh = round(w * s), round(h * s)
    img = img.resize((nw, nh), Image.BILINEAR)
    l, t = (nw - w) // 2, (nh - h) // 2
    return np.asarray(img.crop((l, t, l + w, t + h)))


class Lens:
    def __init__(self, w, h, focal, dist, exposure=1.0, focus=None, dof=700.0):
        self.w, self.h, self.f, self.d = w, h, focal, dist
        self.exposure, self.focus, self.dof = exposure, focus or dist, dof
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
        self.vignette = (1 - 0.42 * smooth(r / 1.25)).astype(np.float32)

    def project(self, X, Y, Z):
        zc = Z + self.d
        s = self.f / np.maximum(zc, 1)
        return self.w / 2 + X * s, self.h / 2 + Y * s, zc

    def develop(self, X, Y, Z, v):
        """Particles in world space → tone-mapped float RGB (h, w, 3)."""
        w, h = self.w, self.h
        sx, sy, zc = self.project(X, Y, Z)
        v = v * np.clip(self.d / zc, 0.3, 2.2) ** 1.4
        coc = np.clip(np.abs(zc - self.focus) / self.dof, 0, 1)
        sharp = splat(w, h, sx, sy, v * (1 - coc))
        soft = splat(w // 4, h // 4, sx / 4, sy / 4, v * coc * 0.06)
        sharp = 0.45 * sharp + 0.55 * blur(sharp, 1)
        base = sharp + up(blur(soft, 2), w, h)
        b1 = up(blur(down(base, 4), 4), w, h)
        b2 = up(blur(down(base, 8), 9), w, h)
        glow = 0.55 * b1 + 0.35 * b2
        # dispersion lives only in the glow, so the core light stays clean
        g_r, g_b = zoom(glow, 1.006), zoom(glow, 0.994)
        tone = lambda x: 1 - np.exp(-x * self.exposure)
        rgb = np.stack([tone(base + g_r) * 0.98, tone(base + glow) * 0.99, tone(base + g_b)], axis=-1)
        return rgb * self.vignette[..., None]


def to_image(rgb, radius=28):
    a = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    img = Image.fromarray(a, "RGB").convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius, fill=255)
    img.putalpha(mask)
    return img


def tracked(draw, xy, text, f, fill, tracking, anchor="l"):
    """Draw letter-spaced text; anchor l / m / r on x."""
    widths = [draw.textlength(ch, font=f) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x, y = xy
    if anchor == "m":
        x -= total / 2
    elif anchor == "r":
        x -= total
    for ch, cw in zip(text, widths):
        draw.text((x, y), ch, font=f, fill=fill)
        x += cw + tracking
    return total


def overlay(img, fn):
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    fn(ImageDraw.Draw(layer))
    return Image.alpha_composite(img, layer)


def text_points(text, n, w, h, size, wght, cy, rng):
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    f = font(size, wght)
    l, t, r, b = d.textbbox((0, 0), text, font=f)
    d.text(((w - (r - l)) / 2 - l, cy - (b - t) / 2 - t), text, font=f, fill=255)
    a = np.asarray(mask)
    ys, xs = np.nonzero(a > 127)
    k = rng.integers(0, len(xs), n)
    return xs[k] + rng.random(n), ys[k] + rng.random(n)


def save(name, frames):
    path = OUT / f"{name}.webp"
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS,
                   loop=0, quality=64, method=6, lossless=False)
    print(f"{path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


# ─── film 1 · genesis ───────────────────────────────────────────────────────


def hero(u_values):
    W, H = 1280, 600
    T = 8.0
    rng = np.random.default_rng(11)
    N = 90000
    lens = Lens(W, H, focal=900, dist=1500, exposure=1.15, dof=900)

    # the galaxy
    r = 30 + 1500 * rng.random(N) ** 2.0
    arm = rng.integers(0, 2, N) * TAU / 2
    th0 = arm + np.log1p(r / 60) * 1.9 + rng.normal(0, 0.16, N) * (1 + r / 900)
    hgt = rng.normal(0, 1, N) * (10 + r * 0.03)
    k = np.where(r < 250, 3, np.where(r < 700, 2, 1))
    tilt = 0.5
    core = 1 + 2.5 * np.exp(-r / 160)

    # the name
    tx, ty = text_points("GHOVINDH", N, W, H, 168, 820, H / 2 - 18, rng)
    TX = (tx - W / 2) * lens.d / lens.f
    TY = (ty - H / 2) * lens.d / lens.f
    TZ = rng.normal(0, 7, N)
    phase = rng.random(N) * TAU
    rng_text = rng.random(N)

    lag_in = rng.random(N) * 0.09
    lag_out = rng.random(N) * 0.08
    base = 0.10 + 0.45 * rng.random(N) ** 2.2

    frames = []
    for u in u_values:
        t = u * T
        th = th0 + TAU * k * u
        gx, gz = r * np.cos(th), r * np.sin(th)
        gy = hgt + 10 * np.sin(TAU * u + r * 0.01)
        GY = gy * np.cos(tilt) - gz * np.sin(tilt)
        GZ = gy * np.sin(tilt) + gz * np.cos(tilt)

        s = smooth((u - 0.16 - lag_in) / 0.2) * (1 - smooth((u - 0.80 - lag_out) / 0.13))
        e = s * s * (3 - 2 * s)

        yaw = 0.22 * (u - 0.58)
        bx = TX * np.cos(yaw) + TZ * np.sin(yaw)
        bz = -TX * np.sin(yaw) + TZ * np.cos(yaw) + 5 * np.sin(TAU * 2 * u + phase)

        X = gx + (bx - gx) * e
        Y = GY + (TY - GY) * e
        Z = GZ + (bz - GZ) * e
        swirl = 2.4 * e * (1 - e)
        c, sn = np.cos(swirl), np.sin(swirl)
        X, Z = X * c - Z * sn, X * sn + Z * c

        v = (base * core) * (1 - e) + (0.36 + 0.1 * rng_text) * e * 2.6
        sweep_x = (u - 0.47) / 0.3 * 1.5 - 0.25
        sx = (bx / (lens.d / lens.f) + W / 2) / W
        v = v * (1 + 3.5 * e * np.exp(-((sx - sweep_x) / 0.035) ** 2))
        v = v * (0.85 + 0.15 * np.sin(TAU * 3 * u + phase))

        img = to_image(lens.develop(X, Y, Z, v))

        hold = float(smooth((u - 0.43) / 0.07) * (1 - smooth((u - 0.79) / 0.05)))
        tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t * FPS) % FPS):02d}"

        def hud(d):
            small = font(12, 400)
            dim = (255, 255, 255, 105)
            tracked(d, (40, 34), "GHOVKG02", small, dim, 3.2)
            tracked(d, (W - 40, 34), "12.97°N  77.59°E", small, dim, 3.2, "r")
            d.ellipse((40, H - 45, 47, H - 38), fill=(255, 255, 255, 150 if int(t * 2) % 2 == 0 else 40))
            tracked(d, (58, H - 48), "LIVE", small, dim, 3.2)
            tracked(d, (W - 40, H - 48), tc, small, dim, 3.2, "r")
            if hold > 0.01:
                tracked(d, (W / 2, H / 2 + 92), "SYSTEMS  ENGINEER   ·   FOUNDER   ·   BENGALURU",
                        font(13, 380), (255, 255, 255, int(200 * hold)), 4.2, "m")

        frames.append(overlay(img, hud))
    return frames


# ─── film 2 · orbit ─────────────────────────────────────────────────────────

SYSTEMS = ["HR & PAYROLL", "ENERGY TRADING", "POWER UTILITIES", "SOLAR EPC", "HOSPITALITY",
           "TRAVEL & STAYS", "RETAIL ANALYTICS", "AUTOMOTIVE", "ART & COLLECTIBLES", "AGRI COMMERCE",
           "EVENTS & EXPOS", "FESTIVE RETAIL", "INTERIORS & SUPPLY", "CITY SERVICES", "FIELD SALES"]


def orbit(u_values):
    W, H = 1280, 720
    T = 12.0
    rng = np.random.default_rng(5)
    lens = Lens(W, H, focal=1000, dist=1500, exposure=1.25, dof=1100)
    R = 175.0

    # planet: fibonacci sphere with noisy continents
    n = 26000
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    gold = np.pi * (3 - np.sqrt(5))
    lon = gold * i
    px, py, pz = np.sin(phi) * np.cos(lon), np.cos(phi), np.sin(phi) * np.sin(lon)
    land = (np.sin(px * 3.1 + 1) + np.sin(py * 4.3 + 2) * 0.8 + np.sin(pz * 5.7) * 0.6
            + np.sin((px + pz) * 9) * 0.3)
    pv = np.where(land > 0.55, 1.0, 0.16) * (0.75 + 0.25 * rng.random(n))

    # rings
    rn = 22000
    band = rng.choice([1.62, 1.70, 1.74, 1.86, 1.95, 2.02], rn, p=[.12, .22, .2, .14, .2, .12])
    rr = R * (band + rng.normal(0, 0.012, rn))
    rth = rng.random(rn) * TAU
    rv = 0.06 + 0.08 * rng.random(rn)
    ring_tilt = 0.42
    ring_k = np.maximum(1, np.round(2 * (R * 1.6 / rr) ** 1.5))

    # dust
    dn = 6000
    dust = rng.normal(0, 1, (3, dn)) * np.array([[900], [420], [700]])
    dv = 0.03 + 0.08 * rng.random(dn) ** 3

    # satellites
    m = len(SYSTEMS)
    s_r = R * np.linspace(1.6, 3.4, m)[rng.permutation(m)]
    s_inc = rng.uniform(-0.75, 0.75, m)
    s_node = rng.uniform(0, TAU, m)
    s_ph = np.linspace(0, TAU, m, endpoint=False) * 3 % TAU
    s_k = np.array([1, -1] * 8)[:m]

    def sat_pos(uu):
        a = s_ph[:, None] + TAU * s_k[:, None] * np.atleast_1d(uu)[None, :]
        x, y, z = s_r[:, None] * np.cos(a), np.zeros_like(a), s_r[:, None] * np.sin(a)
        ci, si = np.cos(s_inc)[:, None], np.sin(s_inc)[:, None]
        y, z = y * ci - z * si, y * si + z * ci
        cn, sn = np.cos(s_node)[:, None], np.sin(s_node)[:, None]
        return x * cn - z * sn, y, x * sn + z * cn

    axial = 0.38
    view = -0.32  # camera pitch

    def to_cam(x, y, z):
        c, s = np.cos(view), np.sin(view)
        return x, y * c - z * s, y * s + z * c

    frames = []
    for u in u_values:
        a = TAU * u
        # planet spin about a tilted axis
        x1 = px * np.cos(a) + pz * np.sin(a)
        z1 = -px * np.sin(a) + pz * np.cos(a)
        y1 = py
        y2 = y1 * np.cos(axial) - x1 * np.sin(axial)
        x2 = y1 * np.sin(axial) + x1 * np.cos(axial)
        light = np.clip(0.25 + 0.95 * (-0.55 * x2 - 0.35 * y2 - 0.75 * z1), 0.05, 1.25)
        rim = np.clip(1 - np.abs(z1), 0, 1) ** 4 * 0.5
        PX, PY, PZ = to_cam(x2 * R, y2 * R, z1 * R)
        front = PZ < 0
        pvv = (pv * light + rim * 0.25) * np.where(front, 1.0, 0.0)

        # rings, rotating; hide the far side behind the planet
        ang = rth + TAU * ring_k * u
        qx, qz = rr * np.cos(ang), rr * np.sin(ang)
        qy = qz * np.sin(ring_tilt)
        qz = qz * np.cos(ring_tilt)
        QX, QY, QZ = to_cam(qx, qy, qz)
        hidden = (QZ > 0) & (np.hypot(QX, QY) < R)
        rvv = rv * np.where(hidden, 0.0, 1.0)

        # satellites with trails
        tr = np.linspace(0, -0.06, 26)
        sx_, sy_, sz_ = sat_pos(u + tr)
        SX, SY, SZ = to_cam(sx_, sy_, sz_)
        fade = np.linspace(1, 0, len(tr)) ** 2
        sv = np.tile(fade * 0.9, (m, 1))
        sv[:, 0] = 6.0
        occl = (SZ > 0) & (np.hypot(SX, SY) < R)
        sv = sv * np.where(occl, 0.0, 1.0)

        # data packets planet → satellite
        hx, hy, hz = sat_pos(u)
        HX, HY, HZ = to_cam(hx, hy, hz)
        pk = (u * 3 + np.arange(m) / m) % 1
        L = np.stack([HX[:, 0], HY[:, 0], HZ[:, 0]])
        nrm = L / np.linalg.norm(L, axis=0) * R
        P = nrm + (L - nrm) * pk
        pkv = np.where((P[2] > 0) & (np.hypot(P[0], P[1]) < R), 0, 2.2)
        line_t = np.linspace(0, 1, 60)
        LP = nrm[:, :, None] + (L - nrm)[:, :, None] * line_t[None, None, :]
        lv = np.full(LP.shape[1:], 0.035)
        lv = np.where((LP[2] > 0) & (np.hypot(LP[0], LP[1]) < R), 0, lv)

        X = np.concatenate([PX, QX, dust[0], SX.ravel(), P[0], LP[0].ravel()])
        Y = np.concatenate([PY, QY, dust[1], SY.ravel(), P[1], LP[1].ravel()])
        Z = np.concatenate([PZ, QZ, dust[2], SZ.ravel(), P[2], LP[2].ravel()])
        V = np.concatenate([pvv, rvv, dv, sv.ravel(), pkv, lv.ravel()])
        img = to_image(lens.develop(X, Y + 10, Z, V))

        sx, sy, zc = lens.project(HX[:, 0], HY[:, 0] + 10, HZ[:, 0])
        vis = ~((HZ[:, 0] > 0) & (np.hypot(HX[:, 0], HY[:, 0]) < R * 1.02))

        ly = sy.copy()
        for side in (1, -1):
            idx = [j for j in np.argsort(sy) if (sx[j] > W / 2) == (side > 0)]
            for p_, q_ in zip(idx, idx[1:]):
                if ly[q_] - ly[p_] < 21:
                    ly[q_] = ly[p_] + 21

        def hud(d):
            lab = font(12, 450)
            for j in range(m):
                depth = float(np.clip(1.15 - (zc[j] - lens.d) / 700, 0.35, 1))
                alpha = int(235 * depth * (1.0 if vis[j] else 0.0))
                if alpha < 8:
                    continue
                label = f"{j + 1:02d}  {SYSTEMS[j]}"
                lw = sum(d.textlength(ch, font=lab) for ch in label) + 3.0 * len(label) + 40
                side = 1 if sx[j] > W / 2 else -1
                if side > 0 and sx[j] + lw > W - 30:
                    side = -1
                elif side < 0 and sx[j] - lw < 30:
                    side = 1
                x0 = sx[j] + side * 16
                d.ellipse((sx[j] - 7, sy[j] - 7, sx[j] + 7, sy[j] + 7), outline=(255, 255, 255, alpha // 2), width=1)
                d.line((sx[j] + side * 9, sy[j], x0 + side * 6, ly[j], x0 + side * 14, ly[j]),
                       fill=(255, 255, 255, alpha // 2), width=1)
                tracked(d, (x0 + side * 20, ly[j] - 8), label, lab,
                        (255, 255, 255, alpha), 3.0, "l" if side > 0 else "r")
            small = font(12, 400)
            dim = (255, 255, 255, 110)
            tracked(d, (40, 34), "15 INDUSTRIES  ·  15 SYSTEMS IN PRODUCTION", small, dim, 3.2)
            tracked(d, (W - 40, 34), "GHOVKG02", small, dim, 3.2, "r")
            tracked(d, (40, H - 48), "1.9K COMMITS", small, dim, 3.2)
            tracked(d, (W - 40, H - 48), "WEB  ·  iOS  ·  ANDROID  ·  EDGE  ·  0 LOCK-IN", small, dim, 3.2, "r")

        frames.append(overlay(img, hud))
    return frames


# ─── film 3 · arsenal ───────────────────────────────────────────────────────

TOOLS = ["TypeScript", "JavaScript", "Python", "Java", "C++", "Bash", "SQL",
         "React", "Next.js", "React Native", "Expo", "Capacitor", "Tailwind", "Vite", "Android", "iOS",
         "Node.js", "Express", "Django", "Django REST", "FastAPI", "Payload CMS", "Prisma", "Auth.js",
         "Socket.io", "BullMQ", "Celery", "WebSockets", "REST APIs",
         "PostgreSQL", "Redis", "MongoDB", "SQLite", "Supabase",
         "PyTorch", "TensorFlow", "YOLOv8", "OpenCV", "ByteTrack", "LLMs", "RAG", "Embeddings", "ML Kit",
         "Docker", "Linux", "Nginx", "Caddy", "PM2", "GitHub Actions", "AWS", "Cloudflare", "Vercel",
         "Playwright", "Vitest", "Razorpay", "Figma", "Git"]


def arsenal(u_values):
    W, H = 1280, 640
    T = 16.0
    rng = np.random.default_rng(3)
    L = 3200.0
    lens = Lens(W, H, focal=760, dist=800, exposure=1.3, focus=900, dof=1600)

    # star field, streaking toward the camera
    n = 9000
    stx = rng.uniform(-2400, 2400, n)
    sty = rng.uniform(-1300, 1300, n)
    stz0 = rng.uniform(0, L, n)
    stv = 0.25 + 1.4 * rng.random(n) ** 3
    core_n = 5000
    core = rng.normal(0, 1, (3, core_n)) * np.array([[50], [32], [300]])
    core[2] = np.abs(core[2]) + L * 0.8

    # every tool on its own lane around the tunnel
    lanes = TOOLS + TOOLS
    m = len(lanes)
    order = rng.permutation(m)
    wz0 = (np.arange(m)[order] + rng.random(m) * 0.6) / m * L
    rank = np.argsort(wz0)
    wang = np.empty(m)
    wang[rank] = np.arange(m) * np.pi * (3 - np.sqrt(5)) + rng.normal(0, 0.15, m)
    wrad = rng.uniform(340, 860, m)
    wx, wy = np.cos(wang) * wrad * 1.5, np.sin(wang) * wrad * 0.85
    fonts = {}

    frames = []
    for u in u_values:
        t = u * T
        z = (stz0 - L * u) % L + 20
        streak = np.linspace(0, 160, 12)
        Z = (z[:, None] + streak[None, :]).ravel()
        X = np.repeat(stx, 12)
        Y = np.repeat(sty, 12)
        V = (stv[:, None] * np.linspace(1, 0.05, 12)[None, :]).ravel()
        V = V * smooth((L - Z) / 600)
        cv = np.full(core_n, 1.1) * (0.8 + 0.2 * np.sin(TAU * 4 * u + np.arange(core_n)))
        img = to_image(lens.develop(np.concatenate([X, core[0]]), np.concatenate([Y, core[1]]),
                                    np.concatenate([Z, core[2]]) - lens.d, np.concatenate([V, cv])))

        wz = (wz0 - L * u) % L + 60

        def hud(d):
            for j in np.argsort(-wz):
                zz = wz[j]
                size = int(np.clip(40 * 760 / zz, 8, 140))
                a = float(smooth((L * 0.86 - zz) / 700) * smooth((zz - 90) / 220))
                x = W / 2 + wx[j] * 760 / zz
                y = H / 2 + wy[j] * 760 / zz
                a *= float(smooth((y - 70) / 40) * smooth((H - 80 - y) / 40))
                if a < 0.02:
                    continue
                f = fonts.setdefault(size, font(size, 520))
                tw = d.textlength(lanes[j], font=f)
                d.text((x - tw / 2, y - size / 2), lanes[j], font=f, fill=(255, 255, 255, int(235 * a)))
            small = font(12, 400)
            dim = (255, 255, 255, 120)
            tracked(d, (40, 34), f"ARSENAL  ·  {len(TOOLS)} TOOLS  ·  7 LANGUAGES", small, dim, 3.2)
            tracked(d, (W - 40, 34), "GHOVKG02", small, dim, 3.2, "r")
            tracked(d, (40, H - 48), "INTERFACE  ·  SERVICES  ·  DATA  ·  AI  ·  OPERATIONS", small, dim, 3.2)
            tc = f"{int(t // 60):02d}:{int(t % 60):02d}:{int((t * FPS) % FPS):02d}"
            tracked(d, (W - 40, H - 48), tc, small, dim, 3.2, "r")

        frames.append(overlay(img, hud))
    return frames


# ─── links ──────────────────────────────────────────────────────────────────


def links():
    """Two link plates per theme, rendered at 2x for retina."""
    for name, label in (("linkedin", "LINKEDIN"), ("email", "EMAIL")):
        for theme, ink in (("dark", (240, 240, 240, 255)), ("light", (14, 14, 14, 255))):
            k = 2
            img = Image.new("RGBA", (220 * k, 52 * k), (0, 0, 0, 0))
            d = ImageDraw.Draw(img)
            d.rounded_rectangle((k, k, img.width - k - 1, img.height - k - 1), 26 * k,
                                outline=ink[:3] + (90,), width=k)
            f = font(12 * k, 500)
            wd = tracked(d, (img.width / 2 - 10 * k, 17 * k), label, f, ink, 4 * k, "m")
            ax = img.width / 2 - 10 * k + wd / 2 + 12 * k
            d.line((ax, 33 * k, ax + 9 * k, 24 * k), fill=ink, width=k + 1)
            d.line((ax + 2 * k, 24 * k, ax + 9 * k, 24 * k, ax + 9 * k, 31 * k), fill=ink, width=k + 1)
            img.save(OUT / f"{name}-{theme}.png", optimize=True)


# ─── main ───────────────────────────────────────────────────────────────────

FILMS = {"hero": (hero, 8.0), "orbit": (orbit, 12.0), "arsenal": (arsenal, 16.0)}


def main():
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) == 3:
        name, u = sys.argv[1], float(sys.argv[2])
        FILMS[name][0]([u])[0].save(OUT / f"_still-{name}.png")
        return
    if sys.argv[1:] == ["links"] or not sys.argv[1:]:
        links()
        if sys.argv[1:]:
            return
    for name in sys.argv[1:] or FILMS:
        fn, T = FILMS[name]
        n = int(T * FPS)
        save(name, fn(np.arange(n) / n))


if __name__ == "__main__":
    main()
