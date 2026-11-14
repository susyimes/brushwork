"""The picture: Astra in the Obsidian Gothic dress in front of an eclipsed moon.

This module is the underdrawing.  Every shape is constructed from hand-placed coordinates
(canvas units, 1000 wide x 1250 tall); no image is ever loaded.  It produces what a painter
carries in their head before touching the canvas:

  target   the colour each place should end up (linear RGB)
  region   which thing each place belongs to
  vx, vy   which way the brush should travel there
  rmin     how small a brush that place deserves
  sheen    how much interference medium the blacks get, and `film` its optical thickness
  geo      the drawing's lines (hair locks, lashes, lace, feathers ...) for the fine work
"""
import math
from types import SimpleNamespace

import cv2
import numpy as np

from brushwork import color as C
from brushwork.engine import value_noise
from brushwork.geom import (Raster, catmull, ellipse_pts, lock_profile, mirror_x, offset, resample,
                            ribbon, tangents)

WU, HU = 1000.0, 1250.0
E = np.array([488.0, 545.0])      # midpoint between the eyes
U = 150.0                         # one face unit: half the face width
TILT = math.radians(-5.0)         # head leans a little to her right
BO = np.array([500.0, 518.0])     # body frame origin (untilted)
HALO_C = np.array([505.0, 498.0])
HALO_R = 402.0
HALO_IN_C = HALO_C + np.array([16.0, 20.0])
HALO_IN_R = 385.0

BG, HAIR_BACK, SKIN, HAIR, DRESS, LACE, BOW, EYE, PIN, PLACKET, BUTTON, FEATHER = range(12)

# ---- palette of the underdrawing (8-bit sRGB) ------------------------------------------------
P = {k: C.rgb8(*v) for k, v in dict(
    skin_light=(255, 239, 229), skin_mid=(251, 224, 214), skin_shadow=(230, 188, 200),
    skin_deep=(198, 152, 180), blush=(255, 176, 178),
    hair_light=(253, 251, 247), hair_mid=(228, 226, 240), hair_shadow=(178, 178, 212),
    hair_deep=(122, 122, 168), hair_back=(126, 126, 172), hair_back_deep=(70, 68, 116),
    hair_cyan=(186, 226, 238), hair_pink=(246, 206, 226),
    lace=(247, 239, 224), lace_shadow=(198, 184, 180), ivory=(241, 235, 223),
    bow_mid=(152, 22, 44), bow_dark=(70, 8, 30), bow_light=(226, 86, 94),
    black=(11, 10, 17), lash=(34, 24, 48), lash_tip=(96, 40, 72),
    sclera=(247, 244, 247), sclera_shadow=(196, 192, 216),
    iris_top=(54, 56, 86), iris_mid=(118, 124, 156), iris_bot=(182, 192, 220),
    iris_ring=(52, 54, 92), pupil=(24, 20, 42), white=(255, 255, 255),
    lip=(214, 122, 138), line=(120, 70, 104), brow=(150, 146, 172),
    pin=(14, 12, 24), button=(22, 18, 34),
).items()}


def hd(p):
    """Head-local (face units, tilted) -> canvas units."""
    p = np.asarray(p, np.float64)
    c, s = math.cos(TILT), math.sin(TILT)
    return E + U * np.stack([p[..., 0] * c - p[..., 1] * s, p[..., 0] * s + p[..., 1] * c], -1)


def bd(p):
    """Body frame (face units, upright) -> canvas units."""
    return BO + U * np.asarray(p, np.float64)


# ---- hair: every lock is a centre line plus a width ---------------------------------------
def _locks(rng):
    L = []

    def add(name, layer, ctrl, w, root=1.0, p=2.2, q=0.85):
        t = np.linspace(0, 1, 56)
        cl = hd(resample(catmull(ctrl, n=14), n=56))
        L.append(dict(name=name, layer=layer, center=cl, width=w * U * lock_profile(t, p=p, q=q, root=root),
                      tone=float(rng.uniform(0.0, 0.42)), t=t))

    # behind the neck and shoulders
    add('K1', 'back', [(-0.9, -0.6), (-1.75, 0.2), (-1.85, 0.95), (-1.60, 1.50)], 0.78)
    add('K2', 'back', [(-0.7, -0.3), (-1.35, 0.6), (-1.30, 1.25), (-1.02, 1.72)], 0.78)
    add('K3', 'back', [(-0.4, 0.0), (-0.85, 0.8), (-0.75, 1.40), (-0.52, 1.80)], 0.72)
    add('K4', 'back', [(0.4, 0.0), (0.85, 0.8), (0.78, 1.40), (0.56, 1.82)], 0.72)
    add('K5', 'back', [(0.7, -0.3), (1.38, 0.6), (1.34, 1.25), (1.08, 1.74)], 0.78)
    add('K6', 'back', [(0.9, -0.6), (1.78, 0.2), (1.88, 0.95), (1.66, 1.46)], 0.78)
    # sides, outermost first
    add('SL1', 'side', [(-0.75, -1.45), (-1.62, -0.75), (-1.98, 0.15), (-2.06, 0.80), (-2.20, 1.20)], 0.62)
    add('SR1', 'side', [(0.80, -1.42), (1.66, -0.72), (2.00, 0.18), (2.04, 0.85), (2.22, 1.13)], 0.60)
    add('SL2', 'side', [(-0.55, -1.35), (-1.40, -0.55), (-1.72, 0.45), (-1.70, 1.05), (-1.52, 1.52)], 0.66)
    add('SR2', 'side', [(0.60, -1.35), (1.44, -0.52), (1.74, 0.48), (1.74, 1.08), (1.60, 1.56)], 0.66)
    add('SL3', 'side', [(-0.50, -1.2), (-1.22, -0.35), (-1.42, 0.45), (-1.36, 1.00), (-1.18, 1.40)], 0.52)
    add('SR3', 'side', [(0.55, -1.2), (1.26, -0.32), (1.44, 0.48), (1.40, 1.02), (1.24, 1.44)], 0.52)
    add('SL4', 'side', [(-0.45, -1.3), (-1.02, -0.55), (-1.14, 0.15), (-1.08, 0.70), (-0.95, 1.08)], 0.40)
    add('SR4', 'side', [(0.50, -1.3), (1.06, -0.52), (1.17, 0.18), (1.12, 0.72), (1.01, 1.10)], 0.40)
    # tufts breaking the outline, outward flicks, curls turning in to the neck
    add('F1', 'side', [(-1.50, -0.1), (-1.86, 0.62), (-2.30, 0.95)], 0.40, root=0.0)
    add('F2', 'side', [(1.55, -0.05), (1.90, 0.66), (2.34, 0.90)], 0.40, root=0.0)
    add('I1', 'side', [(-1.10, 0.25), (-1.04, 0.95), (-0.72, 1.36)], 0.40, root=0.0)
    add('I2', 'side', [(1.14, 0.25), (1.10, 0.98), (0.80, 1.38)], 0.40, root=0.0)
    # fringe: nine blades, outer ones first, the middle one down between the eyes
    add('B1', 'bang', [(-0.34, -1.64), (-0.86, -1.18), (-1.02, -0.60), (-0.93, -0.10)], 0.50, p=3.0, q=0.8)
    add('B5', 'bang', [(0.42, -1.64), (0.88, -1.16), (1.02, -0.60), (0.90, -0.14)], 0.48, p=3.0, q=0.8)
    add('B1b', 'bang', [(-0.22, -1.67), (-0.64, -1.14), (-0.78, -0.60), (-0.72, -0.25)], 0.42, p=3.0, q=0.8)
    add('B5b', 'bang', [(0.32, -1.67), (0.64, -1.14), (0.76, -0.62), (0.70, -0.29)], 0.40, p=3.0, q=0.8)
    add('B2', 'bang', [(-0.10, -1.70), (-0.42, -1.12), (-0.54, -0.56), (-0.45, -0.16)], 0.44, p=3.0, q=0.8)
    add('B4b', 'bang', [(0.24, -1.69), (0.42, -1.12), (0.50, -0.60), (0.43, -0.18)], 0.40, p=3.0, q=0.8)
    add('B2b', 'bang', [(-0.02, -1.71), (-0.20, -1.10), (-0.30, -0.58), (-0.24, -0.27)], 0.36, p=3.0, q=0.8)
    add('B4', 'bang', [(0.16, -1.71), (0.22, -1.10), (0.26, -0.58), (0.18, -0.23)], 0.38, p=3.0, q=0.8)
    add('B3', 'bang', [(0.08, -1.72), (0.00, -1.10), (-0.10, -0.50), (0.03, 0.06)], 0.40, p=3.0, q=0.8)
    return L


def _eye(sx):
    def m(pts):
        a = np.array(pts, np.float64)
        a[:, 0] *= sx
        return a

    up = catmull(m([(0.215, 0.10), (0.30, -0.040), (0.42, -0.120), (0.56, -0.136), (0.70, -0.100),
                    (0.79, -0.035), (0.835, 0.045)]), n=10)
    lo = catmull(m([(0.215, 0.10), (0.30, 0.18), (0.44, 0.228), (0.60, 0.232), (0.74, 0.185),
                    (0.835, 0.045)]), n=10)
    top = catmull(m([(0.20, 0.07), (0.29, -0.110), (0.42, -0.205), (0.56, -0.226), (0.74, -0.180),
                     (0.86, -0.065), (0.93, 0.0)]), n=10)
    ic = np.array([0.515 * sx, 0.040])
    return SimpleNamespace(
        sx=sx,
        upper=hd(up), lower=hd(lo), lash_top=hd(top),
        opening=hd(np.vstack([up, lo[::-1]])),
        lash=hd(np.vstack([up, top[::-1]])),
        crease=hd(catmull(m([(0.27, -0.19), (0.40, -0.295), (0.57, -0.325), (0.73, -0.265), (0.84, -0.165)]), n=10)),
        lower_line=hd(catmull(m([(0.55, 0.236), (0.64, 0.228), (0.74, 0.185), (0.805, 0.10)]), n=10)),
        spikes=[hd(m([(0.80, -0.115), (0.895, -0.160)])), hd(m([(0.85, -0.04), (0.955, -0.07)]))],
        iris_c=hd(ic), iris_rx=0.186 * U, iris_ry=0.226 * U,
        pupil_c=hd(ic + np.array([0.0, -0.005])), pupil_rx=0.086 * U, pupil_ry=0.112 * U,
        hl1=hd(ic + np.array([-0.078, -0.098])), hl1_r=(0.064 * U, 0.05 * U),
        hl2=hd(ic + np.array([0.088, 0.105])), hl2_r=0.025 * U,
        brow=hd(catmull(m([(0.22, -0.47), (0.40, -0.555), (0.60, -0.575), (0.82, -0.50)]), n=10)),
    )


def _feather(base, tip, bend, wmax, rng, nb=60):
    """A flight feather: rachis polyline, vane outline, and the barbs as short curves."""
    base = np.asarray(base, np.float64)
    tip = np.asarray(tip, np.float64)
    d = tip - base
    Lr = np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]]) / Lr
    mid = (base + tip) / 2 + nrm * bend * Lr
    rach = resample(catmull([base, mid, tip], n=30), n=80)
    t = np.linspace(0, 1, 80)
    tau = np.clip((t - 0.13) / 0.87, 0, 1)
    prof = tau ** 0.55 * (1 - tau) ** 0.42
    prof /= prof.max()
    wl = wmax * prof * 1.0
    wr = wmax * prof * 0.72
    tn = tangents(rach)
    nn = np.stack([-tn[:, 1], tn[:, 0]], 1)
    outline = np.vstack([rach + nn * wl[:, None], (rach - nn * wr[:, None])[::-1]])
    barbs = []
    for side, ww in ((1, wl), (-1, wr)):
        for k in np.linspace(12, 77, nb).astype(int):
            if ww[k] < 0.02 * wmax:
                continue
            ang = math.radians(rng.uniform(38, 50))
            dirv = tn[k] * math.cos(ang) + side * nn[k] * math.sin(ang)
            ln = ww[k] / math.sin(ang) * rng.uniform(0.9, 1.05)
            p0 = rach[k]
            p1 = p0 + dirv * ln * 0.55 + tn[k] * ln * 0.03
            p2 = p0 + dirv * ln + tn[k] * ln * 0.14
            barbs.append(dict(pl=resample(catmull([p0, p1, p2], n=8), n=12), t=float(t[k]), side=side))
    return SimpleNamespace(rachis=rach, outline=outline, barbs=barbs, wmax=wmax, length=Lr)


def put(target, m, col):
    target *= (1.0 - m)[..., None]
    target += m[..., None] * col


def toward(target, m, col, k=1.0):
    """Mix the target toward a colour by mask*k."""
    w = np.clip(m * k, 0, 1)[..., None]
    target += (col - target) * w


def tri(u):
    """Triangle wave 0..1..0: a phase that never jumps."""
    f = u - np.floor(u)
    return 1.0 - np.abs(2.0 * f - 1.0)


FILM_LO, FILM_SPAN = 1075.0, 175.0


def film_path(phase):
    """Optical thickness of the interference film along one unbroken walk through the cool
    jewel colours: magenta - purple - blue - teal (second order, 1075..1250 nm), then on
    through aqua - cyan - azure - violet (first order, 760..580 nm).  The yellow-greens and
    oranges in between, which turn to mud when dim on black, are stepped over."""
    u = 2.0 * tri(np.asarray(phase, np.float64))
    return np.where(u < 1.0, FILM_LO + FILM_SPAN * u, 760.0 - 180.0 * (u - 1.0))


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-12), 0, 1)
    return t * t * (3 - 2 * t)


def build(GW, GH, seed=11):
    s = GW / WU
    ra = Raster(GW, GH, s)
    rng = np.random.default_rng(seed)
    X, Y = ra.X, ra.Y
    geo = SimpleNamespace()
    masks = {}

    target = np.empty((GH, GW, 3), np.float32)
    region = np.zeros((GH, GW), np.uint8)
    vx = np.ones((GH, GW), np.float32)
    vy = np.zeros((GH, GW), np.float32)
    sheen = np.zeros((GH, GW), np.float32)
    film = np.full((GH, GW), 700.0, np.float32)

    # local frames as pixel fields
    ct, st = math.cos(TILT), math.sin(TILT)
    dxh = (X - E[0]) / U
    dyh = (Y - E[1]) / U
    Xl = dxh * ct + dyh * st
    Yl = -dxh * st + dyh * ct
    Xb = (X - BO[0]) / U
    Yb = (Y - BO[1]) / U

    # =========================================================== background: the eclipse
    hx = X - HALO_C[0]
    hy = Y - HALO_C[1]
    rr = np.sqrt(hx * hx + hy * hy) + 1e-6
    th = np.arctan2(hy, hx)
    n_lo = value_noise(GW, GH, 150 * s, rng, octaves=3)
    n_hi = value_noise(GW, GH, 46 * s, rng, octaves=3)
    n_w = value_noise(GW, GH, 220 * s, rng, octaves=2)
    out_t = sstep(HALO_R * 0.96, HALO_R * 1.7, rr)
    # brush direction: round the disc, then spiralling out and breaking into turbulence
    ang = th + math.pi / 2 + 0.38 * out_t + (n_lo - 0.5) * 2.6 * (0.18 + 0.82 * out_t)
    in_disc = rr < HALO_R
    vx[:] = np.cos(ang)
    vy[:] = np.sin(ang)
    # streaks of sheen that follow the swirl: noise sampled in polar coordinates
    tex = value_noise(1024, 256, 22, rng, octaves=3)
    tex = np.concatenate([tex, tex[:, ::-1]], 1)
    mu = (((th + 0.55 * np.log(rr / 40.0 + 1.0)) / (2 * math.pi)) % 1.0) * (tex.shape[1] - 1)
    mv = np.clip(np.log(rr / 30.0 + 1.0) / math.log(1200 / 30.0 + 1.0), 0, 1) * (tex.shape[0] - 1)
    streak = cv2.remap(tex, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR)
    ring_glow = np.exp(-((rr - HALO_R * 1.04) / (HALO_R * 0.16)) ** 2)
    sheen_bg = sstep(0.46, 0.74, 0.62 * streak + 0.38 * n_hi) * (0.30 + 0.70 * out_t) + 0.55 * ring_glow * sstep(0.35, 0.7, n_hi)
    sheen_bg *= np.where(in_disc, 0.42, 1.0)
    film_bg = film_path(0.60 * n_w + th / (2 * math.pi) + rr / 1500.0 + 0.25 * streak)
    sheen[:] = np.clip(sheen_bg, 0, 1)
    film[:] = film_bg
    lut = C.thinfilm_lut()
    tint = lut[np.clip((film_bg / 1800.0 * 1023).astype(np.int32), 0, 1023)]
    target[:] = P['black'] * (0.75 + 0.5 * n_hi[..., None]) + tint * (0.012 + 0.03 * sheen[..., None])
    masks['disc'] = in_disc.astype(np.float32)

    # crescent of the eclipse (the painter lays it with long arcs; here only for reference)
    r_in = np.sqrt((X - HALO_IN_C[0]) ** 2 + (Y - HALO_IN_C[1]) ** 2)
    cres = sstep(-1.5, 1.5, HALO_R - rr) * sstep(-1.5, 1.5, r_in - HALO_IN_R)
    masks['crescent'] = cres

    # =========================================================== feathers
    geo.feathers = [
        _feather((236, 40), (48, 236), 0.12, 34, rng, nb=64),
        _feather((972, 500), (912, 700), 0.13, 22, rng, nb=46),
        _feather((846, 58), (952, 150), -0.14, 15, rng, nb=34),
        _feather((34, 690), (86, 838), -0.12, 15, rng, nb=34),
    ]
    fmask = np.zeros((GH, GW), np.float32)
    for f in geo.feathers:
        fm = ra.poly(f.outline)
        fmask = np.maximum(fmask, fm)
        for b in f.barbs:
            ra.field_from_polyline(vx, vy, b['pl'], f.wmax * 0.16)
    put(target, fmask, C.rgb8(16, 14, 26))
    region[fmask > 0.5] = FEATHER
    masks['feather'] = fmask

    # =========================================================== hair behind
    locks = _locks(rng)
    geo.locks = locks
    back_poly = hd(catmull([(-1.80, -0.3), (-1.86, 0.6), (-1.58, 1.32), (-1.12, 1.70), (-0.5, 1.84), (0, 1.78),
                            (0.5, 1.84), (1.12, 1.70), (1.58, 1.32), (1.86, 0.6), (1.80, -0.3), (1.2, -1.2),
                            (0, -1.6), (-1.2, -1.2)], closed=True, n=10))
    back_m = ra.poly(back_poly)
    for L in locks:
        L['mask'] = ra.poly(ribbon(L['center'], L['width']))
    for L in locks:
        if L['layer'] == 'back':
            back_m = np.maximum(back_m, L['mask'])
    depth = sstep(-0.2, 1.5, Yl)
    back_col = C.mix_lab(P['hair_back'], P['hair_back_deep'], np.clip(0.25 + 0.75 * depth - 0.25 * np.abs(Xl) / 1.8, 0, 1)[..., None])
    put(target, back_m, back_col)
    region[back_m > 0.5] = HAIR_BACK
    crown = hd((0.08, -1.72))
    cdx = X - crown[0]
    cdy = Y - crown[1]
    cn = np.sqrt(cdx * cdx + cdy * cdy) + 1e-6
    sel = back_m > 0.5
    vx[sel] = (cdx / cn)[sel]
    vy[sel] = (cdy / cn)[sel]
    for L in locks:
        if L['layer'] == 'back':
            ra.field_from_polyline(vx, vy, L['center'], L['width'])
            edge = L['mask'] * (1 - ra.shift(L['mask'], -0.06 * U, -0.03 * U))
            toward(target, ra.blur(edge, 0.02 * U), P['hair_back_deep'], 0.7)
            lit = L['mask'] * (1 - ra.shift(L['mask'], 0.06 * U, 0.03 * U))
            toward(target, ra.blur(lit, 0.02 * U), P['hair_deep'], 0.55)
    masks['hair_back'] = back_m

    # =========================================================== neck and chest
    neck = np.vstack([hd([(-0.29, 0.95)]), bd([(-0.31, 1.50), (-0.36, 1.78), (-0.72, 1.92), (-1.06, 2.02),
                                              (-0.92, 2.97), (0.92, 2.97), (1.06, 2.02), (0.72, 1.92),
                                              (0.36, 1.78), (0.31, 1.50)]), hd([(0.29, 0.95)])])
    neck_m = ra.poly(neck)
    sk = C.mix_lab(P['skin_light'], P['skin_mid'], np.clip(0.35 + 0.5 * Xb, 0, 1)[..., None])
    put(target, neck_m, sk)
    region[neck_m > 0.5] = SKIN
    masks['neck'] = neck_m
    sel = neck_m > 0.5
    chest = np.clip((Yb - 1.75) / 0.3, 0, 1)
    vx[sel] = chest[sel] * 0.96 + 0.05
    vy[sel] = (1 - chest[sel]) + 0.12 * np.sign(Xb[sel]) * chest[sel]

    # =========================================================== dress
    torso = bd([(-1.04, 2.0), (-1.47, 2.08), (-1.52, 3.2), (-1.40, 5.02), (1.40, 5.02), (1.52, 3.2),
                (1.47, 2.08), (1.04, 2.0), (0.90, 2.95), (-0.90, 2.95)])
    puffL = catmull([(-1.42, 2.06), (-1.95, 2.08), (-2.50, 2.36), (-2.86, 2.92), (-2.90, 3.48), (-2.72, 3.86),
                     (-2.2, 3.98), (-1.70, 4.02), (-1.48, 3.5), (-1.40, 2.7)], closed=True, n=10)
    lowL = np.array([(-2.74, 3.84), (-2.86, 5.02), (-1.76, 5.02), (-1.70, 4.0)])
    dress_m = ra.polys([torso, bd(puffL), bd(mirror_x(puffL)), bd(lowL), bd(mirror_x(lowL))])
    geo.puff = [bd(puffL), bd(mirror_x(puffL))]
    geo.torso = torso
    seamL = bd((-1.50, 2.12))
    seamR = bd((1.50, 2.12))
    sleeve_m = ra.polys([bd(puffL), bd(mirror_x(puffL)), bd(lowL), bd(mirror_x(lowL))])
    masks['sleeve'] = sleeve_m
    # folds fan out from the shoulder seam over the puff, and fall straight on the bodice
    fold_sheen = np.zeros((GH, GW), np.float32)
    fold_film = np.zeros((GH, GW), np.float32)
    dvx = np.zeros((GH, GW), np.float32)
    dvy = np.ones((GH, GW), np.float32)
    n_f = value_noise(GW, GH, 60 * s, rng, octaves=2)
    for seam, sgn in ((seamL, -1.0), (seamR, 1.0)):
        ex = X - seam[0]
        ey = Y - seam[1]
        en = np.sqrt(ex * ex + ey * ey) + 1e-6
        phi = np.arctan2(ey, ex * sgn)
        side = (sleeve_m > 0.5) & (Xb * sgn > 0)
        bulge = 0.55 * np.sin(np.clip(en / (1.9 * U), 0, 1) * math.pi) * sgn
        ca, sa = np.cos(bulge), np.sin(bulge)
        fx = (ex / en) * ca - (ey / en) * sa
        fy = (ex / en) * sa + (ey / en) * ca
        dvx[side] = fx[side]
        dvy[side] = fy[side]
        fs = (0.5 + 0.5 * np.cos(phi * 15.0 + (n_f - 0.5) * 5.0)) ** 3
        fs *= sstep(0.15 * U, 0.7 * U, en)
        fold_sheen[side] = fs[side]
        ff = film_path(phi / math.pi * 1.6 + en / (5.5 * U) + 0.35 * n_f)
        fold_film[side] = ff[side]
    body = sleeve_m <= 0.5
    conv = 0.10 * Xb * (Yb - 2.0) / 2.7
    dvx[body] = -conv[body]
    dvy[body] = 1.0
    bs = (0.5 + 0.5 * np.cos(Xb * 9.5 + (n_f - 0.5) * 4.0 + 0.8 * np.sin(Yb * 2.0))) ** 3
    bust = np.exp(-(((np.abs(Xb) - 0.72) / 0.34) ** 2 + ((Yb - 3.55) / 0.42) ** 2))
    fold_sheen[body] = np.clip(0.55 * bs + 0.75 * bust, 0, 1)[body]
    fold_film[body] = film_path(0.5 + 0.22 * Xb + 0.16 * Yb + 0.4 * n_f)[body]
    dn = np.sqrt(dvx * dvx + dvy * dvy) + 1e-6
    sel = dress_m > 0.5
    vx[sel] = (dvx / dn)[sel]
    vy[sel] = (dvy / dn)[sel]
    sheen[sel] = fold_sheen[sel]
    film[sel] = fold_film[sel]
    tintd = lut[np.clip((fold_film / 1800.0 * 1023).astype(np.int32), 0, 1023)]
    put(target, dress_m, P['black'] * 0.9 + tintd * (0.010 + 0.05 * fold_sheen[..., None]))
    region[sel] = DRESS
    masks['dress'] = dress_m

    # lace: neckline, sleeve bands, placket
    geo.lace_bands = []

    def lace_band(pl, inward, wid=0.17, sc_r=0.058, sc_step=0.106):
        pl = resample(np.asarray(pl, np.float64), step=2.0)
        m = ra.stroke(pl, wid * U)
        tn = tangents(pl)
        nn = np.stack([-tn[:, 1], tn[:, 0]], 1)
        sgn = np.sign(np.sum(nn * (np.asarray(inward) - pl), 1))
        if np.ndim(inward) == 1 and len(inward) == 2 and abs(inward[0]) < 5 and abs(inward[1]) < 5:
            sgn = np.sign(nn @ np.asarray(inward, np.float64))      # a direction, not a point
        edge = pl + nn * (sgn * wid * 0.5 * U)[:, None]
        cen = resample(edge, step=sc_step * U)
        sc = [(c, sc_r * U) for c in cen]
        for c, r in sc:
            m = np.maximum(m, ra.ellipse(c[0], c[1], r, r))
        geo.lace_bands.append(dict(center=pl, edge=edge, scallops=sc, width=wid * U,
                                   out=nn * sgn[:, None]))
        return m, pl

    lace_m = np.zeros((GH, GW), np.float32)
    lvx = np.zeros((GH, GW), np.float32)
    lvy = np.zeros((GH, GW), np.float32)
    ncen = bd((0, 2.45))
    for seg in ([(-1.04, 2.0), (-0.90, 2.95)], [(-0.90, 2.95), (0.90, 2.95)], [(0.90, 2.95), (1.04, 2.0)]):
        m, pl = lace_band(bd(seg), ncen)
        lace_m = np.maximum(lace_m, m)
        ra.field_from_polyline(lvx, lvy, pl, 0.34 * U)
    for sgn in (-1, 1):
        seg = bd(catmull(np.array([(-2.76, 3.83), (-2.25, 3.99), (-1.70, 4.0)]) * [sgn, 1], n=10))
        m, pl = lace_band(seg, np.array([0.0, 1.0]))
        lace_m = np.maximum(lace_m, m * np.maximum(sleeve_m, ra.shift(sleeve_m, 0, -0.1 * U)))
        ra.field_from_polyline(lvx, lvy, pl, 0.34 * U)
    lace_shade = sstep(0.0, 1.0, value_noise(GW, GH, 14 * s, rng))
    put(target, lace_m, C.mix_lab(P['lace'], P['lace_shadow'], (0.18 + 0.4 * lace_shade)[..., None]))
    sel = lace_m > 0.5
    region[sel] = LACE
    vx[sel] = lvx[sel]
    vy[sel] = lvy[sel] + (np.abs(lvx[sel]) + np.abs(lvy[sel]) < 1e-3)
    # cast shadow of the neckline lace on the skin
    toward(target, ra.shift(ra.blur(lace_m, 0.02 * U), 0.03 * U, 0.06 * U) * neck_m * (1 - lace_m), P['skin_shadow'], 0.8)

    plk = bd([(-0.20, 3.0), (-0.20, 5.02), (0.20, 5.02), (0.20, 3.0)])
    plk_m = ra.poly(plk)
    geo.placket_edges = []
    for sgn in (-1, 1):
        edge = resample(bd([(0.20 * sgn, 3.05), (0.20 * sgn, 5.02)]), step=0.09 * U)
        sc = [(c, 0.047 * U) for c in edge]
        geo.placket_edges.append(dict(edge=edge, scallops=sc, sgn=sgn))
        for c, r in sc:
            plk_m = np.maximum(plk_m, ra.ellipse(c[0], c[1], r, r))
    put(target, plk_m, C.mix_lab(P['ivory'], P['lace_shadow'], np.clip(0.1 + 1.6 * np.abs(Xb), 0, 0.6)[..., None]))
    sel = plk_m > 0.5
    region[sel] = PLACKET
    vx[sel] = 0.0
    vy[sel] = 1.0
    masks['lace'] = np.maximum(lace_m, plk_m)
    geo.buttons = [bd((0, yb)) for yb in (3.66, 4.14, 4.60)]
    geo.button_r = 0.066 * U
    for c in geo.buttons:
        bm = ra.ellipse(c[0], c[1], geo.button_r, geo.button_r)
        put(target, bm, P['button'])
        region[bm > 0.5] = BUTTON

    # the wine-red bow
    loopL = catmull([(-0.06, 2.97), (-0.30, 2.80), (-0.60, 2.84), (-0.66, 3.02), (-0.56, 3.20), (-0.30, 3.18),
                     (-0.06, 3.09)], closed=True, n=10)
    tailL = np.array([(-0.07, 3.08), (-0.36, 3.72), (-0.22, 3.66), (-0.12, 3.78), (0.0, 3.12)])
    tailR = np.array([(0.07, 3.08), (0.34, 3.70), (0.21, 3.65), (0.10, 3.76), (0.0, 3.12)])
    knot_c = bd((0, 3.03))
    geo.bow = SimpleNamespace(loops=[bd(loopL), bd(mirror_x(loopL))], tails=[bd(tailL), bd(tailR)], knot=knot_c,
                              knot_r=(0.095 * U, 0.115 * U))
    tails_m = ra.polys(geo.bow.tails)
    loops_m = ra.polys(geo.bow.loops)
    knot_m = ra.ellipse(knot_c[0], knot_c[1], *geo.bow.knot_r)
    bow_m = np.maximum(np.maximum(tails_m, loops_m), knot_m)
    kdx = X - knot_c[0]
    kdy = Y - knot_c[1]
    kd = np.sqrt(kdx * kdx + kdy * kdy) + 1e-6
    bcol = C.mix_lab(P['bow_dark'], P['bow_mid'], sstep(0.05 * U, 0.33 * U, kd)[..., None])
    hl = np.exp(-(((np.abs(Xb) - 0.40) / 0.16) ** 2 + ((Yb - 2.93) / 0.07) ** 2))
    bcol = bcol + (P['bow_light'] - bcol) * (0.8 * hl)[..., None]
    put(target, tails_m, C.mix_lab(P['bow_dark'], P['bow_mid'], (0.25 + 0.5 * sstep(3.1, 3.7, Yb))[..., None]))
    put(target, loops_m, bcol)
    put(target, knot_m, C.mix_lab(P['bow_mid'], P['bow_light'], 0.25))
    sel = bow_m > 0.5
    region[sel] = BOW
    vx[sel] = (kdx / kd)[sel]
    vy[sel] = (kdy / kd)[sel]
    masks['bow'] = bow_m

    # =========================================================== face
    half = [(0.0, -1.60), (0.62, -1.42), (0.98, -0.95), (1.04, -0.40), (1.00, 0.08), (0.90, 0.45),
            (0.66, 0.80), (0.34, 1.07), (0.12, 1.205), (0.0, 1.235)]
    fpts = np.array(half + [(-x, y) for x, y in half[-2:0:-1]])
    face_poly = hd(catmull(fpts, closed=True, n=12))
    geo.face = face_poly
    geo.jaw = hd(catmull([(-1.02, -0.1), (-0.98, 0.25), (-0.90, 0.45), (-0.66, 0.80), (-0.34, 1.07), (-0.12, 1.205),
                          (0.0, 1.235), (0.12, 1.205), (0.34, 1.07), (0.66, 0.80), (0.90, 0.45), (0.98, 0.25),
                          (1.02, -0.1)], n=12))
    face_m = ra.poly(face_poly)
    masks['face'] = face_m
    # the hair in front, needed now for the shadow it throws on the face
    cap = hd(catmull([(-1.70, 0.20), (-1.74, -0.45), (-1.52, -1.10), (-1.05, -1.58), (-0.45, -1.80), (0.10, -1.86),
                      (0.65, -1.76), (1.15, -1.50), (1.55, -1.02), (1.74, -0.42), (1.72, 0.20), (1.25, 0.05),
                      (1.10, -0.48), (0.62, -0.62), (0.0, -0.64), (-0.62, -0.62), (-1.10, -0.48), (-1.25, 0.05)],
                     closed=True, n=10))
    geo.cap = cap
    cap_m = ra.poly(cap)
    front_m = cap_m.copy()
    for L in locks:
        if L['layer'] != 'back':
            front_m = np.maximum(front_m, L['mask'])
    masks['hair_front'] = front_m

    g = np.clip(0.42 + 0.34 * Xl + 0.16 * Yl, 0, 1)
    skin = C.mix_lab(P['skin_light'], P['skin_mid'], g[..., None])
    put(target, face_m, skin)
    region[face_m > 0.5] = SKIN
    form = ra.blur(face_m * (1 - ra.shift(face_m, -0.13 * U, -0.03 * U)), 0.035 * U)
    toward(target, form * face_m, P['skin_shadow'], 0.55)
    for sx in (-1, 1):
        c = hd((0.63 * sx, 0.44))
        bl = np.exp(-(((Xl - 0.63 * sx) / 0.30) ** 2 + ((Yl - 0.44) / 0.15) ** 2))
        toward(target, bl * face_m, P['blush'], 0.42)
    # chin shadow on the neck (hard-edged, the anime way) and hair shadow on the brow
    chin_sh = ra.shift(face_m, 0.05 * U, 0.26 * U) * neck_m * (1 - face_m)
    toward(target, ra.blur(chin_sh, 0.008 * U), P['skin_deep'], 0.82)
    side_sh = ra.blur(ra.shift(back_m * 0 + front_m, 0.10 * U, 0.20 * U), 0.02 * U) * neck_m * (1 - face_m)
    toward(target, side_sh, P['skin_shadow'], 0.6)
    bang_sh = ra.blur(ra.shift(front_m, 0.025 * U, 0.105 * U), 0.012 * U) * face_m
    toward(target, bang_sh, P['skin_shadow'], 0.80)
    fcen = hd((0.0, 0.25))
    fdx = X - fcen[0]
    fdy = Y - fcen[1]
    fd = np.sqrt(fdx * fdx + fdy * fdy) + 1e-6
    sel = face_m > 0.5
    fa = 2.05 + 0.9 * (n_lo - 0.5) + 0.25 * Xl
    vx[sel] = np.cos(fa)[sel]
    vy[sel] = np.sin(fa)[sel]

    # ---- eyes
    geo.eyes = [_eye(-1), _eye(1)]
    eye_all = np.zeros((GH, GW), np.float32)
    for ey in geo.eyes:
        op = ra.poly(ey.opening)
        lash = ra.poly(ey.lash)
        put(target, op, P['sclera'])
        lid_sh = ra.blur(ra.shift(lash, 0.0, 0.085 * U), 0.03 * U) * op
        toward(target, lid_sh, P['sclera_shadow'], 0.95)
        iris = ra.poly(ellipse_pts(ey.iris_c[0], ey.iris_c[1], ey.iris_rx, ey.iris_ry, TILT, n=64)) * op
        ix = (Xl - (0.515 * ey.sx)) / 0.186
        iy = (Yl - 0.040) / 0.226
        v = np.clip(iy * 0.5 + 0.5, 0, 1)
        icol = C.mix_lab(P['iris_top'], P['iris_mid'], sstep(0.18, 0.62, v)[..., None])
        icol = icol + (P['iris_bot'] - icol) * sstep(0.55, 1.0, v)[..., None]
        # opal: cool on one side of the lower iris, warm on the other
        icol = icol + (C.rgb8(112, 204, 214) - icol) * (0.38 * sstep(0.3, 1.0, v) * sstep(0.1, -0.9, ix))[..., None]
        icol = icol + (C.rgb8(202, 150, 226) - icol) * (0.38 * sstep(0.3, 1.0, v) * sstep(-0.1, 0.9, ix))[..., None]
        rad = np.sqrt(ix * ix + iy * iy)
        icol = icol + (P['iris_ring'] - icol) * (0.75 * sstep(0.80, 0.98, rad))[..., None]
        put(target, iris, icol)
        pup = ra.poly(ellipse_pts(ey.pupil_c[0], ey.pupil_c[1], ey.pupil_rx, ey.pupil_ry, TILT, n=40)) * op
        put(target, pup, P['pupil'])
        toward(target, lid_sh * iris, P['pupil'], 0.55)
        lcol = C.mix_lab(P['lash'], P['lash_tip'], sstep(0.55, 0.95, np.abs(Xl))[..., None])
        put(target, lash, lcol)
        h1 = ra.poly(ellipse_pts(ey.hl1[0], ey.hl1[1], ey.hl1_r[0], ey.hl1_r[1], TILT - 0.5, n=32))
        h2 = ra.ellipse(ey.hl2[0], ey.hl2[1], ey.hl2_r, ey.hl2_r)
        put(target, np.maximum(h1, h2), P['white'])
        ey.mask = np.maximum(op, lash)
        eye_all = np.maximum(eye_all, ey.mask)
        toward(target, ra.stroke(ey.crease, 0.016 * U), P['line'], 0.75)
        toward(target, ra.stroke(ey.lower_line, 0.014 * U), P['lash'], 0.8)
        ecx, ecy = ey.iris_c
        edx = X - ecx
        edy = Y - ecy
        en = np.sqrt(edx * edx + edy * edy) + 1e-6
        sel = ey.mask > 0.5
        vx[sel] = (-edy / en)[sel]
        vy[sel] = (edx / en)[sel]
    region[eye_all > 0.5] = EYE
    masks['eyes'] = eye_all

    for ey in geo.eyes:
        toward(target, ra.stroke(ey.brow, 0.026 * U) * face_m, P['brow'], 0.9)

    # ---- nose, mouth
    geo.nose = hd([(0.040, 0.600), (0.052, 0.640), (0.036, 0.662)])
    geo.mouth = hd(catmull([(-0.175, 0.858), (-0.09, 0.888), (0.0, 0.897), (0.09, 0.888), (0.175, 0.858)], n=10))
    geo.lip_low = hd(catmull([(-0.07, 0.985), (0.0, 0.997), (0.07, 0.985)], n=8))
    toward(target, ra.stroke(geo.nose, 0.02 * U), P['skin_deep'], 0.8)
    toward(target, ra.stroke(geo.mouth, 0.022 * U), P['line'], 0.9)
    lipm = np.exp(-((Xl / 0.12) ** 2 + ((Yl - 0.94) / 0.035) ** 2)) * face_m
    toward(target, lipm, P['lip'], 0.45)

    # =========================================================== hair in front
    put(target, cap_m, C.mix_lab(P['hair_light'], P['hair_mid'], np.clip(0.15 + 0.35 * sstep(-1.9, -0.4, Yl), 0, 1)[..., None]))
    sel = cap_m > 0.5
    vx[sel] = (cdx / cn)[sel]
    vy[sel] = (cdy / cn)[sel]
    d = 0.075 * U
    for L in locks:
        if L['layer'] == 'back':
            continue
        m = L['mask']
        cast = ra.blur(ra.shift(m, 0.05 * U, 0.085 * U), 0.012 * U) * front_m * (1 - m)
        toward(target, cast, P['hair_shadow'], 0.75)
        base = C.mix_lab(P['hair_light'], P['hair_mid'], L['tone'])
        put(target, m, base)
        shade = ra.blur(m * (1 - ra.shift(m, -d, -d * 0.45)), 0.018 * U)
        toward(target, shade * m, P['hair_shadow'], 0.85)
        lit = ra.blur(m * (1 - ra.shift(m, d, d * 0.45)), 0.018 * U)
        toward(target, lit * m, P['hair_light'], 0.9)
        ra.field_from_polyline(vx, vy, L['center'], L['width'])
    hair_m = front_m
    # tips cool off; shadows pick up the colours of the room
    tip = sstep(-0.1, 1.5, Yl) * hair_m
    toward(target, tip, P['hair_shadow'], 0.42)
    toward(target, sstep(0.1, 1.9, Xl) * hair_m, P['hair_mid'], 0.45)
    occl = ra.blur(face_m, 0.13 * U) * hair_m * sstep(-0.5, -0.1, Yl)
    toward(target, occl, P['hair_shadow'], 0.55)
    lum = target @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    shadowness = sstep(0.80, 0.45, lum) * hair_m
    n_c = value_noise(GW, GH, 90 * s, rng, octaves=2)
    toward(target, shadowness * sstep(0.50, 0.75, n_c), P['hair_cyan'] * 0.82, 0.55)
    toward(target, shadowness * sstep(0.50, 0.25, n_c), P['hair_pink'] * 0.84, 0.55)
    hc = hd((0.0, -0.62))
    rho = np.sqrt((X - hc[0]) ** 2 + (Y - hc[1]) ** 2) / U
    a_h = np.arctan2(Y - hc[1], X - hc[0])
    band = np.exp(-((rho - (0.97 + 0.03 * np.sin(a_h * 13.0) + 0.06 * (n_c - 0.5))) / 0.085) ** 2)
    band *= sstep(-0.35, -0.8, Yl) * hair_m
    toward(target, band, P['white'], 0.6)
    part = np.exp(-(((Xl - 0.08) / 0.10) ** 2)) * sstep(-1.35, -1.75, Yl) * hair_m
    toward(target, part, P['hair_shadow'], 0.5)
    region[hair_m > 0.5] = HAIR
    masks['hair'] = hair_m

    # brows show through the fringe, as they do in the character art
    for ey in geo.eyes:
        toward(target, ra.stroke(ey.brow, 0.022 * U) * hair_m, P['brow'], 0.30)

    # =========================================================== crescent hairpin
    pc = hd((1.13, -0.83))
    pr = 0.215 * U
    pic = hd((1.13 + 0.088, -0.83 - 0.082))
    pir = 0.172 * U
    pin_m = ra.ellipse(pc[0], pc[1], pr, pr) * (1 - ra.ellipse(pic[0], pic[1], pir, pir))
    geo.pin = SimpleNamespace(c=pc, r=pr, ic=pic, ir=pir)
    toward(target, ra.blur(ra.shift(pin_m, 0.03 * U, 0.05 * U), 0.015 * U) * hair_m, P['hair_shadow'], 0.8)
    put(target, pin_m, P['pin'])
    sel = pin_m > 0.5
    region[sel] = PIN
    pdx = X - pc[0]
    pdy = Y - pc[1]
    pn = np.sqrt(pdx * pdx + pdy * pdy) + 1e-6
    vx[sel] = (-pdy / pn)[sel]
    vy[sel] = (pdx / pn)[sel]
    masks['pin'] = pin_m

    # =========================================================== how fine to work, per place
    rmin = np.full((GH, GW), 4.5, np.float32)
    for reg, r in ((HAIR_BACK, 2.6), (SKIN, 1.5), (HAIR, 1.7), (DRESS, 3.2), (LACE, 1.1), (BOW, 1.4),
                   (EYE, 0.75), (PIN, 1.0), (PLACKET, 1.3), (BUTTON, 0.9), (FEATHER, 1.6)):
        rmin[region == reg] = r
    face_zone = ra.blur(face_m, 0.05 * U) > 0.4
    rmin[face_zone & (region == SKIN)] = 1.15

    figure = (region != BG) & (region != FEATHER)
    masks['figure'] = figure.astype(np.float32)
    return SimpleNamespace(GW=GW, GH=GH, s=s, ra=ra, target=np.clip(target, 0, 1).astype(np.float32),
                           region=region, vx=vx, vy=vy, rmin=rmin, sheen=sheen, film=film, geo=geo,
                           masks=masks, Xl=Xl, Yl=Yl, Xb=Xb, Yb=Yb, rr=rr, th=th)
