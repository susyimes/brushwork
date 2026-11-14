"""Colour science for the painter: sRGB/linear/OKLab, Kubelka-Munk pigments, recipe lookup.

All paint colour inside the simulation is linear-light sRGB reflectance.  Paint is never
blended as RGB: pigments are mixed in Kubelka-Munk (K = absorption, S = scattering) space,
so alizarin + phthalo green goes to black instead of to grey, the way it does on a palette.

There is deliberately no black pigment on this palette.
"""
import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- colour spaces
def srgb_to_lin(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((np.maximum(c, 0) + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, np.float64), 0.0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def rgb8(r, g=None, b=None):
    """8-bit sRGB triple (or '#rrggbb') -> linear float32 triple."""
    if isinstance(r, str):
        h = r.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return srgb_to_lin(np.array([r, g, b], np.float64) / 255.0).astype(np.float32)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i = np.linalg.inv(_M1)
_M2i = np.linalg.inv(_M2)


def lin_to_oklab(c):
    c = np.asarray(c, np.float64)
    lms = np.cbrt(np.clip(c @ _M1.T, 0, None))
    return (lms @ _M2.T).astype(np.float32)


def oklab_to_lin(lab):
    lab = np.asarray(lab, np.float64)
    lms = (lab @ _M2i.T) ** 3
    return (lms @ _M1i.T).astype(np.float32)


def mix_lab(c0, c1, t):
    """Perceptual interpolation between two linear colours (t scalar or array[..., None])."""
    a = lin_to_oklab(c0)
    b = lin_to_oklab(c1)
    return oklab_to_lin(a + (b - a) * t)


# ----------------------------------------------------------------------------- pigments
# name, masstone (8-bit sRGB, thick paint straight from the tube), scattering S
# Low S = transparent, strong-tinting pigment.  No black.
PIGMENT_DEFS = [
    ('white',         (250, 249, 244), 1.00),
    ('cad_yellow',    (250, 205,  20), 0.75),
    ('ochre',         (190, 140,  60), 0.60),
    ('cad_red',       (225,  50,  30), 0.65),
    ('alizarin',      (110,  12,  35), 0.10),
    ('magenta',       (150,  15,  95), 0.10),
    ('violet',        ( 38,  12,  62), 0.07),
    ('ultramarine',   ( 25,  30, 130), 0.14),
    ('phthalo_blue',  (  8,  28,  85), 0.07),
    ('teal',          ( 10, 120, 130), 0.30),
    ('phthalo_green', (  4,  56,  42), 0.08),
    ('umber',         ( 55,  35,  25), 0.25),
    ('sienna',        (130,  60,  30), 0.30),
]
PIG = {name: i for i, (name, _, _) in enumerate(PIGMENT_DEFS)}
NPIG = len(PIGMENT_DEFS)


def _ks_of_reflectance(R):
    R = np.clip(R, 1e-4, 0.9999)
    return (1.0 - R) ** 2 / (2.0 * R)


def reflectance_of_ks(x):
    return 1.0 + x - np.sqrt(x * x + 2.0 * x)


PIG_S = np.array([s for _, _, s in PIGMENT_DEFS], np.float32)
PIG_R = np.stack([rgb8(*c) for _, c, _ in PIGMENT_DEFS]).astype(np.float32)
PIG_K = (_ks_of_reflectance(PIG_R.astype(np.float64)) * PIG_S[:, None]).astype(np.float32)


def mix_reflectance(idx, w):
    """KM mix. idx (..., C) int pigment ids, w (..., C) weights -> linear reflectance (..., 3)."""
    idx = np.asarray(idx)
    w = np.asarray(w, np.float64)
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-12)
    K = (PIG_K[idx].astype(np.float64) * w[..., None]).sum(-2)
    S = (PIG_S[idx].astype(np.float64) * w).sum(-1)
    return reflectance_of_ks(K / S[..., None]).astype(np.float32)


class RecipeBook:
    """Every colour the painter asks for is answered with a recipe of real pigments.

    Table = single pigments and strength-balanced pairs, each let down with white over a
    log ladder.  Lookup is nearest-neighbour in OKLab, choosing at random among near ties so
    the same colour is not always mixed from the same tubes.
    """

    def __init__(self):
        chroma = [i for i in range(NPIG) if i != PIG['white']]
        strength = PIG_K.mean(1) + PIG_S
        bases = []  # (ia, ib, wa, wb)
        for a in chroma:
            bases.append((a, a, 1.0, 0.0))
        ratios = [1 / 8, 1 / 4, 3 / 8, 1 / 2, 5 / 8, 3 / 4, 7 / 8]
        for i, a in enumerate(chroma):
            for b in chroma[i + 1:]:
                for r in ratios:
                    wa = r / strength[a]
                    wb = (1 - r) / strength[b]
                    s = wa + wb
                    bases.append((a, b, wa / s, wb / s))
        frac = [1, .7, .5, .35, .25, .17, .12, .08, .055, .037, .025, .017, .011, .007,
                .0045, .003, .002, .0012, .0007, .0004, .0002]
        idx, w = [], []
        W = PIG['white']
        for (a, b, wa, wb) in bases:
            for p in frac:
                idx.append((a, b, W, W))
                w.append((wa * p, wb * p, 1 - p, 0.0))
        idx.append((W, W, W, W))
        w.append((1.0, 0, 0, 0))
        self.idx = np.array(idx, np.int32)
        self.w = np.array(w, np.float32)
        self.rgb = mix_reflectance(self.idx, self.w)
        self.lab = lin_to_oklab(self.rgb)
        self.tree = cKDTree(self.lab)

    def lookup(self, rgb_lin, rng=None, k=4, tol=0.012):
        """rgb_lin (N,3) linear -> (idx (N,4) int32, w (N,4) float32, achieved rgb (N,3))."""
        lab = lin_to_oklab(np.atleast_2d(rgb_lin))
        d, ii = self.tree.query(lab, k=k)
        if rng is None or k == 1:
            pick = ii[:, 0] if k > 1 else ii
        else:
            ok = d <= (d[:, :1] + tol)
            r = rng.random(d.shape) * ok
            pick = ii[np.arange(len(ii)), np.argmax(r, 1)]
        return self.idx[pick].copy(), self.w[pick].copy(), self.rgb[pick]


def recipe(**parts):
    """recipe(alizarin=.5, phthalo_green=.5) -> (idx[4], w[4]) padded."""
    items = list(parts.items())[:4]
    idx = [PIG[k] for k, _ in items] + [0] * (4 - len(items))
    w = [float(v) for _, v in items] + [0.0] * (4 - len(items))
    s = sum(w)
    return np.array(idx, np.int32), np.array([x / s for x in w], np.float32)


# Chromatic blacks: complementary dark pigments, the classic way a painter makes black
# without a black tube.  Each is "black" as a mix and deeply coloured in every streak.
CHROMATIC_BLACKS = {
    'blue':   dict(ultramarine=.40, phthalo_blue=.32, umber=.28),
    'violet': dict(violet=.52, phthalo_green=.22, alizarin=.26),
    'green':  dict(phthalo_green=.50, alizarin=.28, phthalo_blue=.22),
    'red':    dict(alizarin=.50, phthalo_green=.24, violet=.26),
    'teal':   dict(phthalo_green=.36, phthalo_blue=.40, umber=.24),
    'warm':   dict(umber=.42, ultramarine=.30, alizarin=.28),
    'indigo': dict(phthalo_blue=.46, violet=.36, umber=.18),
    'wine':   dict(alizarin=.36, violet=.34, phthalo_blue=.30),
}


# ----------------------------------------------------------------------------- thin film
def thinfilm_lut(n=1024, opd_max=1800.0):
    """Interference colour of a thin film vs optical path difference (nm), linear sRGB.

    Two-beam interference with one half-wave phase jump: R(lambda) = sin^2(pi*OPD/lambda),
    integrated against CIE 1931 matching functions (Wyman-Sloan-Shirley analytic fit) under
    an equal-energy illuminant.  This is the Newton's-rings / oil-slick / interference-
    pigment colour sequence.  Returns (n,3) float32, peak-normalised.
    """
    lam = np.arange(380.0, 781.0, 2.0)

    def g(x, mu, s1, s2):
        s = np.where(x < mu, s1, s2)
        return np.exp(-0.5 * ((x - mu) / s) ** 2)

    xb = 1.056 * g(lam, 599.8, 37.9, 31.0) + 0.362 * g(lam, 442.0, 16.0, 26.7) - 0.065 * g(lam, 501.1, 20.4, 26.2)
    yb = 0.821 * g(lam, 568.8, 46.9, 40.5) + 0.286 * g(lam, 530.9, 16.3, 31.1)
    zb = 1.217 * g(lam, 437.0, 11.8, 36.0) + 0.681 * g(lam, 459.0, 26.0, 13.8)
    opd = np.linspace(0, opd_max, n)[:, None]
    refl = np.sin(np.pi * opd / lam[None, :]) ** 2
    X = (refl * xb).sum(1) / yb.sum()
    Y = (refl * yb).sum(1) / yb.sum()
    Z = (refl * zb).sum(1) / yb.sum()
    # equal-energy white -> D65-ish sRGB primaries (Bradford-free; fine for a sheen LUT)
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    rgb = np.stack([X, Y, Z], 1) @ M.T
    rgb = np.clip(rgb, 0, None)
    rgb /= rgb.max()
    return rgb.astype(np.float32)


# ----------------------------------------------------------------------------- conveniences
def as_lin(c):
    """'#rrggbb' or an (r, g, b) list/tuple of 0-255 sRGB values -> linear float32 triple.
    A float ndarray is taken to be linear already and passed through."""
    if isinstance(c, np.ndarray) and c.dtype.kind == 'f':
        return c.astype(np.float32)
    if isinstance(c, str):
        return rgb8(c)
    c = list(c)
    return rgb8(c[0], c[1], c[2])


# Interference colours by name -> optical path difference of the film, nm (see thinfilm_lut).
SHEEN_HUES = dict(violet=550, blue=600, azure=650, cyan=700, aqua=750, lime=800, yellow=850, gold=900,
                  orange=950, pink=1000, magenta=1050, purple=1100, indigo=1150, teal=1225)


def film_for(hue):
    """Name from SHEEN_HUES (or a number of nm) -> film optical thickness in nm."""
    if isinstance(hue, (int, float)):
        return float(hue)
    return float(SHEEN_HUES[hue])
