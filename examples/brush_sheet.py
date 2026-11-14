"""Brush sampler: a sheet of test strokes, to look at the brush model on its own.

    python examples/brush_sheet.py [scale]       ->  out/brush_sheet.jpg

Left half, on a toned ground: loaded strokes of several sizes; the same brush running dry;
wet-into-wet (yellow dragged across wet blue goes green in Kubelka-Munk space, and smears
more as the pickup rate rises).  Right half, on black: the chromatic blacks with rising
"unmixedness" on the brush, bands of interference medium glazed across them, strokes that
carry the medium themselves, and thick dabs.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from brushwork import color as C
from brushwork.engine import Canvas, Strokes, make_params


def curve(x0, y0, x1, y1, bend=0.0, n=40, wob=0.0, rng=None):
    t = np.linspace(0, 1, n)
    x = x0 + (x1 - x0) * t
    y = y0 + (y1 - y0) * t
    dx, dy = x1 - x0, y1 - y0
    L = np.hypot(dx, dy)
    nx, ny = -dy / L, dx / L
    off = bend * L * np.sin(np.pi * t)
    if wob and rng is not None:
        off = off + wob * np.sin(2 * np.pi * (t * rng.uniform(1, 2.5) + rng.random()))
    return np.stack([x + nx * off, y + ny * off], 1).astype(np.float32)


def one(path, rec, **kw):
    s = Strokes()
    idx, w = rec
    s.add([path], None, make_params(1, **kw), idx[None], w[None])
    return s


def main(scale=1.0):
    W, H = int(1500 * scale), int(1100 * scale)
    k = scale
    rng = np.random.default_rng(3)
    cv = Canvas(W, H, ground=C.rgb8(120, 96, 92), pitch=max(3.0, 9.0 * k), seed=5)
    cv.col[:, W // 2:] = C.rgb8(10, 9, 14)
    book = C.RecipeBook()

    def rec_of(r, g, b):
        i, w, _ = book.lookup(C.rgb8(r, g, b)[None])
        return i[0], w[0]

    t0 = time.time()
    y = 90 * k
    # 1. loaded brushes, light over a mid ground
    for n, (rgb, R) in enumerate([((246, 240, 228), 34), ((235, 150, 165), 26), ((120, 190, 215), 20),
                                  ((250, 210, 90), 14), ((200, 60, 70), 9), ((240, 240, 250), 5)]):
        p = curve(60 * k, y, 690 * k, y + 20 * k, bend=0.05 * (-1) ** n, n=50)
        cv.paint(one(p, rec_of(*rgb), radius=R * k, thick=1.6, loadlen=900 * k, unmix=0.5))
        y += (R * 2 + 26) * k
    # 2. the same brush running dry
    for n, lam in enumerate([900, 320, 150, 70]):
        p = curve(60 * k, y, 690 * k, y - 10 * k, bend=0.03, n=50)
        cv.paint(one(p, rec_of(244, 236, 220), radius=18 * k, thick=1.3, loadlen=lam * k, dry=0.6))
        y += 52 * k
    # 3. wet into wet: crossing strokes smear each other
    yb = y + 10 * k
    cv.paint(one(curve(80 * k, yb + 60 * k, 680 * k, yb + 60 * k, 0.02), rec_of(40, 70, 170),
                 radius=40 * k, thick=1.6, loadlen=2000 * k))
    for n, x in enumerate(np.linspace(150, 620, 6)):
        cv.paint(one(curve(x * k, yb - 10 * k, (x + 30) * k, yb + 150 * k, 0.06), rec_of(250, 235, 120),
                     radius=13 * k, thick=1.4, loadlen=500 * k, pickup=[0.0, 0.01, 0.03, 0.06, 0.12, 0.25][n]))

    # ---- right half: on black ----------------------------------------------------------
    y = 90 * k
    names = list(C.CHROMATIC_BLACKS)
    for n, name in enumerate(names):
        p = curve(800 * k, y, 1440 * k, y + 14 * k, bend=0.05 * (-1) ** n, n=50)
        cv.paint(one(p, C.recipe(**C.CHROMATIC_BLACKS[name]), radius=24 * k, thick=1.7,
                     loadlen=1500 * k, unmix=[0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 1.2, 1.2][n]))
        y += 62 * k
    cv.dry(1.0)
    cv.update_tooth()
    # interference medium glazed over the dry blacks: almost no colour of its own, all sheen
    for n, film in enumerate([580, 640, 700, 760, 1080, 1140, 1200, 1250]):
        x = (830 + n * 76) * k
        p = curve(x, 60 * k, x + 20 * k, 560 * k, bend=0.03, n=50)
        cv.paint(one(p, C.recipe(violet=1), radius=22 * k, opacity=0.10, thick=0.9, iri=0.9, film=film,
                     filmvar=30, pickup=0, dry=0.2, furrow=0.7, loadlen=420 * k))
    y = 640 * k
    for n in range(5):
        p = curve(800 * k, y, 1440 * k, y, bend=0.08, n=50, wob=6 * k, rng=rng)
        cv.paint(one(p, C.recipe(**C.CHROMATIC_BLACKS[names[n]]), radius=(30 - n * 5) * k, thick=1.8,
                     loadlen=700 * k, unmix=1.0, iri=0.7, film=1250 - 45 * n, filmvar=40))
        y += 70 * k
    # dabs
    for n in range(40):
        x, yy = rng.uniform(820, 1420) * k, rng.uniform(980, 1060) * k
        a = rng.uniform(0, 6.28)
        ln = rng.uniform(4, 16) * k
        p = curve(x, yy, x + np.cos(a) * ln, yy + np.sin(a) * ln, n=6)
        cv.paint(one(p, rec_of(250, 250, 240), radius=rng.uniform(2, 5) * k, thick=2.4, blob=1.0))
    print('paint %.2fs  strokes=%d trails=%d pix=%.1fM' % (time.time() - t0, cv.n_strokes, cv.n_trails, cv.n_pix / 1e6))
    out = os.path.join(ROOT, 'out')
    os.makedirs(out, exist_ok=True)
    ok, buf = cv2.imencode('.jpg', cv.render()[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 93])
    with open(os.path.join(out, 'brush_sheet.jpg'), 'wb') as f:
        f.write(buf.tobytes())


if __name__ == '__main__':
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
