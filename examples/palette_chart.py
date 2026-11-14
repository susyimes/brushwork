"""Colour chart: every tube straight from the tube and let down with white, painted with the brush.

    python examples/palette_chart.py [--width 2000]     ->  out/palette_chart.jpg

Each row is one pigment: on the left as it comes, then with more and more white.  The names
are written on afterwards; everything else is paint.  Tubes below the line in the right-hand
column are not part of the default palette (see PALETTES in brushwork/color.py).
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from brushwork import PALETTES, PIGMENT_DEFS, Studio

SHARE = (1.0, 0.35, 0.12, 0.04, 0.012)          # how much of the mix is the tube colour, the rest white
ROWS, PITCH, TOP = 14, 52.0, 44.0
LABEL, STEP, WIDE = 158.0, 64.0, 56.0


def cell(k):
    """Top-left corner, in units, of the row for tube number k."""
    return 22.0 + 500.0 * (k // ROWS), TOP + PITCH * (k % ROWS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--width', type=int, default=2000)
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'palette_chart.jpg'))
    a = ap.parse_args()
    st = Studio(a.width, int(a.width * 0.78), seed=4, primer='#77736e', log=None)
    items = []
    for k, (name, _, _) in enumerate(PIGMENT_DEFS):
        x0, y0 = cell(k)
        for i, share in enumerate(SHARE if name != 'white' else SHARE[:1]):
            x = x0 + LABEL + i * STEP
            mix = {name: 1.0} if share == 1.0 else {name: share, 'white': 1.0 - share}
            for dy in (7.0, 18.0, 29.0):         # three passes of a flat brush make one swatch
                items.append(dict(path=[[x, y0 + dy], [x + WIDE, y0 + dy + 0.6]], brush=6.6, pigments=mix,
                                  style='body', thick=1.7, unmix=0.0, taper=0.15, load=3.0))
    st.strokes(items)
    img = np.ascontiguousarray(st.cv.render()[:, :, ::-1])
    S = st.S
    full = set(PALETTES['full'])
    for k, (name, rgb, _) in enumerate(PIGMENT_DEFS):
        x0, y0 = cell(k)
        ink = (238, 236, 232) if name in full else (150, 200, 240)
        cv2.putText(img, name, (int(x0 * S), int((y0 + 23) * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.50 * S, ink,
                    max(1, int(round(1.1 * S))), cv2.LINE_AA)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 93])
    with open(a.out, 'wb') as f:
        f.write(buf.tobytes())
    print('%d tubes, %d brushstrokes -> %s' % (len(PIGMENT_DEFS), st.cv.n_strokes, a.out))


if __name__ == '__main__':
    main()
