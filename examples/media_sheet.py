"""The same little still life painted in every medium, side by side.

    python examples/media_sheet.py [--width 520] [--only gouache watercolor]   ->  out/media_sheet.jpg

Every cell is the same underdrawing and the same calls (paint(), then two accents); only the
medium changes.  Use it to choose a medium, or to check one you have written yourself
(--extra path/to/my_medium.json).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2
import numpy as np

from brushwork import Studio
from brushwork import media as MEDIA


def still_life(sk):
    sk.background({'linear': [[0, 0], [1000, 480]], 'colors': ['#b8b9a8', '#6f7568']}, region='wall',
                  direction={'angle': 70, 'spread': 35}, detail=5.5)
    sk.rect(0, 470, 1000, 330, color='#a3815c', region='table', direction=3, detail=4)
    sk.ellipse((640, 470), 120, 150, color='#3d6fa3', region='jug', detail=1.8,
               direction={'contour': True}, vary=0.3)
    sk.form_shadow('jug', '#1f3657', light=(-1, -0.6), width=90, amount=0.85)
    sk.form_light('jug', '#9cc3e6', light=(-1, -0.6), width=40, amount=0.7)
    sk.ellipse((400, 560), 110, 80, rot=-10, color='#e6c022', region='lemon', detail=1.6,
               direction={'angle': 15, 'spread': 30, 'scale': 60})
    sk.form_shadow('lemon', '#a2780f', light=(-1, -0.6), width=60, amount=0.85)
    sk.form_light('lemon', '#fbe670', light=(-1, -0.6), width=50, amount=0.8)
    sk.cast_shadow('lemon', 'table', '#35261c', offset=(44, 14), blur=5)
    sk.cast_shadow('jug', ['table', 'wall'], '#2a2119', offset=(60, 16), blur=8)


def accents(st):
    st.strokes([dict(path=[[355, 528], [375, 522]], brush=7, color='#fffce6', style='dab', thick=2.6),
                dict(path=[[575, 390], [565, 430], [568, 470]], brush=5, color='#e8f2fb', style='dab', smooth=True),
                dict(path=[[300, 640], [520, 655]], brush=1.4, color='#3a2a1c', style='line')])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--width', type=int, default=520)
    ap.add_argument('--only', nargs='*', default=None, help='media to paint (default: all)')
    ap.add_argument('--extra', nargs='*', default=[], help='your own medium JSON files to add')
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'media_sheet.jpg'))
    a = ap.parse_args()
    which = (a.only or MEDIA.names()) + a.extra
    cells = []
    for m in which:
        t0 = time.time()
        st = Studio(a.width, a.width * 4 // 5, seed=3, medium=m, log=None)
        still_life(st.sk)
        st.paint()
        accents(st)
        img = np.ascontiguousarray(st.render()[:, :, ::-1])
        label = st.medium.title
        band = np.full((34, img.shape[1], 3), 250, np.uint8)
        cv2.putText(band, label.split(' ')[-1], (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        cells.append(np.concatenate([band, img], 0))
        print('%-12s %6d strokes  %5.1f s' % (st.medium.name, st.cv.n_strokes, time.time() - t0))
    cols = min(4, len(cells))
    rows = (len(cells) + cols - 1) // cols
    h, w = cells[0].shape[:2]
    sheet = np.full((rows * (h + 8) + 8, cols * (w + 8) + 8, 3), 255, np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        sheet[8 + r * (h + 8):8 + r * (h + 8) + h, 8 + k * (w + 8):8 + k * (w + 8) + w] = c
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ok, buf = cv2.imencode('.jpg', sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(a.out, 'wb') as f:
        f.write(buf.tobytes())
    print('->', a.out)


if __name__ == '__main__':
    main()
