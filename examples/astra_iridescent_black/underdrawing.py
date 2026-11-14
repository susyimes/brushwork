"""Render the underdrawing on its own: what the painter intends before any paint is laid.

    python underdrawing.py [width]     ->  out/astra_underdrawing.jpg, out/astra_underdrawing_directions.jpg

The first image is the intended colour of every place (the blacks are only a dark tint here;
their colour is decided on the palette).  The second overlays the direction the brush will
travel.  Both are constructed from coordinates in scene.py; nothing is loaded.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import cv2
import numpy as np

import scene
from brushwork import color as C


def save(path, im, q=92):
    ok, buf = cv2.imencode('.jpg', im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def main():
    GW = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    GH = GW * 5 // 4
    t = time.time()
    sc = scene.build(GW, GH)
    print('underdrawing built in %.1fs' % (time.time() - t))
    out = os.path.join(ROOT, 'out')
    os.makedirs(out, exist_ok=True)
    img = (C.lin_to_srgb(sc.target) * 255 + 0.5).astype(np.uint8)
    save(os.path.join(out, 'astra_underdrawing.jpg'), img)
    f = img.copy()
    step = max(8, GW // 70)
    for y in range(step // 2, GH, step):
        for x in range(step // 2, GW, step):
            dx, dy = sc.vx[y, x], sc.vy[y, x]
            cv2.line(f, (int(x - dx * step * .4), int(y - dy * step * .4)),
                     (int(x + dx * step * .4), int(y + dy * step * .4)), (255, 120, 40), 1, cv2.LINE_AA)
    save(os.path.join(out, 'astra_underdrawing_directions.jpg'), f)


if __name__ == '__main__':
    main()
