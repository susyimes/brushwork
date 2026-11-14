"""Lemons and a blue-and-white bowl: a small still life, start to finish, in about a hundred lines.

    python examples/still_life.py                 # 2000 x 1600 -> out/still_life.png
    python examples/still_life.py --width 1000    # quick study

A complete picture made only with the public brushwork API: an underdrawing built from
shapes and modelling calls, the standard passes, then a handful of explicit strokes.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from brushwork import Studio, catmull, ellipse_pts

LIGHT = (-1.0, -0.6)                      # the light comes from the upper left


def lemon(sk, name, c, rx, ry, rot):
    """An ellipse with a nipple at one end, lit, shaded, with a highlight and a cool bounce."""
    a = math.radians(rot)
    sk.ellipse(c, rx, ry, rot, color='#e6c022', region=name, detail=1.6,
               direction={'angle': rot + 25, 'spread': 30, 'scale': 60})
    sk.ellipse((c[0] + math.cos(a) * rx * 0.97, c[1] + math.sin(a) * rx * 0.97), rx * 0.14, ry * 0.2, rot,
               color='#d7ab1a', region=name)
    sk.form_light(name, '#fbe670', light=LIGHT, width=rx * 0.5, amount=0.8)
    sk.form_shadow(name, '#a2780f', light=LIGHT, width=rx * 0.55, amount=0.85)
    sk.tint(name, '#fff8c8', amount=0.9, blob=(c[0] - rx * 0.36, c[1] - ry * 0.42, rx * 0.2, ry * 0.15, rot))
    sk.tint(name, '#8c8a2e', amount=0.35, blob=(c[0] + rx * 0.3, c[1] + ry * 0.6, rx * 0.5, ry * 0.25))


def underdrawing(sk):
    """Back to front: wall, table, the inside of the bowl, the lemons in it, the bowl, the lemon in front."""
    sk.background({'linear': [[0, 0], [1000, 480]], 'colors': ['#676b60', '#2a2e2c']}, region='wall',
                  direction={'angle': 70, 'spread': 35}, detail=5.5)
    sk.rect(0, 470, 1000, 330, color={'linear': [[0, 470], [1000, 800]], 'colors': ['#aa845c', '#6c4e35']},
            region='table', direction={'angle': 3, 'spread': 6}, detail=4)
    sk.tint('table', '#3b2a1e', amount=0.5, ramp=((0, 690), (0, 800)))           # the near edge falls away

    cx, cy = 400, 400
    sk.ellipse((cx, cy), 190, 48, region='inside', detail=2.0, direction={'around': [cx, cy - 70]},
               color={'linear': [[225, 400], [575, 400]], 'colors': ['#56667f', '#9aa7ba', '#dde4ec']})
    lemon(sk, 'lemon_a', (345, 372), 80, 58, -18)
    lemon(sk, 'lemon_b', (462, 366), 76, 56, 22)
    sk.cast_shadow('lemon_a', 'lemon_b', '#8a6a10', offset=(26, 10), blur=5, amount=0.6)

    rim = ellipse_pts(cx, cy, 190, 48, n=40, a0=0.0, a1=math.pi)                  # the near half of the rim
    under = catmull([(210, 400), (218, 452), (262, 520), (330, 556), (470, 556), (538, 520), (582, 452), (590, 400)],
                    n=10)
    sk.polygon(np.vstack([rim, under[1:-1]]), region='bowl', detail=1.8,
               color={'linear': [[215, 430], [590, 500]], 'colors': ['#f2f5f7', '#dfe6ee', '#8a99af']},
               direction={'along': [[212, 420], [300, 500], [400, 520], [500, 500], [588, 420]]})
    sk.form_shadow('bowl', '#5d6b84', light=LIGHT, width=70, amount=0.7)
    sk.ellipse((400, 558), 74, 9, color='#b7c2d0', region='bowl')                 # the foot
    band = [[226, 462], [300, 492], [400, 502], [500, 492], [574, 462]]
    sk.line(band, 15, '#2b4a9b', amount=0.88, on='bowl')                          # cobalt band and its two rules
    for dy in (-15, 15):
        sk.line([[x, y + dy] for x, y in band], 2.2, '#3a57a0', amount=0.8, on='bowl')
    sk.line(np.vstack([ellipse_pts(cx, cy, 190, 48, n=64), [[cx + 190, cy]]]), 5, '#f6f9fb', smooth=False)

    lemon(sk, 'lemon', (725, 602), 108, 78, -10)
    sk.spline([(632, 566), (590, 524), (538, 504), (556, 546), (600, 578)], region='leaf', detail=1.3,
              color={'linear': [[630, 570], [540, 505]], 'colors': ['#48783a', '#2e5428']},
              direction={'along': [[630, 565], [545, 508]]})
    sk.line([[628, 563], [586, 533], [546, 509]], 2.4, '#a3c67e', amount=0.8, on='leaf')

    for caster, off, blur, amt in (('bowl', (58, 18), 7, 0.75), ('lemon', (44, 14), 5, 0.8), ('leaf', (14, 8), 3, 0.6)):
        sk.cast_shadow(caster, 'table', '#35261c', offset=off, blur=blur, amount=amt)
    sk.tint('table', '#281a12', amount=0.75, blob=(404, 566, 92, 8))              # where things touch the table
    sk.tint('table', '#281a12', amount=0.75, blob=(738, 678, 82, 7))


def accents(st):
    """The last touches, placed by hand: wet highlights, the glint on the rim, the brushed pattern."""
    items = []
    for (x, y), r in (((688, 568), 11.0), ((317, 349), 7.5), ((437, 343), 7.0)):
        items.append(dict(path=[[x - r, y + r * 0.3], [x + r, y - r * 0.3]], brush=r * 0.62, color='#fffce6',
                          style='dab', thick=2.6))
    glint = ellipse_pts(400, 400, 190, 48, n=24, a0=math.pi * 0.56, a1=math.pi * 0.93)
    items.append(dict(path=glint, brush=1.5, color='#ffffff', style='hair', thick=2.2, taper=0.8))
    for k in range(13):                                                   # a row of little brushed commas
        t = (k + 0.5) / 13
        x = 232 + 336 * t
        y = 462 + 40 * math.sin(math.pi * t)
        items.append(dict(path=[[x - 7, y - 8], [x, y + 1], [x + 8, y - 7]], brush=1.3, smooth=True, style='hair',
                          pigments={'ultramarine': .7, 'phthalo_blue': .3}, thick=1.4))
    st.strokes(items, on=['bowl', 'lemon', 'lemon_a', 'lemon_b', 'inside'])
    st.dabs(90, regions=['lemon', 'lemon_a', 'lemon_b'], brush=(0.4, 1.0), colors=['#f6e27a', '#cf9c14'],
            style='body', opacity=0.5, thick=0.6, seed=4)                # the pitted skin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--width', type=int, default=2000)
    ap.add_argument('--out', default=os.path.join(ROOT, 'out', 'still_life'))
    ap.add_argument('--palette', nargs='+', default=['full'],
                    help='tubes to mix from, e.g. "white ochre cad_red ivory_black" for a Zorn palette')
    a = ap.parse_args()

    st = Studio(width=a.width, height=a.width * 4 // 5, seed=5, palette=a.palette)
    underdrawing(st.sk)
    st.tone_ground('#75604a')
    st.underpaint()
    st.block_in()
    st.cut_in('wall')
    st.cut_in('table')
    st.mend()
    st.refine(passes='broad', label='broad correction')
    st.refine(passes='coarse')
    st.refine(passes='fine', label='fine refinement')
    st.mend()
    accents(st)
    st.save_image(a.out + '.png')
    st.look(a.out + '_sketch_and_canvas.jpg', what='both', width=900)
    print('done: %(brushstrokes)d brushstrokes, %(bristle_trails)d bristle trails' % st.stats(), '->', a.out + '.png')


if __name__ == '__main__':
    main()
