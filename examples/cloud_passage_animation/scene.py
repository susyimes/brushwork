"""《云渡》. One water layer, one newly drawn blink and a pass of warm oil glints."""
import math
from pathlib import Path

import cv2
import numpy as np

from brushwork import Animation, PaintedLayer, RepaintLayer, StrokeLayer, Studio, paint_over


ASSETS = Path(__file__).with_name('assets')
SCALE = 1.6                       # pixels per composition unit in the 1600 x 2000 plate
TOP, HORIZON = 760, 772
WATER_Y = round(TOP * SCALE)


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def image(name, mode):
    return cv2.imdecode(np.fromfile(ASSETS/name, np.uint8), mode)


def waves(x, y, phase):
    h = np.maximum(0, y - HORIZON)
    a = 26.5*np.log1p(h/150) + .014*x + .52*np.sin(x/84+y/120)
    b = 39*np.log1p(h/115) - .012*x + .43*np.sin(x/103-y/68)
    t = math.tau * phase
    slip = .72*(np.sin(a-t)-np.sin(a)) + .28*(np.sin(b-2*t)-np.sin(b))
    lift = .66*(np.cos(a-t)-np.cos(a)) + .34*(np.cos(b-2*t)-np.cos(b))
    facing = .73*np.cos(a-t) + .27*np.cos(b-2*t)
    return slip, lift, facing


def reflection_layer(still):
    distance = image('water-distance.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)*(40/255)
    tile = still[WATER_Y:]
    height = (np.arange(tile.shape[0])[:, None] + .5)/SCALE + TOP
    matte = smooth((distance-.20)/4.6)*smooth((height-HORIZON)/12)

    def sample(frame, x, y):
        u, v = (x+.5)/SCALE, TOP+(y+.5)/SCALE
        dist = distance[np.clip(y.astype(int), 0, tile.shape[0]-1),
                        np.clip(x.astype(int), 0, tile.shape[1]-1)]
        d = np.clip((v-HORIZON)/(1250-HORIZON), 0, 1)
        reach = smooth(dist/26)*smooth((v-HORIZON)/17)
        slip, lift, _ = waves(u, v, frame.phase)
        ax = np.minimum(.42*dist, (1.2+13.2*d**1.25)*reach)
        ay = np.minimum(.31*dist, (.31+2.6*d)*reach)
        return -SCALE*ax*slip, -SCALE*ay*lift

    return PaintedLayer(tile, origin=(0,WATER_Y), warp=sample, mesh=5, edge='reflect', clip=matte)


def closure(t, delay=0):
    t -= 3.65+delay
    if t <= 0 or t >= .425:
        return 0.
    return float(smooth(t/.139) if t < .139 else 1-smooth((t-.191)/.234))


def blink_layer(still):
    with np.load(ASSETS/'eyelids.npz', allow_pickle=False) as file:
        original = file['tile']
        x0, y0 = file['origin']
        w, h = file['out_size']
        fields = ('origin','skin','upper','lower','closed','half_width','side','top',
                  'x','y','scale','offset')
        eyes = [{key:file[f'{key}{i}'] for key in fields} for i in range(2)]
    base = still[y0:y0+h,x0:x0+w].astype(np.int16)
    reference = cv2.resize(original, (w,h), interpolation=cv2.INTER_AREA).astype(np.int16)

    def fresh():
        return Studio(original.shape[1], original.shape[0], seed=913, medium='oil',
                      palette='with_black', log=None)

    def draw(st, frame):
        tile = original.copy()
        marks, patches = [], []
        for i, e in enumerate(eyes):
            k = closure(frame.seconds, .009*i)
            px, py = e['origin']
            eh, ew = e['skin'].shape[:2]
            box = np.s_[py:py+eh,px:px+ew]
            patches.append((box, smooth(k/.18)))
            if k <= 0:
                continue
            up = e['upper'] + (e['closed']-e['upper'])*k
            low = e['lower'] + (e['closed']-e['lower'])*k**1.25
            half = e['half_width']*(1-.49*k)
            y = e['y'][:, None]
            above = smooth((y-e['top']+.82)/1.3)*smooth((up+.68*half+.46-y)/.85)
            below = smooth((y-low+.52)/.92)*smooth((e['lower']+1.88-y)/1.48)
            cover = (np.maximum(above, below)*e['side'])[..., None]
            tile[box] = np.rint(original[box] + (e['skin'].astype(float)-original[box])*cover)
            start, end = (442.4,492.) if i == 0 else (526.5,574.1)
            xs = np.linspace(start, end, 100)
            mid = np.interp(xs,e['x'],up)
            widths = np.interp(xs,e['x'],half)

            def stroke(xx, yy, r, ink, opacity=.98):
                path = (np.column_stack((xx, yy))*e['scale'] + e['offset'] + e['origin'])/st.S
                marks.append(dict(path=path, brush=r*e['scale']/st.S, color=ink, style='line',
                                  thick=.52, opacity=opacity, dry=.13, pickup=0, unmix=.055, taper=.95))

            for offset,r,ink in [(1.03,.32,'#a58180'),(.74,.41,'#735c62'),(.51,.39,'#49414a'),
                                 (.23,.40,'#393641'),(-.05,.43,'#353440'),(-.32,.40,'#363441'),
                                 (-.63,.36,'#413b46'),(-.84,.28,'#55464e')]:
                trim = 1.8*abs(offset)
                keep = (xs >= start+trim) & (xs <= end-trim)
                stroke(xs[keep],(mid+offset*widths)[keep],r,ink)
            side = -1 if i==0 else 1
            for inset,length in ((6.,4.3),(8.5,3.1)):
                x = start+inset if i==0 else end-inset
                y = np.interp(x,xs,mid)
                stroke([x,x+side*.56*length,x+side*length],[y+.31,y+.10,y-.7-.4*k],
                       .34,'#403a44',.91)
        wet = paint_over(st, tile, lambda canvas: canvas.strokes(marks), amb=.83, relief=.51,
                         spec=.005, shadow=.11, weave_amp=.065, blur=.66)
        for box, blend in patches:
            wet[box] = np.rint(original[box] + (wet[box].astype(float)-original[box])*blend)
        change = cv2.resize(wet,(w,h),interpolation=cv2.INTER_AREA).astype(np.int16)-reference
        return np.uint8(np.clip(base+change,0,255))

    return RepaintLayer(fresh, draw, origin=(x0,y0), active=lambda f: 3.65 < f.seconds < 4.085)


def glint_layer():
    width, height = 512, 2000-WATER_Y
    def fresh():
        return Studio(width,height,seed=510,medium='oil',log=None)
    painter = fresh()
    rng = np.random.default_rng(4057)
    marks = []
    inks = ['#fff2d7','#ffeac8','#f5ddbc','#fff7e2']
    # Small families share a crest. A light passes from mark to mark across the left reflection.
    for _ in range(172):
        d = rng.beta(1.4,2.1)
        y = HORIZON+5+d*465
        x = rng.normal(146-77*d+20*np.sin(y/48), 48+43*d)
        for _ in range(rng.integers(2,5)):
            px, py = x+rng.normal(0,12+14*d), y+rng.normal(0,1.05+d)
            length = rng.uniform(1.3,6.1+12*d)
            if px<1 or px+length>311:
                continue
            path = np.array([(px,py+.10),(px+.30*length,py-.20),(px+.70*length,py-.14),
                             (px+length,py+.07)])
            local = (path-[0,TOP])*SCALE/painter.S
            marks.append(dict(path=local,brush=rng.uniform(.34,.85+.34*d)*SCALE/painter.S,
                              color=rng.choice(inks),style='impasto',thick=1.20,opacity=rng.uniform(.78,.98),
                              pickup=.018,unmix=.19,dry=.26,taper=.84))
    with painter.capture_strokes(seed=510) as tape:
        painter.strokes(marks)

    def move(batch, frame):
        x, y = batch.centers.T/SCALE
        y = y+TOP
        d = np.clip((y-HORIZON)/(1250-HORIZON),0,1)
        slip, lift, facing = waves(x,y,frame.phase)
        batch.translate(SCALE*(1.2+13.2*d**1.25)*slip, SCALE*(.31+2.6*d)*lift)
        batch.parameter('opacity')[:] *= .05+.95*smooth((facing+.22)/1.12)**1.65

    return StrokeLayer(fresh,tape,move,origin=(0,WATER_Y),over=True,
                       light=dict(amb=.60,relief=.74,spec=.043,shadow=.24,blur=.65))


def build():
    still = cv2.cvtColor(image('underpainting.webp',cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    scene = Animation(still, seconds=8, fps=30)
    scene.add('reflection', reflection_layer(still))
    scene.add('blink', blink_layer(still))
    scene.add('glints', glint_layer())
    return scene
