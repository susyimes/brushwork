"""Separate the two painted irises from the project's flattened gouache portrait.

The ellipse measurements come from works/fingertip-starlight/face.py. Lid
curves follow the actual finished paint at 2560 × 1888, including the lower
lashes. This preparation is specific to this portrait, independent of editing.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree

from brushwork import ObjectEditor


HERE = Path(__file__).resolve().parent

# Fixed eyelid outlines in the native-pixel eye detail starting at (930, 570).
# Each pair of curves follows the upper and lower lid from left to right.
FAR = (
    [(84,251),(100,234),(120,226),(140,221),(160,221),(175,223),(190,225),
     (205,228),(220,232),(235,239),(250,248),(265,259),(280,271),(289,284),(295,321)],
    [(84,251),(94,284),(109,313),(120,332),(140,341),(160,352),(175,359),(190,363),
     (205,365),(216,364),(235,356),(250,350),(265,346),(280,341),(292,337),(295,321)],
)
NEAR = (
    [(514,197),(530,174),(545,152),(560,133),(575,119),(590,109),(605,100),
     (620,94),(635,89),(650,84),(665,82),(680,80),(695,79),(710,79),(725,80),
     (740,82),(755,84),(770,87),(785,92),(805,99),(818,105),(821,123)],
    [(514,197),(540,205),(560,218),(575,239),(590,255),(605,260),(620,257),
     (635,253),(650,250),(665,247),(680,246),(695,245),(710,242),(725,238),
     (740,236),(755,232),(770,223),(785,209),(805,193),(817,171),(821,123)],
)
EYES = (
    ('far_iris', (570.8, 432.1, 25.6, 39.4, -31), FAR),
    ('near_iris', (779.4, 380.0, 34.2, 40.6, -21), NEAR),
)


def aperture(path):
    curves = []
    for curve in path:
        xx, yy = np.array(curve).T
        xp = np.arange(xx[0], xx[-1]+.5, .5)
        curves.append(np.column_stack((xp, PchipInterpolator(xx, yy)(xp))))
    pts = np.vstack([curves[0], curves[1][::-1]]) + (930, 570)
    # Leave room in the moving tile for parts of the iris hidden by the lids.
    x, y = np.floor(pts.min(axis=0)).astype(int) - 24
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int) + 25
    # Supersampling preserves the narrow turns at both corners of each lid.
    high = np.zeros(((y1-y)*4, (x1-x)*4), np.uint8)
    cv2.fillPoly(high, [np.rint((pts - (x, y)) * 4).astype(np.int32)], 255)
    return cv2.resize(high, (x1-x, y1-y), interpolation=cv2.INTER_AREA), (int(x), int(y))


def eye_layers(original, underpainting, geometry, path):
    clip, (x, y) = aperture(path)
    h, w = clip.shape
    tile = original[y:y+h, x:x+w]
    yy, xx = np.indices((h, w))
    cx, cy, rx, ry, degrees = geometry
    angle = np.deg2rad(degrees)
    dx, dy = xx+x-2*cx, yy+y-2*cy
    radial_x = (dx*np.cos(angle)+dy*np.sin(angle)) / (2*rx)
    radial_y = (dy*np.cos(angle)-dx*np.sin(angle)) / (2*ry)
    radius = np.hypot(radial_x, radial_y)
    alpha = np.rint(255*np.clip((1.095-radius)/.046, 0, 1)).astype(np.uint8)

    # Reconstruct the hidden eye white from the visible paint in this eye.
    # Fit its gentle lighting; transfer the remaining paper grain from the
    # opposite side of the eye so the exposed crescent retains the paper.
    safe = (cv2.erode(clip, np.ones((7, 7), np.uint8)) == 255)
    white = safe & (radius > 1.13) & (tile.min(axis=2) > 132)
    nx = (xx - w/2)/w
    ny = (yy - h/2)/h
    basis = np.stack([np.ones_like(nx), nx, ny, nx*ny, ny*ny, nx*nx], axis=-1)
    coeff = np.linalg.lstsq(basis[white], tile[white], rcond=None)[0]
    lighting = basis @ coeff
    interior = cv2.distanceTransform(white.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sy, sx = np.unravel_index(interior.argmax(), interior.shape)
    side = max(2, int(interior[sy,sx] / np.sqrt(2)) - 1)
    paper = tile[sy-side:sy+side, sx-side:sx+side].astype(np.float32)
    grain = paper - cv2.GaussianBlur(paper, (0,0), 1.8)
    repeats = ((h+grain.shape[0]-1)//grain.shape[0], (w+grain.shape[1]-1)//grain.shape[1], 1)
    clean = np.clip(np.rint(lighting + np.tile(grain, repeats)[:h, :w]), 0, 255).astype(np.uint8)

    # Only fully covered pixels need a new base. Edge pixels shared with the
    # stationary lid retain their actual antialiasing in the original pose.
    opaque = (clip == 255) & (alpha == 255)
    underpainting[y:y+h, x:x+w][opaque] = clean[opaque]

    rgb = tile.copy()
    # A flat painting contains no colour for the parts behind the lids. Extend
    # the nearest painted rim at the same iris radius into those small arcs.
    visible = (clip > 0) & (radius <= 1.10)
    hidden = (clip == 0) & (alpha > 0)
    # The tile must include the obscured cap even beyond the aperture's bounds.
    iy, ix = np.nonzero(visible)
    feature = np.stack([radius*5, radial_y*1.6, radial_x], axis=-1)
    if hidden.any():
        _, nearest = cKDTree(feature[visible]).query(feature[hidden])
        rgb[hidden] = tile[iy[nearest], ix[nearest]]
    rgba = np.dstack((rgb, alpha))
    return rgba, (x, y), clip


def open_portrait():
    """Build an editable portrait. Its initial render equals the original pixels."""
    original = ObjectEditor(HERE / 'starlight.webp').render()
    underpainting = original.copy()
    selected = np.zeros(original.shape[:2], bool)
    layers = []
    for name, geometry, path in EYES:
        rgba, xy, clip = eye_layers(original, underpainting, geometry, path)
        x, y = xy
        h, w = clip.shape
        selected[y:y+h, x:x+w] |= clip > 0
        layers.append((name, rgba, xy, clip))
    editor = ObjectEditor(underpainting)
    for name, rgba, xy, clip in layers:
        editor.add(name, rgba, xy=xy, clip=clip, clip_xy=xy)
    if not np.array_equal(editor.render(), original):
        raise RuntimeError('Separating the eyes changed the original painting')
    return editor, original, selected
