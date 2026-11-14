"""Small vector-drawing toolkit used to construct the underdrawing.

All geometry is authored in canvas units (the canvas is 1000 units wide) and rasterised
at whatever resolution the run uses.
"""
import math

import cv2
import numpy as np


def catmull(P, closed=False, n=12):
    """Centripetal Catmull-Rom spline through the points P (k,2) -> dense polyline."""
    P = np.asarray(P, np.float64)
    if closed:
        Q = np.vstack([P[-1], P, P[0], P[1]])
        segs = len(P)
    else:
        Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
        segs = len(P) - 1
    out = []
    for i in range(segs):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        t = np.linspace(t1, t2, n, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        out.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    if not closed:
        out.append(P[-1][None])
    return np.vstack(out)


def arclen(pl):
    d = np.hypot(*np.diff(pl, axis=0).T)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(pl, step=None, n=None):
    """Uniform arc-length resampling of a polyline."""
    pl = np.asarray(pl, np.float64)
    s = arclen(pl)
    L = s[-1]
    if L <= 1e-9:
        return pl[:1].repeat(2, 0)
    if n is None:
        n = max(2, int(math.ceil(L / step)) + 1)
    t = np.linspace(0, L, n)
    return np.stack([np.interp(t, s, pl[:, 0]), np.interp(t, s, pl[:, 1])], 1)


def tangents(pl):
    t = np.gradient(np.asarray(pl, np.float64), axis=0)
    return t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-12)


def normals(pl):
    t = tangents(pl)
    return np.stack([-t[:, 1], t[:, 0]], 1)


def offset(pl, d):
    """Polyline shifted sideways by d (scalar or per-point); +d is to the left of travel in
    a y-down frame, i.e. along (-ty, tx)."""
    return np.asarray(pl, np.float64) + normals(pl) * np.reshape(d, (-1, 1))


def ribbon(center, width):
    """Closed polygon around a centre line with per-point full width."""
    n = normals(center)
    w = np.reshape(width, (-1, 1)) * 0.5
    return np.vstack([center + n * w, (center - n * w)[::-1]])


def lock_profile(t, p=2.2, q=0.85, root=1.0):
    """Hair-lock width profile: full at the root, a point at the tip."""
    return (1.0 - t ** p) ** q * (root + (1 - root) * np.sin(np.pi * np.minimum(t * 1.4, 1.0)))


def ellipse_pts(cx, cy, rx, ry, rot=0.0, n=48, a0=0.0, a1=2 * math.pi):
    a = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * math.pi - 1e-9)
    x = np.cos(a) * rx
    y = np.sin(a) * ry
    c, s = math.cos(rot), math.sin(rot)
    return np.stack([cx + x * c - y * s, cy + x * s + y * c], 1)


def mirror_x(P):
    P = np.asarray(P, np.float64).copy()
    P[..., 0] *= -1
    return P


class Raster:
    """Rasterises unit-space shapes to float masks at guide resolution."""

    def __init__(self, GW, GH, s):
        self.GW, self.GH, self.s = GW, GH, float(s)
        yy, xx = np.mgrid[0:GH, 0:GW].astype(np.float32)
        self.X = (xx + 0.5) / s
        self.Y = (yy + 0.5) / s

    def poly(self, pts):
        m = np.zeros((self.GH, self.GW), np.uint8)
        p = np.round(np.asarray(pts, np.float64) * self.s * 16.0).astype(np.int32)
        cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_AA, shift=4)
        return m.astype(np.float32) / 255.0

    def polys(self, plist):
        m = np.zeros((self.GH, self.GW), np.float32)
        for p in plist:
            m = np.maximum(m, self.poly(p))
        return m

    def stroke(self, pl, width):
        """Constant- or variable-width line as a mask."""
        pl = np.asarray(pl, np.float64)
        w = np.broadcast_to(np.asarray(width, np.float64), (len(pl),))
        m = self.poly(ribbon(pl, w))
        for k in (0, -1):
            if w[k] > 1e-6:
                m = np.maximum(m, self.poly(ellipse_pts(pl[k, 0], pl[k, 1], w[k] / 2, w[k] / 2, n=16)))
        return m

    def ellipse(self, cx, cy, rx, ry, rot=0.0):
        return self.poly(ellipse_pts(cx, cy, rx, ry, rot, n=72))

    def blur(self, m, sigma):
        sg = max(0.01, sigma * self.s)
        return cv2.GaussianBlur(m, (0, 0), sg)

    def shift(self, m, dx, dy):
        M = np.float32([[1, 0, dx * self.s], [0, 1, dy * self.s]])
        return cv2.warpAffine(m, M, (self.GW, self.GH), flags=cv2.INTER_LINEAR, borderValue=0)

    def sdf(self, m):
        """Signed distance in units: positive inside the mask."""
        b = (m > 0.5).astype(np.uint8)
        di = cv2.distanceTransform(b, cv2.DIST_L2, 5)
        do = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
        return (di - do) / self.s

    def field_from_polyline(self, vx, vy, pl, width, weight=None):
        """Write the tangent direction of a ribbon into the direction field (vx, vy)."""
        pl = np.asarray(pl, np.float64)
        t = tangents(pl)
        w = np.broadcast_to(np.asarray(width, np.float64), (len(pl),))
        n = np.stack([-t[:, 1], t[:, 0]], 1)
        L = pl + n * w[:, None] * 0.5
        R = pl - n * w[:, None] * 0.5
        tmp = np.zeros((self.GH, self.GW), np.uint8)
        for k in range(len(pl) - 1):
            quad = np.round(np.array([L[k], L[k + 1], R[k + 1], R[k]]) * self.s * 16).astype(np.int32)
            tmp[:] = 0
            x0 = max(0, int(quad[:, 0].min() / 16) - 1)
            x1 = min(self.GW, int(quad[:, 0].max() / 16) + 2)
            y0 = max(0, int(quad[:, 1].min() / 16) - 1)
            y1 = min(self.GH, int(quad[:, 1].max() / 16) + 2)
            if x1 <= x0 or y1 <= y0:
                continue
            sub = np.zeros((y1 - y0, x1 - x0), np.uint8)
            q = quad - np.array([x0 * 16, y0 * 16])
            cv2.fillConvexPoly(sub, q, 1, shift=4)
            sel = sub > 0
            tx, ty = (t[k] + t[k + 1]) * 0.5
            vx[y0:y1, x0:x1][sel] = tx
            vy[y0:y1, x0:x1][sel] = ty
