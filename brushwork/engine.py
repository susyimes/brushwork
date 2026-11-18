"""Bristle-brush oil paint simulation.

The canvas is a stack of per-pixel fields: paint reflectance, scattering strength, wetness,
relief height (impasto), accumulated thickness, and an interference-medium layer (amount +
optical film thickness).  A stroke is a ribbon swept by a row of bristles.  Every bristle
carries its own pigment mix and paint load; as it travels it

  * lays paint down (opaque hiding, broken up by canvas tooth as the load runs out),
  * picks wet paint back up and mixes it into its own load in Kubelka-Munk space,
  * ploughs its own furrow into the relief, squeezes a ridge out at the stroke edges and
    leaves a blob where the brush lifts.

Nothing here knows what is being painted; the picture lives in scene.py / painter.py.
"""
import math

import cv2
import numpy as np
from numba import njit, prange

from . import color as C

# ---- stroke parameter columns ------------------------------------------------------------
(P_RADIUS, P_OPACITY, P_THICK, P_LOADLEN, P_DRY, P_PICKUP, P_UNMIX, P_CAP,
 P_NB, P_FURROW, P_RIDGE, P_IRI, P_FILM, P_FILMVAR, P_WETDEP, P_TAPER,
 P_MODE, P_RAGGED, P_MULT, P_BLOB, P_WOBBLE, P_SKIP, P_FILMLOAD) = range(23)
NPAR = 23

DEFAULTS = dict(radius=10.0, opacity=0.95, thick=1.0, loadlen=1e9, dry=0.45, pickup=0.02,
                unmix=0.6, cap=0.8, nb=0.0, furrow=0.6, ridge=0.8, iri=0.0, film=600.0,
                filmvar=0.0, wetdep=1.0, taper=0.35, mode=0.0, ragged=0.6, mult=0.0, blob=0.6, wobble=1.0, skip=0.35,
                filmload=0.45)
_COLS = dict(radius=P_RADIUS, opacity=P_OPACITY, thick=P_THICK, loadlen=P_LOADLEN, dry=P_DRY,
             pickup=P_PICKUP, unmix=P_UNMIX, cap=P_CAP, nb=P_NB, furrow=P_FURROW,
             ridge=P_RIDGE, iri=P_IRI, film=P_FILM, filmvar=P_FILMVAR, wetdep=P_WETDEP,
             taper=P_TAPER, mode=P_MODE, ragged=P_RAGGED, mult=P_MULT, blob=P_BLOB,
             wobble=P_WOBBLE, skip=P_SKIP, filmload=P_FILMLOAD)


def make_params(n, **kw):
    """(n, NPAR) float32 parameter block; every keyword may be a scalar or an (n,) array."""
    sp = np.empty((n, NPAR), np.float32)
    for k, v in DEFAULTS.items():
        sp[:, _COLS[k]] = v
    for k, v in kw.items():
        sp[:, _COLS[k]] = v
    return sp


# ---- tiny deterministic RNG usable inside numba -------------------------------------------
@njit(cache=True)
def _rnd(st):
    s = st[0]
    s = s * 6364136223846793005 + 1442695040888963407
    st[0] = s
    return ((s >> 33) & 0x7FFFFFFF) * 4.656612873077393e-10


@njit(cache=True)
def _wprof(s, L, R, taper):
    a = s / (2.0 * R + 1e-3)
    if a > 1.0:
        a = 1.0
    b = (L - s) / (2.5 * R + 1e-3)
    if b > 1.0:
        b = 1.0
    return (1.0 - taper * (1.0 - a) * (1.0 - a)) * (1.0 - taper * (1.0 - b) * (1.0 - b))


@njit(cache=True, fastmath=True)
def paint_strokes(col, sca, wet, hgt, thk, iri, ifd, stamp, tooth,
                  pts, off, sp, seeds, comp_idx, comp_w, pig_K, pig_S, order, stamp_base,
                  clip=None, deposited=None):
    """Lay strokes on the canvas, in `order`.  Returns (pixels touched, bristle trails)."""
    H = hgt.shape[0]
    W = hgt.shape[1]
    MAXB = 128
    bR = np.zeros((MAXB, 3), np.float32)
    bK = np.zeros((MAXB, 3), np.float32)
    bS = np.zeros(MAXB, np.float32)
    bld = np.zeros(MAXB, np.float32)
    bgh = np.zeros(MAXB + 1, np.float32)
    bfd = np.zeros(MAXB, np.float32)
    bs0 = np.zeros(MAXB, np.float32)
    bs1 = np.zeros(MAXB, np.float32)
    bfq = np.zeros(MAXB, np.float32)
    bph = np.zeros(MAXB, np.float32)
    blend_ready = np.zeros(MAXB, np.bool_)
    zc = np.zeros(4, np.float64)
    rs = np.zeros(1, np.int64)
    npix = 0
    ntrail = 0
    SPC = 0.7  # forward-mapping sample pitch; <= 1/sqrt(2) guarantees no holes
    for oi in range(order.shape[0]):
        i = order[oi]
        n0 = off[i]
        n1 = off[i + 1]
        if n1 - n0 < 2:
            continue
        R = sp[i, P_RADIUS]
        op = sp[i, P_OPACITY]
        thick = sp[i, P_THICK]
        lam = sp[i, P_LOADLEN]
        dry = sp[i, P_DRY]
        pick = sp[i, P_PICKUP]
        unmix = sp[i, P_UNMIX]
        cap = sp[i, P_CAP]
        furrow = sp[i, P_FURROW]
        ridge = sp[i, P_RIDGE]
        iri_b = sp[i, P_IRI]
        film = sp[i, P_FILM]
        filmvar = sp[i, P_FILMVAR]
        wetdep = sp[i, P_WETDEP]
        taper = sp[i, P_TAPER]
        mode = sp[i, P_MODE]
        rag = sp[i, P_RAGGED]
        mult = sp[i, P_MULT]
        blob = sp[i, P_BLOB]
        wobble = sp[i, P_WOBBLE]
        skip = sp[i, P_SKIP]
        filmload = sp[i, P_FILMLOAD]

        L = 0.0
        for k in range(n0, n1 - 1):
            dx = pts[k + 1, 0] - pts[k, 0]
            dy = pts[k + 1, 1] - pts[k, 1]
            L += math.sqrt(dx * dx + dy * dy)
        if L < 0.3:
            continue

        NB = int(sp[i, P_NB])
        if NB <= 0:
            NB = int(2.0 * R / 2.4)
        if NB < 2:
            NB = 2
        if NB > MAXB:
            NB = MAXB
        ntrail += NB
        pixels_before = npix
        rs[0] = seeds[i] * 2654435761 + 88172645463325252
        _rnd(rs)

        # ---- load the brush: every bristle gets its own slightly different mix ----------
        for c in range(4):
            zc[c] = (_rnd(rs) * 2.0 - 1.0) * 1.3
        zg = (_rnd(rs) * 2.0 - 1.0)
        zh = (_rnd(rs) * 2.0 - 1.0)
        zf = (_rnd(rs) * 2.0 - 1.0)
        # the brush never tracks perfectly: lanes meander, splay and the width breathes
        w_f1 = 6.2832 / (R * (7.0 + 9.0 * _rnd(rs)) + 4.0)
        w_f2 = 6.2832 / (R * (2.5 + 3.0 * _rnd(rs)) + 2.0)
        w_f3 = 6.2832 / (R * (9.0 + 9.0 * _rnd(rs)) + 4.0)
        w_p1 = 6.2832 * _rnd(rs)
        w_p2 = 6.2832 * _rnd(rs)
        w_p3 = 6.2832 * _rnd(rs)
        rho = 0.78
        sq = math.sqrt(1.0 - rho * rho) * 1.732
        for j in range(NB):
            uj = (j + 0.5) / NB * 2.0 - 1.0
            K0 = 0.0
            K1 = 0.0
            K2 = 0.0
            S = 0.0
            wsum = 0.0
            for c in range(4):
                wc = comp_w[i, c]
                if wc <= 0.0:
                    continue
                zc[c] = rho * zc[c] + sq * (_rnd(rs) * 2.0 - 1.0)
                wv = wc * math.exp(unmix * zc[c])
                p = comp_idx[i, c]
                K0 += wv * pig_K[p, 0]
                K1 += wv * pig_K[p, 1]
                K2 += wv * pig_K[p, 2]
                S += wv * pig_S[p]
                wsum += wv
            if wsum <= 0.0:
                wsum = 1.0
                S = 1.0
            K0 /= wsum
            K1 /= wsum
            K2 /= wsum
            S /= wsum
            bK[j, 0] = K0
            bK[j, 1] = K1
            bK[j, 2] = K2
            bS[j] = S
            x = K0 / S
            bR[j, 0] = 1.0 + x - math.sqrt(x * x + 2.0 * x)
            x = K1 / S
            bR[j, 1] = 1.0 + x - math.sqrt(x * x + 2.0 * x)
            x = K2 / S
            bR[j, 2] = 1.0 + x - math.sqrt(x * x + 2.0 * x)
            zg = 0.6 * zg + 0.8 * 1.732 * (_rnd(rs) * 2.0 - 1.0)
            zh = 0.93 * zh + 0.3676 * 1.732 * (_rnd(rs) * 2.0 - 1.0)
            g = 0.5 + 0.17 * zg + 0.22 * zh + 0.25 * (_rnd(rs) - 0.5)
            if g < 0.0:
                g = 0.0
            if g > 1.0:
                g = 1.0
            bgh[j] = 1.0 - furrow * g
            r3 = _rnd(rs)
            bld[j] = (1.0 - 0.6 * r3 * r3 * r3) * (1.0 - 0.22 * uj * uj * uj * uj)
            zf = 0.8 * zf + 0.6 * 1.732 * (_rnd(rs) * 2.0 - 1.0)
            bfd[j] = film + filmvar * zf
            e = 1.0 - math.sqrt(max(0.0, 1.0 - uj * uj))
            bs0[j] = cap * R * e + rag * R * 0.30 * _rnd(rs)
            r4 = _rnd(rs)
            bs1[j] = cap * R * e + rag * R * 1.2 * r4 * r4
            bfq[j] = 6.2832 / (R * (1.2 + 4.0 * _rnd(rs)) + 3.0)
            bph[j] = 6.2832 * _rnd(rs)
        bgh[NB] = bgh[NB - 1]

        lane = 2.0 * R / NB
        inv_lane = 1.0 / lane
        if inv_lane > 1.0:
            inv_lane = 1.0
        sid = stamp_base + oi + 1
        first = True
        s_acc = 0.0

        for k in range(n0, n1 - 1):
            x0 = pts[k, 0]
            y0 = pts[k, 1]
            x1 = pts[k + 1, 0]
            y1 = pts[k + 1, 1]
            sl = math.sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0))
            if sl < 1e-5:
                continue
            ka = k - 1 if k > n0 else k
            tx0 = pts[k + 1, 0] - pts[ka, 0]
            ty0 = pts[k + 1, 1] - pts[ka, 1]
            tn = math.sqrt(tx0 * tx0 + ty0 * ty0) + 1e-9
            tx0 /= tn
            ty0 /= tn
            kb = k + 2 if k + 2 < n1 else k + 1
            tx1 = pts[kb, 0] - pts[k, 0]
            ty1 = pts[kb, 1] - pts[k, 1]
            tn = math.sqrt(tx1 * tx1 + ty1 * ty1) + 1e-9
            tx1 /= tn
            ty1 /= tn
            nx0 = -ty0
            ny0 = tx0
            nx1 = -ty1
            ny1 = tx1
            s0 = s_acc
            r0 = R * _wprof(s0, L, R, taper)
            r1 = R * _wprof(s0 + sl, L, R, taper)
            ax = (x1 + nx1 * r1) - (x0 + nx0 * r0)
            ay = (y1 + ny1 * r1) - (y0 + ny0 * r0)
            o1 = math.sqrt(ax * ax + ay * ay)
            ax = (x1 - nx1 * r1) - (x0 - nx0 * r0)
            ay = (y1 - ny1 * r1) - (y0 - ny0 * r0)
            o2 = math.sqrt(ax * ax + ay * ay)
            outer = max(sl, max(o1, o2))
            na = int(outer / SPC) + 1

            if first and mode > 0.5:
                # blending brush: starts with whatever paint is already under it
                for j in range(NB):
                    uj = (j + 0.5) / NB * 2.0 - 1.0
                    qx = int(x0 + nx0 * uj * r0 + 0.5)
                    qy = int(y0 + ny0 * uj * r0 + 0.5)
                    if qx < 0:
                        qx = 0
                    if qy < 0:
                        qy = 0
                    if qx >= W:
                        qx = W - 1
                    if qy >= H:
                        qy = H - 1
                    if clip is not None:
                        blend_ready[j] = clip[qy, qx]
                        if not blend_ready[j]:
                            continue
                    Sp = sca[qy, qx]
                    bS[j] = Sp
                    for c in range(3):
                        Rp = col[qy, qx, c]
                        bR[j, c] = Rp
                        bK[j, c] = Sp * (1.0 - Rp) * (1.0 - Rp) / (2.0 * Rp + 1e-4)
            first = False

            for ia in range(na):
                f = (ia + 0.5) / na
                cx = x0 + (x1 - x0) * f
                cy = y0 + (y1 - y0) * f
                nx = nx0 + (nx1 - nx0) * f
                ny = ny0 + (ny1 - ny0) * f
                nn = math.sqrt(nx * nx + ny * ny) + 1e-9
                nx /= nn
                ny /= nn
                s = s0 + sl * f
                r = (r0 + (r1 - r0) * f) * (1.0 + 0.06 * wobble * math.sin(s * w_f3 + w_p3))
                lw = wobble * (1.1 * math.sin(s * w_f1 + w_p1) + 0.5 * math.sin(s * w_f2 + w_p2))
                pin = s / (0.6 * R + 1.0)
                if pin > 1.0:
                    pin = 1.0
                pout = (L - s) / (1.4 * R + 1.0)
                if pout > 1.0:
                    pout = 1.0
                press = (0.45 + 0.55 * pin) * (0.25 + 0.75 * pout)
                dep = math.exp(-s / lam)
                e1 = (L - s) / (0.45 * R + 0.5)
                e2 = s / (0.5 * R + 0.5)
                eb = blob * (math.exp(-e1 * e1) + 0.35 * math.exp(-e2 * e2))
                nb = int(2.0 * r / SPC) + 1
                for ib in range(nb):
                    u = (ib + 0.5) / nb * 2.0 - 1.0
                    fx = cx + nx * u * r
                    fy = cy + ny * u * r
                    px = int(math.floor(fx + 0.5))
                    py = int(math.floor(fy + 0.5))
                    if px < 0 or py < 0 or px >= W or py >= H:
                        continue
                    if clip is not None and not clip[py, px]:
                        continue
                    if stamp[py, px] == sid:
                        continue
                    xb = (u * 0.5 + 0.5) * NB - 0.5 + lw * (1.0 - u * u)
                    j = int(xb + 0.5)
                    if j < 0:
                        j = 0
                    if j >= NB:
                        j = NB - 1
                    if s < bs0[j] or s > L - bs1[j]:
                        continue
                    stamp[py, px] = sid
                    w_old = wet[py, px]
                    if clip is not None and mode > 0.5 and not blend_ready[j]:
                        # A blending bristle entering the selection picks up its first colour here.
                        Sp = sca[py, px]
                        bS[j] = Sp
                        for c in range(3):
                            Rp = col[py, px, c]
                            bR[j, c] = Rp
                            bK[j, c] = Sp * (1.0 - Rp) * (1.0 - Rp) / (2.0 * Rp + 1e-4)
                        blend_ready[j] = True

                    # -- pick wet paint up into the bristle (Kubelka-Munk mixing) ---------
                    if pick > 0.0 and w_old > 0.02:
                        pk = pick * w_old * press * inv_lane
                        if pk > 0.4:
                            pk = 0.4
                        Sp = sca[py, px]
                        Sn = bS[j] + (Sp - bS[j]) * pk
                        bS[j] = Sn
                        for c in range(3):
                            Rp = col[py, px, c]
                            Kp = Sp * (1.0 - Rp) * (1.0 - Rp) / (2.0 * Rp + 1e-4)
                            Kn = bK[j, c] + (Kp - bK[j, c]) * pk
                            bK[j, c] = Kn
                            x = Kn / Sn
                            bR[j, c] = 1.0 + x - math.sqrt(x * x + 2.0 * x)

                    # -- how much paint actually transfers --------------------------------
                    if mode > 0.5:
                        a = op * press * min(1.0, w_old * 1.6)
                        q = 0.6
                    else:
                        ct = math.sin(s * bfq[j] + bph[j])
                        if ct < 0.0:
                            ct = 0.0
                        q = bld[j] * dep * press * (1.0 - skip * ct * ct * ct)
                        cv = (q - dry * (1.0 - tooth[py, px])) * 4.0
                        if cv <= 0.0:
                            continue
                        if cv > 1.0:
                            cv = 1.0
                        a = op * cv
                    if a <= 0.001:
                        continue
                    ia_ = 1.0 - a
                    if mult > 0.0:
                        for c in range(3):
                            o = col[py, px, c]
                            n_ = bR[j, c]
                            col[py, px, c] = o * ia_ + a * (n_ * (1.0 - mult) + o * n_ * mult)
                    else:
                        for c in range(3):
                            col[py, px, c] = col[py, px, c] * ia_ + bR[j, c] * a
                    sca[py, px] = sca[py, px] * ia_ + bS[j] * a
                    wet[py, px] = w_old * ia_ + wetdep * a

                    # -- relief: furrows, squeezed-out edge ridge, lift-off blob ----------
                    j0 = int(math.floor(xb))
                    fr = xb - j0
                    if j0 < 0:
                        j0 = 0
                        fr = 0.0
                    if j0 >= NB:
                        j0 = NB - 1
                    g = bgh[j0] + (bgh[j0 + 1] - bgh[j0]) * fr
                    u2 = u * u
                    T = thick * ((0.3 + 0.7 * q) * g * (1.0 + ridge * u2 * u2 * u2) + eb * q)
                    flat = 0.35 + 0.6 * w_old
                    hgt[py, px] = hgt[py, px] * (1.0 - a * flat) + a * T
                    thk[py, px] += a * thick * (0.3 + 0.7 * q)
                    if iri_b > 0.0 or iri[py, px] > 0.0:
                        io = iri[py, px]
                        inew = io * ia_ + iri_b * a
                        if inew > 1e-5:
                            fdn = bfd[j] * (1.0 - filmload * (1.0 - q))
                            ifd[py, px] = (ifd[py, px] * io * ia_ + fdn * iri_b * a) / inew
                        iri[py, px] = inew
                    npix += 1
            s_acc += sl
        if deposited is not None:
            deposited[oi] = npix > pixels_before
            if not deposited[oi]:
                ntrail -= NB
    return npix, ntrail


@njit(parallel=True, cache=True, fastmath=True)
def trace_paths(seeds, vx, vy, gscale, nfwd, nbwd, step, inertia, region, lab, edge_tol,
                use_region, margin, over, out, nout):
    """Follow the direction field from every seed, both ways, into padded polylines.

    seeds are full-res pixel positions; vx/vy/region/lab live at guide resolution
    (guide px = full px * gscale).  A path stops at the canvas edge, when it leaves the
    region it started in, or when the target colour drifts more than edge_tol (OKLab).
    """
    N = seeds.shape[0]
    GH = vx.shape[0]
    GW = vx.shape[1]
    M = out.shape[1]
    Wf = GW / gscale
    Hf = GH / gscale
    tol2 = edge_tol * edge_tol
    for i in prange(N):
        buf = np.empty((M, 2), np.float32)
        sx = seeds[i, 0]
        sy = seeds[i, 1]
        gx = min(max(int(sx * gscale), 0), GW - 1)
        gy = min(max(int(sy * gscale), 0), GH - 1)
        reg0 = region[gy, gx]
        L0 = lab[gy, gx, 0]
        a0 = lab[gy, gx, 1]
        b0 = lab[gy, gx, 2]
        st = step[i]
        d0x = vx[gy, gx]
        d0y = vy[gy, gx]
        dn = math.sqrt(d0x * d0x + d0y * d0y)
        if dn < 1e-6:
            d0x = 1.0
            d0y = 0.0
        else:
            d0x /= dn
            d0y /= dn
        n = 0
        for direction in range(2):
            if direction == 0:
                dx = -d0x
                dy = -d0y
                maxs = min(nbwd[i], M // 2 - 1)
            else:
                dx = d0x
                dy = d0y
                maxs = min(nfwd[i], M - n - 1)
            x = sx
            y = sy
            cnt = 0
            left = over[i]
            for k in range(maxs):
                x2 = x + dx * st
                y2 = y + dy * st
                if x2 < -margin or y2 < -margin or x2 >= Wf + margin or y2 >= Hf + margin:
                    break
                gx = min(max(int(x2 * gscale), 0), GW - 1)
                gy = min(max(int(y2 * gscale), 0), GH - 1)
                inside = True
                if use_region and region[gy, gx] != reg0:
                    left -= 1
                    if left < 0:
                        break
                    inside = False          # carry on straight for a step past the edge
                if inside:
                    dl = lab[gy, gx, 0] - L0
                    da = lab[gy, gx, 1] - a0
                    db = lab[gy, gx, 2] - b0
                    if dl * dl + da * da + db * db > tol2:
                        break
                    fx = vx[gy, gx]
                    fy = vy[gy, gx]
                    if fx * dx + fy * dy < 0:
                        fx = -fx
                        fy = -fy
                    dx = dx * inertia + fx * (1.0 - inertia)
                    dy = dy * inertia + fy * (1.0 - inertia)
                    dn = math.sqrt(dx * dx + dy * dy) + 1e-9
                    dx /= dn
                    dy /= dn
                x = x2
                y = y2
                if direction == 0:
                    buf[cnt, 0] = x
                    buf[cnt, 1] = y
                else:
                    out[i, n, 0] = x
                    out[i, n, 1] = y
                    n += 1
                cnt += 1
            if direction == 0:
                for k in range(cnt - 1, -1, -1):
                    out[i, n, 0] = buf[k, 0]
                    out[i, n, 1] = buf[k, 1]
                    n += 1
                out[i, n, 0] = sx
                out[i, n, 1] = sy
                n += 1
        nout[i] = n


@njit(parallel=True, cache=True, fastmath=True)
def shade(col, Ht, thk, iri, ifd, lut, lut_max, out, lx, ly, lz, amb, spec, shin,
          pearl, glint, shadow_k, shadow_steps, angle_shift):
    """Light the relief: Lambert + Blinn-Phong gloss + short cast shadows, plus the
    interference medium, whose colour comes from the thin-film LUT and slides along it as
    the paint surface turns away from the highlight direction."""
    H = Ht.shape[0]
    W = Ht.shape[1]
    hx = lx
    hy = ly
    hz = lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    hx /= hn
    hy /= hn
    hz /= hn
    lxy = math.sqrt(lx * lx + ly * ly) + 1e-9
    ldx = lx / lxy
    ldy = ly / lxy
    tanel = lz / lxy
    nl = lut.shape[0]
    lsc = (nl - 1) / lut_max
    for yy in prange(H):
        y = np.int64(yy)  # prange indices are unsigned; keep the arithmetic signed
        ym = y - 1 if y > 0 else y
        yp = y + 1 if y < H - 1 else y
        for x in range(W):
            xm = x - 1 if x > 0 else x
            xp = x + 1 if x < W - 1 else x
            gx = (Ht[y, xp] - Ht[y, xm]) * 0.5
            gy = (Ht[yp, x] - Ht[ym, x]) * 0.5
            inv = 1.0 / math.sqrt(gx * gx + gy * gy + 1.0)
            nx = -gx * inv
            ny = -gy * inv
            nz = inv
            ndl = nx * lx + ny * ly + nz * lz
            if ndl < 0.0:
                ndl = 0.0
            d = ndl / lz
            sh = 1.0
            if shadow_k > 0.0:
                h0 = Ht[y, x]
                occ = 0.0
                for k in range(1, shadow_steps + 1):
                    t = k * 1.4
                    ix = int(x + ldx * t + 0.5)
                    iy = int(y + ldy * t + 0.5)
                    if ix < 0 or iy < 0 or ix >= W or iy >= H:
                        break
                    dh = Ht[iy, ix] - (h0 + t * tanel)
                    if dh > occ:
                        occ = dh
                sh = 1.0 / (1.0 + shadow_k * occ)
            ndh = nx * hx + ny * hy + nz * hz
            if ndh < 0.0:
                ndh = 0.0
            gl = thk[y, x] * 0.6
            if gl > 1.0:
                gl = 1.0
            gl = 0.35 + 0.65 * gl
            sp_ = spec * gl * math.pow(ndh, shin) * sh
            light = amb + (1.0 - amb) * d * sh
            r = col[y, x, 0] * light + sp_
            g = col[y, x, 1] * light + sp_
            b = col[y, x, 2] * light + sp_
            ir = iri[y, x]
            if ir > 0.004:
                fd = ifd[y, x] * (1.0 - angle_shift * (1.0 - ndh))
                fi = fd * lsc
                if fi < 0.0:
                    fi = 0.0
                if fi > nl - 1.001:
                    fi = nl - 1.001
                i0 = int(fi)
                ff = fi - i0
                lobe = ir * (pearl * math.pow(ndh, 14.0) * (0.35 + 0.65 * sh)
                             + glint * math.pow(ndh, 70.0) * sh)
                r += lobe * (lut[i0, 0] + (lut[i0 + 1, 0] - lut[i0, 0]) * ff)
                g += lobe * (lut[i0, 1] + (lut[i0 + 1, 1] - lut[i0, 1]) * ff)
                b += lobe * (lut[i0, 2] + (lut[i0 + 1, 2] - lut[i0, 2]) * ff)
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b


# ---- procedural textures -------------------------------------------------------------------
def value_noise(W, H, cell, rng, octaves=1, gain=0.5):
    """Smooth value noise in [0,1] (bicubic-upsampled random lattice), optionally fractal."""
    acc = np.zeros((H, W), np.float32)
    amp = 1.0
    tot = 0.0
    c = float(cell)
    for _ in range(octaves):
        gw = max(2, int(math.ceil(W / c)) + 3)
        gh = max(2, int(math.ceil(H / c)) + 3)
        g = rng.random((gh, gw)).astype(np.float32)
        big = cv2.resize(g, (int(gw * c), int(gh * c)), interpolation=cv2.INTER_CUBIC)
        ox = int(rng.integers(0, max(1, int(c))))
        oy = int(rng.integers(0, max(1, int(c))))
        acc += amp * big[oy:oy + H, ox:ox + W]
        tot += amp
        amp *= gain
        c = max(1.5, c / 2)
    acc /= tot
    return np.clip(acc, 0, 1)


def make_weave(W, H, pitch, rng):
    """Plain-weave canvas height in [0,1]: warp and weft threads passing over and under."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    wob = pitch * 0.45
    xs = (x + (value_noise(W, H, pitch * 9, rng) - 0.5) * 2 * wob) / pitch
    ys = (y + (value_noise(W, H, pitch * 9, rng) - 0.5) * 2 * wob) / pitch
    i = np.floor(xs)
    j = np.floor(ys)
    cw = np.clip(np.cos(np.pi * (xs - i - 0.5)), 0, 1) ** 0.8
    cf = np.clip(np.cos(np.pi * (ys - j - 0.5)), 0, 1) ** 0.8
    ow = 0.5 + 0.5 * np.cos(np.pi * (ys - 0.5) + np.pi * i)
    of = 0.5 - 0.5 * np.cos(np.pi * (xs - 0.5) + np.pi * j)
    h = np.maximum(cw * (0.25 + 0.75 * ow), cf * (0.25 + 0.75 * of))
    h = h * (0.85 + 0.3 * value_noise(W, H, pitch * 3, rng)) + 0.10 * (rng.random((H, W)).astype(np.float32) - 0.5)
    h -= h.min()
    h /= h.max()
    return h.astype(np.float32)


def _streaks(W, H, cell, stretch, angle, rng):
    """Value noise drawn out along one direction (degrees)."""
    big = int(math.ceil(math.hypot(W, H))) + 8
    n = value_noise(max(8, big // max(1, int(stretch))), big, cell, rng)
    n = cv2.resize(n, (big, big), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((big / 2, big / 2), -angle, 1.0)
    M[0, 2] -= (big - W) / 2
    M[1, 2] -= (big - H) / 2
    return cv2.warpAffine(n, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def make_paper(W, H, rng, grain=4.0, coarse=12.0, broad=40.0, mix=(0.55, 0.30, 0.15), fibres=0.0, fibre_len=28.0):
    """Paper height in [0,1]: soft bumps at three sizes (pixels), mixed; with `fibres` > 0 the
    long fibres of rice paper laid over them in every direction."""
    h = (mix[0] * value_noise(W, H, max(1.5, grain), rng) + mix[1] * value_noise(W, H, max(2.0, coarse), rng)
         + mix[2] * value_noise(W, H, max(3.0, broad), rng))
    if fibres > 0:
        f = np.zeros((H, W), np.float32)
        for a in rng.uniform(0, 180, 4):
            s = _streaks(W, H, max(1.5, grain * 0.7), max(2.0, fibre_len / max(1.0, grain)), a, rng)
            f = np.maximum(f, np.clip((s - 0.62) / 0.2, 0, 1))
        h = (1 - fibres) * h + fibres * (0.5 * h + 0.5 * f)
    h -= h.min()
    h /= max(1e-6, h.max())
    return h.astype(np.float32)


class Strokes:
    """A batch of strokes in CSR form, ready for the kernel."""

    def __init__(self):
        self._pts = []
        self._len = []
        self._sp = []
        self._ci = []
        self._cw = []

    def __len__(self):
        return int(sum(len(x) for x in self._len))

    def add(self, pts, npts, sp, comp_idx, comp_w):
        """pts: (N,M,2) padded polylines with npts (N,) valid counts, or a list of (n,2)."""
        if isinstance(pts, np.ndarray) and pts.ndim == 3:
            npts = np.asarray(npts, np.int64)
            keep = npts >= 2
            if not keep.any():
                return
            pts, npts, sp = pts[keep], npts[keep], sp[keep]
            comp_idx, comp_w = comp_idx[keep], comp_w[keep]
            mask = np.arange(pts.shape[1])[None, :] < npts[:, None]
            self._pts.append(pts[mask].astype(np.float32))
            self._len.append(npts)
        else:
            keep = [k for k, p in enumerate(pts) if len(p) >= 2]
            if not keep:
                return
            self._pts.append(np.concatenate([np.asarray(pts[k], np.float32) for k in keep]))
            self._len.append(np.array([len(pts[k]) for k in keep], np.int64))
            sp, comp_idx, comp_w = sp[keep], comp_idx[keep], comp_w[keep]
        self._sp.append(np.asarray(sp, np.float32))
        self._ci.append(np.asarray(comp_idx, np.int32))
        self._cw.append(np.asarray(comp_w, np.float32))

    def arrays(self):
        ln = np.concatenate(self._len)
        off = np.zeros(len(ln) + 1, np.int64)
        np.cumsum(ln, out=off[1:])
        return (np.concatenate(self._pts), off, np.concatenate(self._sp),
                np.concatenate(self._ci), np.concatenate(self._cw))


class Canvas:
    def __init__(self, W, H, ground, pitch, seed=1, ground_S=0.8, support=None):
        """support: None for a woven canvas, or make_paper() arguments (sizes in pixels) for paper."""
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.col = np.empty((H, W, 3), np.float32)
        self.col[:] = np.asarray(ground, np.float32)
        self.sca = np.full((H, W), ground_S, np.float32)
        self.wet = np.zeros((H, W), np.float32)
        self.hgt = np.zeros((H, W), np.float32)
        self.thk = np.zeros((H, W), np.float32)
        self.iri = np.zeros((H, W), np.float32)
        self.ifd = np.full((H, W), 600.0, np.float32)
        self.stamp = np.zeros((H, W), np.int32)
        if support is None:
            self.weave = make_weave(W, H, pitch, rng)                  # (the name stays: it is the support's relief)
            self.grain = value_noise(W, H, max(2.0, pitch * 0.35), rng)
        else:
            self.weave = make_paper(W, H, rng, **support)
            self.grain = value_noise(W, H, max(1.5, support.get('grain', 4.0) * 0.5), rng)
        self.pitch = pitch
        self.tooth = None
        self.stroke_tape = None
        self.update_tooth()
        self.recorder = None                     # replay.Recorder: sees every batch painted, changes nothing
        self.n_strokes = 0
        self.n_trails = 0
        self.n_pix = 0
        self._stamp_base = 0
        self.lut = C.thinfilm_lut()
        self.lut_max = 1800.0

    def update_tooth(self):
        """What a dragged brush catches on: weave + grain + ridges of paint already there."""
        hp = self.hgt - cv2.GaussianBlur(self.hgt, (0, 0), max(1.5, self.pitch * 0.5))
        hide = np.exp(-self.thk * 0.5)
        t = 0.18 + (0.62 * self.weave + 0.30 * (self.grain - 0.5)) * (0.35 + 0.65 * hide) \
            + 0.30 * (1 - hide) + 0.45 * hp
        self.tooth = np.clip(t, 0, 1).astype(np.float32)
        if self.stroke_tape is not None:
            self.stroke_tape.tooth()

    def paint(self, strokes, order=None, shuffle=True, clip=None):
        if len(strokes) == 0:
            return 0
        pts, off, sp, ci, cw = strokes.arrays()
        n = len(off) - 1
        if order is None:
            order = self.rng.permutation(n) if shuffle else np.arange(n)
        seeds = self.rng.integers(1, 2 ** 31 - 1, n).astype(np.int64)
        return self.deposit(pts, off, sp, ci, cw, np.asarray(order, np.int64), seeds, clip=clip)

    def deposit(self, pts, off, sp, ci, cw, order, seeds, clip=None):
        """Apply an already loaded batch in its physical order, with its original bristle seeds."""
        n = len(off) - 1
        deposited = None
        if clip is not None:
            clip = np.asarray(clip, dtype=bool)
            if clip.shape != (self.H, self.W):
                raise ValueError('Brush selection must match the canvas size')
            deposited = np.zeros(len(order), bool)
        npix, ntrail = paint_strokes(self.col, self.sca, self.wet, self.hgt, self.thk,
                                     self.iri, self.ifd, self.stamp, self.tooth,
                                     pts, off, sp, seeds, ci, cw, C.PIG_K, C.PIG_S,
                                     order, self._stamp_base, clip, deposited)
        visible = order if deposited is None else order[deposited]
        self._stamp_base += n + 1
        self.n_strokes += len(visible)
        self.n_trails += int(ntrail)
        self.n_pix += int(npix)
        if self.recorder is not None and len(visible):
            self.recorder.strokes(pts, off, sp, visible, clip=clip)
        if self.stroke_tape is not None:
            self.stroke_tape.batch(pts, off, sp, ci, cw, order, seeds, clip=clip)
        return len(visible)

    def dry(self, amount=1.0):
        self.wet *= (1.0 - amount)
        if self.stroke_tape is not None:
            self.stroke_tape.dry(amount)

    def rewet(self, mask, amount):
        self.wet[mask] = np.maximum(self.wet[mask], amount)
        if self.stroke_tape is not None:
            self.stroke_tape.rewet(mask, amount)

    def relief(self, relief=1.0, weave_amp=0.5, weave_hide=0.7, blur=0.6):
        Ht = self.hgt * relief + self.weave * weave_amp * np.exp(-self.thk * weave_hide)
        if blur > 0:
            Ht = cv2.GaussianBlur(Ht, (0, 0), blur)
        return Ht.astype(np.float32)

    def render(self, light=(-0.50, -0.62, 0.60), amb=0.42, spec=0.07, shin=50.0,
               pearl=0.55, glint=0.9, shadow=0.55, shadow_steps=10, angle_shift=0.25,
               relief=1.0, weave_amp=0.5, weave_hide=0.7, blur=0.6, lit=True, dither=True):
        """-> uint8 sRGB image (H,W,3)."""
        if lit:
            Ht = self.relief(relief, weave_amp, weave_hide, blur)
            l = np.asarray(light, np.float64)
            l = l / np.linalg.norm(l)
            out = np.empty_like(self.col)
            shade(self.col, Ht, self.thk, self.iri, self.ifd, self.lut, self.lut_max, out,
                  l[0], l[1], l[2], amb, spec, shin, pearl, glint, shadow, shadow_steps,
                  angle_shift)
        else:
            out = self.col.copy()
        # soft shoulder so impasto whites roll off instead of clipping
        knee = 0.82
        hi = out > knee
        out[hi] = knee + (1 - knee) * np.tanh((out[hi] - knee) / (1 - knee))
        s = C.lin_to_srgb(out) * 255.0
        if dither:
            s = s + (np.random.default_rng(7).random(s.shape) - 0.5)
        return np.clip(s + 0.5, 0, 255).astype(np.uint8)
