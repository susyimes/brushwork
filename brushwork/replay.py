"""Recording a painting while it is painted, and playing it back in a web page.

    st = Studio(1200, 1500)
    st.record()                          # from here on every stroke and every stage is kept
    ...                                  # paint as usual
    st.save_replay('process.html', title='Sky Mirror')

or, for a command-line session, `brush.py replay <session> --width 1200 --out final.png
--html process.html` (a session keeps all its instructions, so any session can be replayed).

The page is one self-contained HTML file, nothing to install and nothing fetched:

  keyframes   the canvas as it stood at the end of every stage (WebP, `frame_width` wide)
  strokes     the path and width of every stroke, in the order it was laid (canvas units,
              thinned to what a stroke of that width needs, delta-coded, gzipped)

Played back, the strokes of each stage uncover that stage's keyframe along their own paths, in
their real order, over the keyframe before.  So the motion is the painting's own and every pixel
shown is one the engine painted; nothing is simulated again in the browser.  (Inside one stage
a stroke shows the stage's final paint along its path, so a stroke that a later one in the same
stage covers is seen already covered; at playback speed this does not show, and more stages
mean finer steps.)

Recording only watches: it takes nothing from the random streams, so a painting comes out the
same with or without it.
"""
import base64
import gzip
import json
import os
import struct

import cv2
import numpy as np

from .engine import P_RADIUS

SCALE = 16            # stored point coordinates are canvas units x SCALE (int16)
RSCALE = 64           # stored radii are canvas units x RSCALE (uint16)
MAGIC = b'BWRP'
VERSION = 1


class Recorder:
    def __init__(self, W, H, S, frame_width=1200, quality=85):
        self.W, self.H, self.S = int(W), int(H), float(S)
        self.fw = int(min(frame_width, W))
        self.fh = int(round(self.H * self.fw / self.W))
        self.quality = int(quality)
        self.n = 0
        self.counts, self.radii, self.points = [], [], []
        self.frames = []
        self.clip_ids = {}                       # (display x, y, cropped PNG) -> id
        self.clip_ranges = []                    # [first stroke, exclusive end, clip id]
        self._last_clip = (None, None)            # consecutive brush batches often reuse an eye or hair mask

    # ------------------------------------------------------------------ while painting
    def _clip_id(self, clip):
        mask = np.asarray(clip, dtype=bool)
        if mask.shape != (self.H, self.W):
            raise ValueError('Replay selection must match the canvas dimensions')
        bits = np.packbits(mask).tobytes()
        if self._last_clip[0] == bits:
            return self._last_clip[1]
        alpha = cv2.resize(mask.astype(np.uint8) * 255, (self.fw, self.fh),
                           interpolation=cv2.INTER_AREA)
        x, y, w, h = cv2.boundingRect(alpha)
        tile = np.zeros((max(1, h), max(1, w), 4), np.uint8)
        if w and h:
            tile[:, :, 3] = alpha[y:y + h, x:x + w]
        ok, png = cv2.imencode('.png', tile)
        if not ok:
            raise ValueError('Could not encode the replay selection')
        clip_id = self.clip_ids.setdefault((x, y, png.tobytes()), len(self.clip_ids))
        self._last_clip = (bits, clip_id)
        return clip_id

    def strokes(self, pts, off, sp, order, clip=None):
        """Called by Canvas.paint with the batch it has just laid (pixels) and the order it used."""
        n = len(order)
        if n <= 0:
            return
        if clip is not None:
            clip_id = self._clip_id(clip)
            if self.clip_ranges and self.clip_ranges[-1][1:] == [self.n, clip_id]:
                self.clip_ranges[-1][1] += n
            else:
                self.clip_ranges.append([self.n, self.n + n, clip_id])
        P = np.asarray(pts, np.float64) / self.S                     # canvas units
        lens = np.diff(off).astype(np.int64)
        R = np.asarray(sp[:, P_RADIUS], np.float64) / self.S
        step = np.clip(0.6 * R, 0.8, 8.0)                             # a wide stroke needs few points
        seg = np.zeros(len(P))
        if len(P) > 1:
            seg[1:] = np.hypot(P[1:, 0] - P[:-1, 0], P[1:, 1] - P[:-1, 1])
        cs = np.cumsum(seg)
        starts = off[:-1].astype(np.int64)
        s_local = cs - np.repeat(cs[starts], lens)
        bucket = np.floor(s_local / np.repeat(step, lens))
        keep = np.zeros(len(P), bool)
        keep[starts] = True
        keep[off[1:].astype(np.int64) - 1] = True
        keep[1:] |= bucket[1:] != bucket[:-1]
        kcount = np.add.reduceat(keep.astype(np.int64), starts)
        kpts = P[keep]
        koff = np.concatenate([[0], np.cumsum(kcount)])
        order = np.asarray(order, np.int64)
        lo = kcount[order]
        idx = np.arange(int(lo.sum())) - np.repeat(np.cumsum(lo) - lo, lo) + np.repeat(koff[:-1][order], lo)
        q = np.clip(np.round(kpts[idx] * SCALE), -32768, 32767).astype(np.int32)
        d = q.copy()
        d[1:] -= q[:-1]
        first = np.cumsum(lo) - lo
        d[first] = q[first]                                           # each stroke starts absolute
        self.points.append(np.clip(d, -32768, 32767).astype('<i2'))
        self.counts.append(np.minimum(lo, 65535).astype('<u2'))
        self.radii.append(np.clip(np.round(R[order] * RSCALE), 0, 65535).astype('<u2'))
        self.n += n

    def keyframe(self, name, img, force=False):
        """Keep the canvas as it stands (an RGB image at painting size) as the end of a stage.
        Without new strokes since the last keyframe there is nothing to keep, and the step
        keeps the name it already has."""
        if self.frames and self.frames[-1]['at'] == self.n and not force:
            return
        small = cv2.resize(np.ascontiguousarray(img), (self.fw, self.fh), interpolation=cv2.INTER_AREA)
        data, mime = _encode(small, self.quality)
        self.frames.append(dict(name=str(name), at=self.n, data=data, mime=mime))

    # ------------------------------------------------------------------ the page
    def stroke_data(self):
        """The packed stroke block: header, point counts, radii, points (little-endian)."""
        counts = np.concatenate(self.counts) if self.counts else np.zeros(0, '<u2')
        radii = np.concatenate(self.radii) if self.radii else np.zeros(0, '<u2')
        pts = np.concatenate(self.points) if self.points else np.zeros((0, 2), '<i2')
        head = MAGIC + struct.pack('<III', VERSION, len(counts), len(pts))
        return head + counts.tobytes() + radii.tobytes() + pts.tobytes()

    def export(self, path, title='', sketch=None, info=None):
        if len(self.frames) < 2:
            raise ValueError('nothing was painted while recording')
        raw = self.stroke_data()
        blob = base64.b64encode(gzip.compress(raw, 9)).decode('ascii')
        meta = dict(title=title, width=self.fw, height=self.fh, units=[1000, round(1000.0 * self.H / self.W, 2)],
                    strokes=int(self.n), scale=SCALE, rscale=RSCALE, info=info or {},
                    frames=[dict(name=f['name'], at=f['at']) for f in self.frames])
        if self.clip_ranges:
            meta['clipRanges'] = self.clip_ranges
            meta['clips'] = [dict(x=x, y=y, src='data:image/png;base64,' + base64.b64encode(png).decode('ascii'))
                             for x, y, png in self.clip_ids]
        frames = ['data:%s;base64,%s' % (f['mime'], base64.b64encode(f['data']).decode('ascii')) for f in self.frames]
        sk = ''
        if sketch is not None:
            small = cv2.resize(np.ascontiguousarray(sketch), (self.fw, self.fh), interpolation=cv2.INTER_AREA)
            data, mime = _encode(small, self.quality)
            sk = 'data:%s;base64,%s' % (mime, base64.b64encode(data).decode('ascii'))
        page = (PAGE.replace('/*TITLE*/', _html(title or 'brushwork'))
                .replace('/*META*/', json.dumps(meta, ensure_ascii=False).replace('</', '<\\/'))
                .replace('/*FRAMES*/', json.dumps(frames))
                .replace('/*SKETCH*/', json.dumps(sk))
                .replace('/*STROKES*/', blob))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(page)
        return dict(path=os.path.abspath(path), frames=len(self.frames), strokes=int(self.n),
                    bytes=os.path.getsize(path), stroke_bytes=len(blob))


def _encode(rgb, quality):
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    ok, buf = cv2.imencode('.webp', bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if ok:
        return buf.tobytes(), 'image/webp'
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes(), 'image/jpeg'


def _html(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def read(path):
    """Read a replay page back: (meta, number of keyframes, decoded stroke arrays).  For tests
    and tools; the page itself needs none of this."""
    with open(path, encoding='utf-8') as f:
        page = f.read()

    def block(name):
        a = page.index('<script id="%s"' % name)
        a = page.index('>', a) + 1
        return page[a:page.index('</script>', a)]

    meta = json.loads(block('meta'))
    frames = json.loads(block('frames'))
    raw = gzip.decompress(base64.b64decode(block('strokes')))
    if raw[:4] != MAGIC:
        raise ValueError('not a brushwork replay')
    version, n, npts = struct.unpack('<III', raw[4:16])
    counts = np.frombuffer(raw, '<u2', n, 16)
    radii = np.frombuffer(raw, '<u2', n, 16 + 2 * n)
    pts = np.frombuffer(raw, '<i2', 2 * npts, 16 + 4 * n).reshape(-1, 2)
    return meta, len(frames), dict(counts=counts, radii=radii, points=pts)


PAGE = r'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>/*TITLE*/ · 作画过程</title>
<style>
  :root { --bg: #141518; --panel: #1c1e23; --raise: #262930; --ink: #ebe8e1; --dim: #8f8c85; --accent: #e0b56e; }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--ink);
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  .app { display: grid; grid-template-columns: minmax(0, 1fr) 300px; grid-template-rows: 100%; gap: 16px; height: 100%; padding: 16px; }
  .main { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  header { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; padding-bottom: 10px; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: .02em; }
  .now { color: var(--accent); font-size: 13px; text-align: right; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .view { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
  canvas { display: block; max-width: 100%; max-height: 100%; width: auto; height: auto; background: #000;
    box-shadow: 0 10px 40px rgba(0, 0, 0, .5); }
  .bar { display: flex; align-items: center; gap: 10px; padding-top: 12px; }
  button, select { background: var(--raise); color: var(--ink); border: 0; border-radius: 6px; height: 32px; padding: 0 12px;
    font: inherit; cursor: pointer; white-space: nowrap; }
  button:hover, select:hover { background: #30343c; }
  button:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #play { width: 44px; padding: 0; font-size: 15px; }
  input[type=range] { flex: 1; min-width: 60px; accent-color: var(--accent); }
  .count { color: var(--dim); font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 9em; text-align: right; }
  aside { background: var(--panel); border-radius: 10px; padding: 10px; overflow: auto; min-height: 0; }
  aside h2 { margin: 2px 6px 8px; font-size: 13px; font-weight: 600; color: var(--dim); }
  .step { display: grid; grid-template-columns: 1.8em 1fr auto; gap: 6px; align-items: baseline; width: 100%; height: auto;
    padding: 6px 8px; border-radius: 6px; background: none; color: var(--dim); text-align: left; }
  .step .n { color: #5f5c56; font-variant-numeric: tabular-nums; }
  .step .k { font-size: 12px; font-variant-numeric: tabular-nums; }
  .step.done { color: var(--ink); }
  .step.now { background: var(--raise); color: var(--accent); }
  .note { color: var(--dim); font-size: 12px; margin: 10px 6px 2px; }
  @media (max-width: 820px) {
    .app { display: block; height: auto; padding: 12px; }
    .view { display: block; }
    canvas { width: 100%; max-height: 78vh; object-fit: contain; }
    .bar { flex-wrap: wrap; }
    input[type=range] { order: -1; flex-basis: 100%; }
    .count { flex: 1; min-width: 0; text-align: left; }
    aside { margin-top: 12px; max-height: 45vh; }
  }
</style>
</head>
<body>
<div class="app">
  <div class="main">
    <header><h1 id="title"></h1><div class="now" id="now"></div></header>
    <div class="view"><canvas id="cv" aria-label="作画过程"></canvas></div>
    <div class="bar">
      <button id="play" aria-label="播放">&#9654;</button>
      <input id="seek" type="range" min="0" max="10000" step="1" value="10000" aria-label="进度">
      <span class="count" id="count"></span>
      <select id="speed" aria-label="速度">
        <option value="0.5">0.5×</option><option value="1" selected>1×</option>
        <option value="2">2×</option><option value="4">4×</option>
      </select>
      <button id="sketch" title="按住看底稿">底稿</button>
    </div>
  </div>
  <aside><h2>作画步骤</h2><div id="steps"></div><p class="note" id="note"></p></aside>
</div>
<script id="meta" type="application/json">/*META*/</script>
<script id="frames" type="application/json">/*FRAMES*/</script>
<script id="sketchimg" type="application/json">/*SKETCH*/</script>
<script id="strokes" type="application/octet-stream">/*STROKES*/</script>
<script>
(async function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const meta = JSON.parse($('meta').textContent);
  const frameSrc = JSON.parse($('frames').textContent);
  const sketchSrc = JSON.parse($('sketchimg').textContent);
  const fmt = (v) => v.toLocaleString('zh-CN');
  $('title').textContent = meta.title || 'brushwork';

  const cv = $('cv'), ctx = cv.getContext('2d');
  cv.width = meta.width; cv.height = meta.height;
  const load = (src) => new Promise((ok, bad) => { const im = new Image(); im.onload = () => ok(im); im.onerror = bad; im.src = src; });
  const imgs = await Promise.all(frameSrc.map(load));
  const clips = await Promise.all((meta.clips || []).map(async ({x, y, src}) => ({x, y, image: await load(src)})));
  const clipFor = clips.length ? new Uint32Array(meta.strokes) : null;
  for (const [from, to, id] of meta.clipRanges || []) clipFor.fill(id + 1, from, to);
  const sketch = sketchSrc ? await load(sketchSrc) : null;
  if (!sketch) $('sketch').hidden = true;

  // ---- the strokes: decode once into display pixels ---------------------------------------
  let xy = null, start = null, width = null, nStrokes = 0;
  try {
    const bin = Uint8Array.from(atob($('strokes').textContent.trim()), (c) => c.charCodeAt(0));
    const buf = await new Response(new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    const dv = new DataView(buf);
    if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'BWRP') throw new Error('bad data');
    const n = dv.getUint32(8, true), np = dv.getUint32(12, true);
    const counts = new Uint16Array(buf, 16, n), radii = new Uint16Array(buf, 16 + 2 * n, n);
    const pts = new Int16Array(buf, 16 + 4 * n, 2 * np);
    const k = meta.width / 1000 / meta.scale;
    xy = new Float32Array(2 * np); start = new Uint32Array(n + 1); width = new Float32Array(n);
    let j = 0;
    for (let i = 0; i < n; i++) {
      start[i] = j;
      let x = 0, y = 0;
      for (let c = 0; c < counts[i]; c++, j++) {
        if (c === 0) { x = pts[2 * j]; y = pts[2 * j + 1]; } else { x += pts[2 * j]; y += pts[2 * j + 1]; }
        xy[2 * j] = x * k; xy[2 * j + 1] = y * k;
      }
      width[i] = Math.max(1, 2.3 * radii[i] / meta.rscale * meta.width / 1000);
    }
    start[n] = j; nStrokes = n;
  } catch (e) {
    xy = null;
    $('note').textContent = '这个浏览器不能解压笔触数据，只能在步骤之间渐变播放。';
  }

  // ---- the timeline: one step per keyframe after the first ----------------------------------
  const F = meta.frames, steps = [];
  let T = 0;
  for (let s = 1; s < F.length; s++) {
    const a = F[s - 1].at, b = F[s].at, cnt = b - a;
    const d = cnt > 0 ? Math.min(7, 0.8 + 0.032 * Math.sqrt(cnt)) : 0.6;
    steps.push({ s, a, b, cnt, t0: T, d, name: F[s].name });
    T += d;
  }
  const HOLD = 1.2;
  T += HOLD;
  const total = meta.strokes;

  // ---- drawing -----------------------------------------------------------------------------
  const pats = imgs.map((im) => ctx.createPattern(im, 'no-repeat'));
  const clippedPatterns = new Map();
  let patternStep = -1;
  function strokePattern(step, id) {
    if (!id) return pats[step];
    if (patternStep !== step) {
      clippedPatterns.clear();
      patternStep = step;
    }
    if (!clippedPatterns.has(id)) {
      const {x, y, image} = clips[id - 1];
      const tile = document.createElement('canvas');
      tile.width = image.width; tile.height = image.height;
      const tileCtx = tile.getContext('2d');
      tileCtx.drawImage(image, 0, 0);
      tileCtx.globalCompositeOperation = 'source-in';
      tileCtx.drawImage(imgs[step], -x, -y);
      const pattern = ctx.createPattern(tile, 'no-repeat');
      pattern.setTransform(new DOMMatrix().translate(x, y));
      clippedPatterns.set(id, pattern);
    }
    return clippedPatterns.get(id);
  }
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  let shown = null, shownCount = 0;            // which step is on the canvas, and how many of its strokes

  function strokesOf(st, from, to) {
    ctx.strokeStyle = pats[st.s];
    for (let i = st.a + from; i < st.a + to; i++) {
      if (clipFor) ctx.strokeStyle = strokePattern(st.s, clipFor[i]);
      const p0 = start[i], p1 = start[i + 1];
      if (p1 - p0 < 1) continue;
      ctx.lineWidth = width[i];
      ctx.beginPath();
      ctx.moveTo(xy[2 * p0], xy[2 * p0 + 1]);
      if (p1 - p0 === 1) ctx.lineTo(xy[2 * p0] + 0.01, xy[2 * p0 + 1]);
      for (let p = p0 + 1; p < p1; p++) ctx.lineTo(xy[2 * p], xy[2 * p + 1]);
      ctx.stroke();
    }
  }

  function stepAt(t) {
    for (const st of steps) if (t < st.t0 + st.d) return st;
    return null;
  }

  function show(t) {
    const st = stepAt(t);
    if (!st) {
      if (shown !== 'end') { ctx.globalAlpha = 1; ctx.drawImage(imgs[imgs.length - 1], 0, 0); shown = 'end'; }
      return { index: steps.length, done: total };
    }
    const f = Math.min(1, Math.max(0, (t - st.t0) / st.d));
    if (st.cnt === 0 || !xy) {                  // no strokes to follow: fade from one keyframe to the next
      ctx.globalAlpha = 1; ctx.drawImage(imgs[st.s - 1], 0, 0);
      ctx.globalAlpha = f; ctx.drawImage(imgs[st.s], 0, 0); ctx.globalAlpha = 1;
      shown = null;
      return { index: st.s - 1, done: st.a + Math.floor(f * st.cnt), step: st };
    }
    const c = Math.min(st.cnt, Math.floor(f * st.cnt));
    if (shown !== st.s || c < shownCount) {
      ctx.globalAlpha = 1; ctx.drawImage(imgs[st.s - 1], 0, 0);
      shown = st.s; shownCount = 0;
    }
    if (c > shownCount) { strokesOf(st, shownCount, c); shownCount = c; }
    return { index: st.s - 1, done: st.a + c, step: st };
  }

  // ---- the controls ------------------------------------------------------------------------
  const list = $('steps');
  steps.forEach((st, i) => {
    const b = document.createElement('button');
    b.className = 'step';
    b.innerHTML = '<span class="n"></span><span class="t"></span><span class="k"></span>';
    b.querySelector('.n').textContent = i + 1;
    b.querySelector('.t').textContent = st.name;
    b.querySelector('.k').textContent = st.cnt ? fmt(st.cnt) : '';
    b.title = st.cnt ? st.name + '：' + fmt(st.cnt) + ' 笔' : st.name;
    b.addEventListener('click', () => { pause(); seekTo(st.t0 + st.d - 1e-6); });
    list.appendChild(b);
  });
  const items = Array.from(list.children);

  let t = T, playing = false, last = 0, speed = 1, lastIndex = -2;
  function update(r) {
    $('seek').value = Math.round(t / T * 10000);
    $('count').textContent = fmt(r.done) + ' / ' + fmt(total) + ' 笔';
    $('now').textContent = r.step ? ('第 ' + (r.index + 1) + ' / ' + steps.length + ' 步 · ' + r.step.name) : '完成';
    if (r.index !== lastIndex) {
      items.forEach((el, i) => { el.classList.toggle('done', i < r.index); el.classList.toggle('now', i === r.index); });
      if (items[r.index] && playing) items[r.index].scrollIntoView({ block: 'nearest' });
      lastIndex = r.index;
    }
  }
  function seekTo(v) { t = Math.min(T, Math.max(0, v)); update(show(t)); }
  function frame(now) {
    if (!playing) return;
    t = Math.min(T, t + (now - last) / 1000 * speed); last = now;
    update(show(t));
    if (t >= T) { pause(); return; }
    requestAnimationFrame(frame);
  }
  function play() {
    if (t >= T) { t = 0; shown = null; }
    playing = true; last = performance.now();
    $('play').innerHTML = '&#10073;&#10073;'; $('play').setAttribute('aria-label', '暂停');
    requestAnimationFrame(frame);
  }
  function pause() { playing = false; $('play').innerHTML = '&#9654;'; $('play').setAttribute('aria-label', '播放'); }

  $('play').addEventListener('click', () => (playing ? pause() : play()));
  $('seek').addEventListener('input', (e) => { pause(); seekTo(e.target.value / 10000 * T); });
  $('speed').addEventListener('change', (e) => { speed = parseFloat(e.target.value); });
  const peek = (on) => { if (!sketch) return; if (on) { ctx.drawImage(sketch, 0, 0); } else { shown = null; update(show(t)); } };
  $('sketch').addEventListener('pointerdown', () => peek(true));
  ['pointerup', 'pointerleave', 'pointercancel'].forEach((ev) => $('sketch').addEventListener(ev, () => peek(false)));
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT') return;
    const idx = Math.max(0, steps.findIndex((st) => t < st.t0 + st.d));
    if (e.code === 'Space') { e.preventDefault(); playing ? pause() : play(); }
    else if (e.code === 'ArrowRight') { pause(); const st = steps[Math.min(steps.length - 1, t >= T ? steps.length - 1 : idx)]; seekTo(st.t0 + st.d + 1e-6); }
    else if (e.code === 'ArrowLeft') { pause(); const st = steps[Math.max(0, (t >= T ? steps.length : idx) - 1)]; seekTo(st.t0 + st.d - 1e-6); }
  });
  if (meta.info && meta.info.medium) $('note').textContent += (meta.info.medium ? '画材：' + meta.info.medium + '。' : '');
  update(show(t));
  window.__replay = { T, steps: steps.length, strokes: nStrokes, seek: seekTo, play, pause, state: () => ({ t, playing }) };
})();
</script>
</body>
</html>
'''
