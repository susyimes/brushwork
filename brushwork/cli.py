"""Painting step by step from the command line: for agents that would rather send
instructions than write a script.

A session is a directory that holds the whole state of a painting between calls.

    brush.py new     <session> [--width 1200] [--height 1500] [--seed 1] [--medium oil] [--primer "#e2dbcc"]
                     [--palette full]
    brush.py do      <session> <ops.json | ->  [--look view.jpg]
    brush.py look    <session> [--out view.jpg] [--crop X Y W H] [--zoom 2] [--what canvas|sketch|both|error] [--unlit]
    brush.py undo    <session>
    brush.py info    <session>
    brush.py replay  <session> --width 3200 --out final.png [--html process.html]
    brush.py media   [name]          # the painting media: oil, gouache, watercolor, pastel ...
    brush.py palette
    brush.py crop    <image> X Y W H [--zoom 2] [--out crop.png]
    brush.py install-skill [--dest ~/.agents/skills/brushwork-painting]

`do` takes a JSON list of operations, each {"op": "<name>", ...arguments}.  The names and
arguments are exactly the methods of Sketch (drawing the underdrawing) and Studio (painting);
see skill/SKILL.md.  Every call prints one JSON object describing what happened.

Sessions are resolution-independent: work at 1000-1200 pixels wide, then `replay` the same
instructions at 3200 for the finished picture.
"""
import argparse
import json
import os
import shutil
import sys
import time

import cv2
import numpy as np

from . import color as C
from . import media as MEDIA
from .studio import PASSES, SIZES, STYLE, Studio, _imwrite

SKETCH_OPS = ('background', 'polygon', 'spline', 'ellipse', 'rect', 'ribbon', 'tint', 'form_shadow', 'form_light',
              'cast_shadow', 'line', 'direction', 'detail', 'sheen_field', 'region_props')
STUDIO_OPS = ('tone_ground', 'underpaint', 'block_in', 'cut_in', 'mend', 'refine', 'soften', 'strokes', 'dabs',
              'glaze', 'sheen', 'dry', 'paint', 'look')
STATE_FILES = ('meta.json', 'canvas.npz', 'sketch.npz')
UNDO_DEPTH = 3


def run_ops(st, ops, skip=()):
    """Apply a list of {"op": name, ...} to a Studio.  Returns one result per operation."""
    results = []
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or 'op' not in op:
            raise ValueError('operation #%d must be an object with an "op" field' % i)
        args = dict(op)
        name = args.pop('op')
        if name in skip:
            results.append(dict(op=name, skipped=True))
            continue
        if name in SKETCH_OPS:
            fn = getattr(st.sk, name)
        elif name in STUDIO_OPS:
            fn = getattr(st, name)
        else:
            raise ValueError('operation #%d: unknown op %r.  Sketch ops: %s.  Painting ops: %s.'
                             % (i, name, ', '.join(SKETCH_OPS), ', '.join(STUDIO_OPS)))
        before = st.cv.n_strokes
        try:
            r = fn(**args)
        except TypeError as e:
            raise ValueError('operation #%d (%s): %s' % (i, name, e))
        except KeyError as e:
            raise ValueError('operation #%d (%s): %s' % (i, name, e.args[0] if e.args else e))
        res = dict(op=name)
        if st.cv.n_strokes != before:
            res['strokes'] = int(st.cv.n_strokes - before)
        if isinstance(r, str):
            res['path'] = r
        results.append(res)
    return results


# ---------------------------------------------------------------------------------- sessions
def _files(d):
    return [os.path.join(d, f) for f in STATE_FILES]


def _rotate_undo(session):
    """Move the current state into undo/1, shifting older states down."""
    u = os.path.join(session, 'undo')
    last = os.path.join(u, str(UNDO_DEPTH))
    if os.path.isdir(last):
        shutil.rmtree(last)
    for k in range(UNDO_DEPTH - 1, 0, -1):
        a = os.path.join(u, str(k))
        if os.path.isdir(a):
            os.replace(a, os.path.join(u, str(k + 1)))
    first = os.path.join(u, '1')
    os.makedirs(first, exist_ok=True)
    for f in STATE_FILES:
        os.replace(os.path.join(session, f), os.path.join(first, f))


def _read_ops_log(session):
    p = os.path.join(session, 'ops.jsonl')
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_ops_log(session, batches):
    with open(os.path.join(session, 'ops.jsonl'), 'w', encoding='utf-8') as f:
        for b in batches:
            f.write(json.dumps(b, ensure_ascii=False) + '\n')


def _quiet(*a):
    pass


def cmd_new(a):
    if os.path.exists(os.path.join(a.session, 'meta.json')):
        raise ValueError('a session already exists at %s' % a.session)
    medium = MEDIA.load(a.medium)                   # fail before anything is written
    palette = None if a.palette is None else (a.palette[0] if len(a.palette) == 1 else a.palette)
    C.palette_tubes(palette or medium.palette or 'full')
    os.makedirs(os.path.join(a.session, 'views'), exist_ok=True)
    st = Studio(a.width, a.height, seed=a.seed, primer=a.primer, log=_quiet, palette=palette, medium=medium)
    st.save(a.session)
    _write_ops_log(a.session, [])
    return dict(session=os.path.abspath(a.session), width=st.W, height=st.H, units=[1000, round(st.HU, 1)],
                medium=st.medium.name, primer=st.primer, palette=st.palette, tubes=list(st.book.tubes))


def cmd_do(a):
    raw = sys.stdin.read() if a.ops == '-' else open(a.ops, encoding='utf-8-sig').read()
    ops = json.loads(raw)
    if isinstance(ops, dict):
        ops = [ops]
    st = Studio.load(a.session, log=_quiet)
    st.view_dir = os.path.join(a.session, 'views')
    t0 = time.time()
    before = st.cv.n_strokes
    results = run_ops(st, ops)
    out = dict(operations=results, strokes=int(st.cv.n_strokes - before), total_strokes=int(st.cv.n_strokes),
               seconds=round(time.time() - t0, 1))
    if a.look:
        out['look'] = os.path.abspath(st.look(a.look))
    _rotate_undo(a.session)
    st.save(a.session)
    _write_ops_log(a.session, _read_ops_log(a.session) + [ops])
    return out


def cmd_look(a):
    st = Studio.load(a.session, log=_quiet)
    st.view_dir = os.path.join(a.session, 'views')
    p = st.look(a.out, width=a.width, crop=a.crop, zoom=a.zoom, lit=not a.unlit, what=a.what)
    with open(os.path.join(a.session, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    meta['looks'] = st._looks                       # so the next automatic name does not overwrite this one
    with open(os.path.join(a.session, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f)
    return dict(path=os.path.abspath(p))


def cmd_undo(a):
    first = os.path.join(a.session, 'undo', '1')
    if not os.path.isdir(first):
        raise ValueError('nothing to undo')
    for f in STATE_FILES:
        os.replace(os.path.join(first, f), os.path.join(a.session, f))
    os.rmdir(first)
    for k in range(2, UNDO_DEPTH + 1):
        d = os.path.join(a.session, 'undo', str(k))
        if os.path.isdir(d):
            os.replace(d, os.path.join(a.session, 'undo', str(k - 1)))
    batches = _read_ops_log(a.session)
    undone = batches[-1] if batches else []
    _write_ops_log(a.session, batches[:-1])
    return dict(undone=[op.get('op') for op in undone], batches_left=max(0, len(batches) - 1))


def cmd_info(a):
    with open(os.path.join(a.session, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    z = np.load(os.path.join(a.session, 'sketch.npz'), allow_pickle=False)
    sk = json.loads(str(z['meta']))
    batches = _read_ops_log(a.session)
    undo = sum(os.path.isdir(os.path.join(a.session, 'undo', str(k))) for k in range(1, UNDO_DEPTH + 1))
    return dict(session=os.path.abspath(a.session), width=meta['width'], height=meta['height'],
                units=[1000, round(1000.0 * meta['height'] / meta['width'], 1)], seed=meta['seed'],
                medium=MEDIA.load(meta.get('medium', 'oil')).name, palette=meta.get('palette', 'classic'),
                regions=sorted(sk['regions'], key=lambda n: sk['regions'][n]['id']),
                masks=[k[5:] for k in z.files if k.startswith('mask:')],
                batches=len(batches), operations=sum(len(b) for b in batches),
                brushstrokes=meta['n_strokes'], bristle_trails=meta['n_trails'], undo_available=undo)


def cmd_replay(a):
    with open(os.path.join(a.session, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    W = a.width
    H = int(round(W * meta['height'] / meta['width']))
    st = Studio(W, H, seed=meta['seed'], primer=meta['primer'], log=_quiet, palette=meta.get('palette', 'classic'),
                medium=meta.get('medium', 'oil'))
    t0 = time.time()
    if a.html:
        st.record(frame_width=a.frame_width)
    for k, b in enumerate(_read_ops_log(a.session)):
        run_ops(st, b, skip=('look',))
        st.keyframe('do %d: %s' % (k + 1, ', '.join(op.get('op', '?') for op in b)))
    p = st.save_image(a.out)
    out = dict(path=os.path.abspath(p), width=W, height=H, brushstrokes=int(st.cv.n_strokes),
               bristle_trails=int(st.cv.n_trails), seconds=round(time.time() - t0, 1))
    if a.html:
        r = st.save_replay(a.html, title=a.title or os.path.basename(os.path.abspath(a.session)))
        out['replay'] = dict(path=r['path'], steps=r['frames'] - 1, megabytes=round(r['bytes'] / 1e6, 1))
    return out


def cmd_media(a):
    if a.name:
        m = MEDIA.load(a.name)
        return dict(medium=m.describe(), config=m.cfg,
                    note='copy this config to a .json file, change what you like, and pass its path as the medium')
    return dict(media=[MEDIA.load(n).describe() for n in MEDIA.names()],
                note='choose with `new --medium NAME` (default oil); `media NAME` shows one in full; a medium '
                     'can also be your own JSON file with the same keys (see brushwork/media.py)')


def cmd_palette(a):
    return dict(pigments={n: '#%02x%02x%02x' % rgb for n, rgb, _ in C.PIGMENT_DEFS},
                media={n: MEDIA.load(n).title for n in MEDIA.names()},
                palettes={k: list(v) for k, v in C.PALETTES.items()},
                note='colours are mixed from the palette chosen at `new` (default full: every colour, no black, so '
                     'darks are mixed). Choose with_black, or a list such as white ivory_black, for neutral black '
                     'and limited palettes; any pigment can always be named outright in a stroke.',
                brush_sizes=dict(SIZES, note='radius in canvas units; any brush argument takes these names'),
                chromatic_blacks=C.CHROMATIC_BLACKS, styles={k: v for k, v in STYLE.items()},
                sheen_hues=C.SHEEN_HUES, refine_passes=PASSES)


def cmd_crop(a):
    img = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('cannot read image %s' % a.image)
    x, y, w, h = a.box
    c = img[y:y + h, x:x + w]
    if a.zoom != 1.0:
        c = cv2.resize(c, None, fx=a.zoom, fy=a.zoom,
                       interpolation=cv2.INTER_AREA if a.zoom < 1 else cv2.INTER_CUBIC)
    out = a.out or os.path.splitext(a.image)[0] + '_crop.png'
    return dict(path=os.path.abspath(_imwrite(out, c[:, :, ::-1])), size=[c.shape[1], c.shape[0]])


SKILL_ROOT = '<工程目录>'                      # how skill/SKILL.md refers to this checkout


def cmd_install_skill(a):
    """Put the agent guide where agents look for skills, with this checkout's real path filled in."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'skill', 'SKILL.md'), encoding='utf-8', newline='') as f:
        text = f.read()
    head, mark, rest = text.partition('<!-- repo-only -->')
    if mark:                                    # a note that only makes sense inside the repository
        text = head + rest.partition('<!-- /repo-only -->')[2].lstrip('\r\n')
    text = text.replace(SKILL_ROOT + '/', root + os.sep).replace(SKILL_ROOT, root)
    dest = os.path.abspath(os.path.expanduser(a.dest))
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, 'SKILL.md')
    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    return dict(path=path, project=root)


def main(argv=None):
    ap = argparse.ArgumentParser(prog='brush.py', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('new', help='start a session')
    p.add_argument('session')
    p.add_argument('--width', type=int, default=1200)
    p.add_argument('--height', type=int, default=None)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--medium', default='oil',
                   help='oil (default), acrylic, gouache, watercolor, pastel, charcoal, ink, or a medium .json file')
    p.add_argument('--primer', default=None, help="colour of the bare support (default: the medium's)")
    p.add_argument('--palette', nargs='+', default=None, metavar='NAME',
                   help="full, classic, with_black, or a list of pigments (default: the medium's, else full)")
    p.set_defaults(fn=cmd_new)
    p = sub.add_parser('do', help='apply a JSON list of operations')
    p.add_argument('session')
    p.add_argument('ops', help='path to a JSON file, or - for stdin')
    p.add_argument('--look', default=None, help='also save a view of the whole canvas here')
    p.set_defaults(fn=cmd_do)
    p = sub.add_parser('look', help='save a view of the canvas')
    p.add_argument('session')
    p.add_argument('--out', default=None)
    p.add_argument('--width', type=int, default=1200)
    p.add_argument('--crop', type=float, nargs=4, metavar=('X', 'Y', 'W', 'H'), default=None, help='in canvas units')
    p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--what', default='canvas', choices=('canvas', 'sketch', 'both', 'error'))
    p.add_argument('--unlit', action='store_true')
    p.set_defaults(fn=cmd_look)
    p = sub.add_parser('undo', help='take back the last `do`')
    p.add_argument('session')
    p.set_defaults(fn=cmd_undo)
    p = sub.add_parser('info', help='describe a session')
    p.add_argument('session')
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser('replay', help='repaint the whole session at another size')
    p.add_argument('session')
    p.add_argument('--width', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--html', default=None, help='also write a web page that plays the painting back')
    p.add_argument('--frame-width', type=int, default=1200, help='width of the replay page\'s pictures')
    p.add_argument('--title', default=None, help='title shown on the replay page')
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser('media', help='list the painting media, or show one in full')
    p.add_argument('name', nargs='?', default=None)
    p.set_defaults(fn=cmd_media)
    p = sub.add_parser('palette', help='list pigments, brush styles and sheen hues')
    p.set_defaults(fn=cmd_palette)
    p = sub.add_parser('crop', help='cut a detail out of any image file (pixels)')
    p.add_argument('image')
    p.add_argument('box', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    p.add_argument('--zoom', type=float, default=1.0)
    p.add_argument('--out', default=None)
    p.set_defaults(fn=cmd_crop)
    p = sub.add_parser('install-skill', help='install skill/SKILL.md for agents, with this checkout\'s path filled in')
    p.add_argument('--dest', default=os.path.join('~', '.agents', 'skills', 'brushwork-painting'))
    p.set_defaults(fn=cmd_install_skill)
    a = ap.parse_args(argv)
    try:
        out = dict(ok=True, **a.fn(a))
        code = 0
    except Exception as e:                      # one JSON line either way, so a caller can always parse it
        out = dict(ok=False, error='%s: %s' % (type(e).__name__, e))
        code = 1
    sys.stdout.write(json.dumps(out, ensure_ascii=False, default=lambda o: list(o)) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
