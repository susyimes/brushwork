"""Self-check for brushwork.  Takes about a minute.

    python tests/check.py

1. Every sketch and painting operation runs (tests/all_ops.json uses each one at least once).
2. A painting saved and reloaded between steps comes out identical to one painted in a single run.
3. The session command line works end to end: new, do, look, undo, replay.
4. The Astra example still paints exactly the picture it painted before the toolkit was made
   general (pixel hash at 800 px wide; the reference hash is from the machine it was made on).
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'examples', 'astra_iridescent_black'))

import numpy as np

from brushwork import Studio, run_ops
from brushwork import cli

ASTRA_800 = '1119f1ea3d3575aaa886731072ba3de2e02b44ed1c1eed35a250b3a9b2fb8fcc'


def pixels(st):
    return hashlib.sha256(np.ascontiguousarray(st.cv.render(dither=False)).tobytes()).hexdigest()


def seascape_ops():
    d = os.path.join(ROOT, 'examples', 'seascape')
    return [json.load(open(os.path.join(d, f), encoding='utf-8')) for f in
            ('01_sketch.json', '02_paint.json', '03_accents.json')]


def quiet(*a):
    pass


def check_persistence(tmp):
    batches = seascape_ops()
    a = Studio(600, 375, seed=2, log=quiet)
    for b in batches:
        run_ops(a, b)
    d = os.path.join(tmp, 'state')
    b_ = Studio(600, 375, seed=2, log=quiet)
    for b in batches:
        run_ops(b_, b)
        b_.save(d)
        b_ = Studio.load(d, log=quiet)
    return pixels(a) == pixels(b_), '%d strokes' % a.cv.n_strokes


def check_cli(tmp):
    s = os.path.join(tmp, 'session')
    d = os.path.join(ROOT, 'examples', 'seascape')
    out = os.path.join(tmp, 'replay.png')

    def run(*argv):
        old, sys.stdout = sys.stdout, open(os.devnull, 'w')
        try:
            return cli.main(list(argv))
        finally:
            sys.stdout.close()
            sys.stdout = old

    codes = [run('new', s, '--width', '500', '--height', '312', '--seed', '2')]
    for f in ('01_sketch.json', '02_paint.json', '03_accents.json'):
        codes.append(run('do', s, os.path.join(d, f)))
    codes.append(run('look', s, '--crop', '560', '250', '200', '150', '--zoom', '2'))
    codes.append(run('undo', s))
    codes.append(run('replay', s, '--width', '400', '--out', out))
    bad = run('do', s, os.path.join(d, 'no_such_file.json'))
    return all(c == 0 for c in codes) and bad == 1 and os.path.getsize(out) > 10000, 'exit codes %s, bad input -> %d' % (codes, bad)


def check_all_ops(tmp):
    """tests/all_ops.json touches every operation and every kind of argument once."""
    ops = json.load(open(os.path.join(ROOT, 'tests', 'all_ops.json'), encoding='utf-8'))
    st = Studio(400, seed=4, log=quiet)
    res = run_ops(st, ops)
    for what in ('canvas', 'sketch', 'both', 'error'):
        st.look(os.path.join(tmp, what + '.jpg'), what=what, width=300)
    st.look(os.path.join(tmp, 'crop.jpg'), crop=(300, 250, 300, 250), zoom=2, lit=False)
    used = {r['op'] for r in res}
    missing = (set(cli.SKETCH_OPS) | set(cli.STUDIO_OPS)) - used - {'look'}
    return not missing and st.cv.n_strokes > 1000, '%d operations, %d strokes%s' % (
        len(res), st.cv.n_strokes, (', never exercised: %s' % sorted(missing)) if missing else '')


def check_astra():
    from paint import AstraPainter
    p = AstraPainter(800, log=quiet)
    p.run()
    h = hashlib.sha256(np.ascontiguousarray(p.cv.render()).tobytes()).hexdigest()      # as saved: dithered
    return h == ASTRA_800, '%d strokes, hash %s...' % (p.cv.n_strokes, h[:16])


def main():
    tmp = tempfile.mkdtemp(prefix='brushwork_check_')
    ok = True
    try:
        for name, fn in (('every operation runs', lambda: check_all_ops(tmp)),
                         ('save/load is exact', lambda: check_persistence(tmp)),
                         ('session command line', lambda: check_cli(tmp)),
                         ('Astra example unchanged', check_astra)):
            good, note = fn()
            ok &= good
            print('%-28s %s   (%s)' % (name, 'PASS' if good else 'FAIL', note))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
