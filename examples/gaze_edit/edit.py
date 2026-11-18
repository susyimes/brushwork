"""Move the gaze of 《星落之前》 toward the star by the fingertip.

Run `python examples/gaze_edit/edit.py` from the project root.
The bundled lossless WebP has the exact RGB pixels of out/星落之前-水粉.png,
originally painted by works/fingertip-starlight/paint.py at 2560 x 1888.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from brushwork import ObjectEditor
from brushwork.inspection import inspect_edit
from brushwork.studio import _imwrite
from starlight import open_portrait


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    editor, original, eyes = open_portrait()
    # The farther eye is foreshortened, so the same glance travels fewer pixels.
    shifts = {'far_iris': (-39, 10), 'near_iris': (-47, 12)}
    for name, (dx, dy) in shifts.items():
        editor.move(name, dx, dy)
    after = editor.render()
    doc = out / 'starlight-eyes.npz'
    editor.save(doc)
    reopened = ObjectEditor.load(doc)
    if not np.array_equal(reopened.render(), after):
        raise RuntimeError('Reopening the object document changed the painting')
    for name, (dx, dy) in shifts.items():
        reopened.move(name, -dx, -dy)
    if not np.array_equal(reopened.render(), original):
        raise RuntimeError('Reversing the gaze did not restore the original')
    _imwrite(out / 'before.png', original)
    _imwrite(out / 'after.png', after)
    report = inspect_edit(original, after, mask=eyes, padding=38, path=out/'comparison.png')
    if report.metadata['changed_outside_selection'] or not report.metadata['changed_pixels']:
        raise RuntimeError('Expected a visible edit confined to the two eye openings')
    data = dict(report.metadata, offsets=shifts, original_reconstructed_exactly=True,
                saved_edit_reopened_exactly=True, reverse_move_restored_original=True)
    (out / 'inspection.json').write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'out'/'gaze-edit')
    main(parser.parse_args().out)
