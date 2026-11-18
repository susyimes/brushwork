"""Agent commands for persistent paint objects and local before/after inspections."""
import json
from pathlib import Path
import sys

from .editor import ObjectEditor
from .inspection import changed_bounds, inspect_edit
from .studio import Studio, _imwrite


def read_ops(source):
    raw = sys.stdin.read() if source == '-' else Path(source).read_text(encoding='utf-8-sig')
    ops = json.loads(raw)
    if isinstance(ops, dict):
        return [ops]
    if not isinstance(ops, list):
        raise ValueError('Expected an operation or a list of operations')
    return ops


def _apply(editor, ops, directory):
    methods = {'add': editor.add, 'add_mask': editor.add_mask, 'move': editor.move,
               'replace': editor.replace, 'set_visible': editor.set_visible}
    for i, op in enumerate(ops):
        if not isinstance(op, dict) or op.get('op') not in methods:
            raise ValueError(f'Unknown object operation at index {i}. Choose from {", ".join(methods)}')
        args = dict(op)
        method = methods[args.pop('op')]
        for key in ('image', 'clip', 'mask'):
            if isinstance(args.get(key), str):
                args[key] = directory / args[key]
        try:
            method(**args)
        except (TypeError, ValueError, KeyError, OSError) as exc:
            raise ValueError(f'Object operation {i} ({op["op"]}): {exc}') from exc


def _include_objects(mask, editor, ids):
    for id in ids:
        if id in editor.object_ids:
            covered = editor.mask(id)
            if mask is None:
                mask = covered
            else:
                mask |= covered
    return mask


def _distinct(paths):
    supplied = [Path(p).resolve() for p in paths if p is not None]
    if len(supplied) != len(set(supplied)):
        raise ValueError('Use separate paths for the document, sources, preview and exported image')


def cmd_edit(a):
    document = Path(a.document)
    _distinct((document, a.base, a.preview, a.out, a.ops if a.ops != '-' else None))
    if a.base is not None and document.exists():
        raise FileExistsError('The object document already exists. Omit --base to open it.')
    editor = ObjectEditor(a.base) if a.base is not None else ObjectEditor.load(document)
    before = editor.render()
    focus = _include_objects(None, editor, a.focus)
    if a.ops:
        directory = Path.cwd() if a.ops == '-' else Path(a.ops).resolve().parent
        _apply(editor, read_ops(a.ops), directory)
    for id in a.focus:
        if id not in editor.object_ids:
            raise ValueError(f'Unknown focus object: {id!r}')
    focus = _include_objects(focus, editor, a.focus)
    after = editor.render()
    if a.preview:
        result = inspect_edit(before, after, mask=focus, padding=a.padding, path=a.preview).metadata
    else:
        box, count = changed_bounds(before, after)
        result = dict(changed_bbox=box, changed_pixels=count)
    if a.out:
        result['out'] = str(Path(_imwrite(a.out, after)).resolve())
    if a.base is not None or a.ops:
        editor.save(document)
    return dict(document=str(document.resolve()), objects=list(editor.object_ids), **result)


def cmd_inspect(a):
    folder = Path(a.session)
    st = Studio.load(folder, log=None)
    previous = folder / 'undo' / '1'
    before = Studio.load(previous, log=None) if (previous / 'meta.json').is_file() else None
    report = st.inspect(a.region, object_id=a.object, before=before, path=a.out, what=a.what,
                        padding=a.padding, lit=not a.unlit)
    return dict(baseline='previous_batch' if before is not None else 'current', **report.metadata)


def add_commands(sub):
    p = sub.add_parser('edit', help='edit named pixel objects and compare a batch against its input')
    p.add_argument('document', help='editable Brushwork object archive')
    p.add_argument('ops', nargs='?', help='JSON operations file or - for stdin; assets are relative to this file')
    p.add_argument('--base', help='start a new document using this background image')
    p.add_argument('--preview', help='write a before/after comparison sheet')
    p.add_argument('--focus', nargs='+', default=(), help='object IDs used to select the local preview')
    p.add_argument('--padding', type=int, default=16, help='context around the focus in canvas pixels')
    p.add_argument('--out', help='export the composited image')
    p.set_defaults(fn=cmd_edit)
    p = sub.add_parser('inspect', help='compare a painting session to its previous batch')
    p.add_argument('session')
    focus = p.add_mutually_exclusive_group()
    focus.add_argument('--region', help='region or mask name for the local preview')
    focus.add_argument('--object', help='retained drawing object ID for the local preview')
    p.add_argument('--what', choices=('canvas', 'sketch'), default='canvas')
    p.add_argument('--unlit', action='store_true', help='inspect paint without relief lighting')
    p.add_argument('--padding', type=int, default=16, help='context in output pixels')
    p.add_argument('--out', required=True, help='path for full and local before/after comparisons')
    p.set_defaults(fn=cmd_inspect)
