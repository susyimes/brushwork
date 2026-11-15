"""Painting media: what the paint is, and what it is painted on.

A medium is configuration, not code.  Each one is a JSON file in brushwork/media/.  Choose it
by name, Studio(medium='gouache') or `brush.py new --medium gouache`, or pass the path of your
own JSON file, or a dict with the same keys.  Every key is optional: whatever a medium leaves
out behaves as oil, which is the default.

  title, summary, use   what it is and when to choose it, for people and agents
  support               what the picture is painted on
                          {"kind": "canvas"}                      a woven canvas, as for oil
                          {"kind": "paper", "grain": 1.6, "coarse": 4.5, "broad": 14,
                           "mix": [0.55, 0.30, 0.15], "fibres": 0}
                        paper grain at three sizes (canvas units) and their mix; fibres
                        (0..1) adds the long fibres of rice paper
  primer                colour of the bare support
  ground                the toned ground paint() lays first: a colour, null for the usual
                        warm mix, or false for none (watercolour keeps the white paper)
  palette               the tubes it mixes from unless the painter says otherwise
  scale                 multipliers on every brush style's own settings
  all                   settings every brush style takes instead (after scale)
  styles                settings for one brush style, after `all`: {"line": {"opacity": 0.9}}
  body                  how far the paint can be piled up: multiplies the thickness of every
                        stroke last of all, even of strokes whose thickness a pass or the
                        painter set explicitly (oil 1, gouache 0.3, watercolour 0.06)
  render                how the finished surface is lit: Canvas.render arguments (relief,
                        spec, shadow, amb, weave_amp, weave_hide, blur ...)
  mend                  false when the support showing between strokes is the look itself
                        (pastel, charcoal): mend() then leaves it alone

The stroke settings are the brush-style knobs listed in brushwork.studio.STYLE (opacity, thick,
dry, pickup, unmix, furrow, ridge, blob, cap, taper, skip, wobble, ragged, mode) and two more:
`mult`, which moves a stroke from opaque towards a transparent glaze that only darkens what is
under it, and `wetdep`, how wet the paint is laid.
"""
import json
import os

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')
KEYS = ('title', 'summary', 'use', 'support', 'primer', 'ground', 'palette', 'scale', 'all', 'styles', 'body',
        'render', 'mend')
STROKE_KEYS = ('opacity', 'thick', 'dry', 'pickup', 'unmix', 'cap', 'furrow', 'ridge', 'blob', 'taper', 'skip',
               'wobble', 'ragged', 'mode', 'mult', 'wetdep', 'loadlen', 'nb')
RENDER_KEYS = ('light', 'amb', 'spec', 'shin', 'pearl', 'glint', 'shadow', 'shadow_steps', 'angle_shift', 'relief',
               'weave_amp', 'weave_hide', 'blur')


def names():
    """The media that come with brushwork."""
    return sorted(f[:-5] for f in os.listdir(MEDIA_DIR) if f.endswith('.json'))


def _read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load(medium=None):
    """A medium by name, JSON path or dict -> Medium.  None means oil."""
    if isinstance(medium, Medium):
        return medium
    if medium is None:
        medium = 'oil'
    if isinstance(medium, dict):
        return Medium(medium, medium.get('name', 'custom'), spec=medium)
    if isinstance(medium, str) and medium.lower().endswith('.json'):
        cfg = _read(medium)
        return Medium(cfg, cfg.get('name') or os.path.splitext(os.path.basename(medium))[0], spec=cfg)
    path = os.path.join(MEDIA_DIR, '%s.json' % medium)
    if not os.path.exists(path):
        raise ValueError('unknown medium %r: the media are %s (or give a .json file or a dict)'
                         % (medium, ', '.join(names())))
    return Medium(_read(path), medium, spec=medium)


class Medium:
    def __init__(self, cfg, name, spec=None):
        bad = [k for k in cfg if k not in KEYS and k != 'name']
        if bad:
            raise ValueError('medium %s: unknown key(s) %s; the keys are %s' % (name, bad, ', '.join(KEYS)))
        for part in ('scale', 'all'):
            odd = [k for k in cfg.get(part, {}) if k not in STROKE_KEYS]
            if odd:
                raise ValueError('medium %s: %s has unknown stroke setting(s) %s' % (name, part, odd))
        for st, d in cfg.get('styles', {}).items():
            odd = [k for k in d if k not in STROKE_KEYS]
            if odd:
                raise ValueError('medium %s: styles.%s has unknown stroke setting(s) %s' % (name, st, odd))
        odd = [k for k in cfg.get('render', {}) if k not in RENDER_KEYS]
        if odd:
            raise ValueError('medium %s: render has unknown setting(s) %s' % (name, odd))
        self.name = name
        self.spec = name if spec is None else spec         # what a saved painting records
        self.cfg = cfg
        self.title = cfg.get('title', name)
        self.support = dict(cfg.get('support') or {'kind': 'canvas'})
        if self.support.get('kind', 'canvas') not in ('canvas', 'paper'):
            raise ValueError('medium %s: support kind must be canvas or paper' % name)
        self.primer = cfg.get('primer')
        self.ground = cfg.get('ground')
        self.palette = cfg.get('palette')
        self.scale = dict(cfg.get('scale') or {})
        self.all = dict(cfg.get('all') or {})
        self.styles = {k: dict(v) for k, v in (cfg.get('styles') or {}).items()}
        self.body = float(cfg.get('body', 1.0))
        self.render = dict(cfg.get('render') or {})
        self.mend_holes = bool(cfg.get('mend', True))
        self.plain = not (self.scale or self.all or self.styles) and self.body == 1.0

    def preset(self, style, base):
        """A brush style's settings in this medium, from the oil preset `base`."""
        d = dict(base)
        if self.plain:
            return d
        for k, f in self.scale.items():
            if k in d:
                d[k] = d[k] * f
        d.update(self.all)
        d.update(self.styles.get(style, {}))
        return d

    def finish(self, d):
        """Last word on a stroke's settings, after the painter's own: the paint's body."""
        if self.body != 1.0 and 'thick' in d:
            d['thick'] = d['thick'] * self.body
        return d

    def support_px(self, S):
        """The support in pixels for a canvas of S pixels per unit; None for a woven canvas."""
        s = self.support
        if s.get('kind', 'canvas') == 'canvas':
            return None
        return dict(grain=float(s.get('grain', 1.6)) * S, coarse=float(s.get('coarse', 4.5)) * S,
                    broad=float(s.get('broad', 14.0)) * S, mix=tuple(s.get('mix', (0.55, 0.30, 0.15))),
                    fibres=float(s.get('fibres', 0.0)), fibre_len=float(s.get('fibre_len', 9.0)) * S)

    def describe(self):
        return dict(name=self.name, title=self.title, summary=self.cfg.get('summary', ''), use=self.cfg.get('use', ''),
                    support=self.support.get('kind', 'canvas'), primer=self.primer, ground=self.ground,
                    palette=self.palette)
