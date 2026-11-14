"""brushwork: paint with simulated oil brushes.

    from brushwork import Studio
    st = Studio(width=1000, height=800)
    st.sk.background('#3a3f3a', region='wall', direction={'angle': 70, 'spread': 35}, detail=5.5)
    st.sk.ellipse((500, 520), 110, 80, color='#e6c022', region='lemon', direction=15, detail=1.6)
    st.paint(ground='#75604a')
    st.look('view.jpg')

Studio is the canvas and the painter's passes; Studio.sk is the underdrawing (a Sketch).
The full guide, written for agents, is skill/SKILL.md in the project.
"""
from .color import (CHROMATIC_BLACKS, PALETTES, PIG, PIGMENT_DEFS, SHEEN_HUES, as_lin, film_for, palette_tubes,
                    recipe, rgb8)
from .geom import arclen, catmull, ellipse_pts, normals, offset, resample, ribbon, tangents
from .sketch import Sketch
from .studio import BROAD, COARSE, FINE, STYLE, Studio
from .cli import run_ops

__all__ = ['Studio', 'Sketch', 'run_ops', 'STYLE', 'BROAD', 'COARSE', 'FINE', 'SHEEN_HUES', 'CHROMATIC_BLACKS',
           'PIGMENT_DEFS', 'PIG', 'PALETTES', 'palette_tubes', 'as_lin', 'rgb8', 'recipe', 'film_for', 'catmull', 'resample', 'ellipse_pts',
           'ribbon', 'tangents', 'normals', 'offset', 'arclen']
