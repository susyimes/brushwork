"""Regression checks for all three animation methods, including physically exact replay."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import runpy
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brushwork import Animation, PaintedLayer, RepaintLayer, StrokeLayer, StrokeTape, Studio


def studio():
    st = Studio(128, 96, seed=41, primer='#65798b', log=None)
    st.strokes([dict(path=[(50,460),(970,430)], brush=75, color='#8eb8cc', style='body')])
    return st


def wet_pass(st):
    st.strokes([dict(path=[(170,200+i*53),(850,370+i*24)], brush=35, color=col, sheen=.48)
                for i,col in enumerate(('#dfb779','#c7d0bf','#453d62'))], shuffle=True)
    st.dry(.58)
    mask = np.zeros((st.H, st.W), bool)
    mask[28:67, 40:84] = True
    st.cv.rewet(mask,.73)
    st.strokes([dict(path=[(640,95),(350,620)], brush=57, color='#a76162', pickup=.28)])
    st.cv.update_tooth()


class AnimationChecks(unittest.TestCase):
    def test_physical_recording_and_poses(self):
        source, untouched, dest = studio(), studio(), studio()
        source.record(frame_width=80)             # stage-image replay can watch the same brushes
        with source.capture_strokes() as tape:
            wet_pass(source)
        wet_pass(untouched)
        self.assertEqual(tape.count,4)
        self.assertEqual(source.rng.bit_generator.state, untouched.rng.bit_generator.state)
        self.assertEqual(source.cv.rng.bit_generator.state, untouched.cv.rng.bit_generator.state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'accents.brushes'
            tape.save(path)
            restored = StrokeTape.load(path)
        restored.replay(dest)
        for field in ('col','sca','wet','hgt','thk','iri','ifd','tooth'):
            np.testing.assert_array_equal(getattr(source.cv,field),getattr(untouched.cv,field))
            np.testing.assert_array_equal(getattr(source.cv,field),getattr(dest.cv,field))
        expected = source.render()
        ids = []
        def move(batch, frame):
            ids.extend(batch.ids.tolist())
            batch.translate(8*frame.index, batch.ids*frame.index)
            batch.parameter('opacity')[:] *= 1-.14*frame.index
        scene = Animation(expected,seconds=1,fps=4)
        scene.add('oil',StrokeLayer(studio, restored, move))
        np.testing.assert_array_equal(expected,scene.frame(0))
        self.assertEqual(ids,[0,1,2,3])
        moved = scene.frame(2)
        self.assertFalse(np.array_equal(moved,expected))
        np.testing.assert_array_equal(moved,scene.frame(-2))
        np.testing.assert_array_equal(expected,scene.frame(scene.count))
        with self.assertRaises(ValueError):
            restored.replay(Studio(96,96,log=None))

    def test_capture_seed_scope_and_failure(self):
        st1, st2 = studio(), studio()
        st2.rng.random(19)
        st2.cv.rng.random(31)
        states = st2.rng.bit_generator.state, st2.cv.rng.bit_generator.state
        with st1.capture_strokes(seed=70) as a:
            wet_pass(st1)
        with st2.capture_strokes(seed=70) as b:
            wet_pass(st2)
        np.testing.assert_array_equal(st1.cv.col,st2.cv.col)
        self.assertEqual(states,(st2.rng.bit_generator.state,st2.cv.rng.bit_generator.state))
        with self.assertRaisesRegex(RuntimeError,'interrupted'):
            with st2.capture_strokes(seed=11):
                raise RuntimeError('interrupted')
        self.assertIsNone(st2.cv.stroke_tape)
        self.assertEqual(states,(st2.rng.bit_generator.state,st2.cv.rng.bit_generator.state))
        self.assertEqual(a.count,b.count)

    def test_pixel_layer_transparent_edges_and_order(self):
        backdrop = np.zeros((6,8,3),np.uint8)
        lower = np.full((4,4,3),[0,100,0],np.uint8)
        icon = np.zeros((2,2,4),np.uint8)
        icon[:,0] = [255,0,0,0]
        icon[:,1] = [0,0,240,255]
        scene = Animation(backdrop,seconds=1,fps=2)
        scene.add('lower',PaintedLayer(lower,origin=(-1,1)))
        scene.add('icon',PaintedLayer(icon,origin=lambda f:(f.index,2),warp=lambda f,x,y:(.5,0)))
        out = scene.frame(0,['icon','lower'])
        np.testing.assert_array_equal(out[2,0],[0,50,120])
        np.testing.assert_array_equal(scene.frame(1)[2,0],[0,100,0])
        self.assertEqual(scene.frame(1,'lower')[2,0,1],100)
        self.assertFalse(scene.frame(0,[]).any())
        with self.assertRaises(ValueError):
            scene.frame(0,'missing')

    def test_fresh_drawings_on_twos_local_and_full_canvas(self):
        calls = []
        def draw(st, f):
            calls.append(f.index)
            st.sk.background('#708c9d',region='sky',direction=0,detail=5)
            st.sk.ellipse((260+48*f.index,370),119,67,color='#ead7a9',region='cloud',detail=2)
            st.paint()
        plate = np.full((104,146,3),17,np.uint8)
        scene = Animation(plate,seconds=1,fps=6)
        factory = lambda: Studio(128,96,seed=18,log=None)
        scene.add('cloud',RepaintLayer(factory,draw,origin=(7,4),hold=2))
        a = scene.frame(0)
        np.testing.assert_array_equal(a,scene.frame(1))
        b = scene.frame(2)
        np.testing.assert_array_equal(b,scene.frame(3))
        np.testing.assert_array_equal(a[:4],b[:4])
        np.testing.assert_array_equal(a[:,135:],b[:,135:])
        self.assertFalse(np.array_equal(a,b))
        np.testing.assert_array_equal(a,scene.frame(0))
        self.assertEqual(calls,[0,2,0])
        full = Animation(plate[4:100,7:135],seconds=1,fps=6)
        full.add('whole',RepaintLayer(factory,draw))
        np.testing.assert_array_equal(full.frame(2),b[4:100,7:135])
        self.assertFalse(np.array_equal(full.frame(1),full.frame(2)))

    def test_wet_overlay_keeps_live_underpaint_and_fixed_ids(self):
        st = studio()
        with st.capture_strokes(seed=19) as tape:
            st.strokes([dict(path=[[40,360],[880,340]],brush=37,color='#ffe7ab',style='impasto')])
        plate = np.full((92,106,3),[109,148,177],np.uint8)
        def motion(batch, f):
            batch.parameter('opacity')[:] *= f.index
        scene = Animation(plate,seconds=1,fps=2)
        scene.add('glint',StrokeLayer(studio,tape,motion,origin=(-12,-2),over=True))
        np.testing.assert_array_equal(plate,scene.frame(0))
        self.assertGreater(np.abs(scene.frame(1).astype(int)-plate).max(),12)
        np.testing.assert_array_equal(plate,scene.frame(2))
        expected = [scene.frame(i) for i in range(2)]
        with ThreadPoolExecutor(max_workers=2) as workers:
            for i, img in zip((1,0,1,0),workers.map(scene.frame,(1,0,1,0))):
                np.testing.assert_array_equal(img,expected[i])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg is needed for an export')
    def test_failed_export_preserves_previous_movie(self):
        scene = Animation(np.zeros((48,64,3),np.uint8),seconds=1,fps=2)
        def fail(st, frame):
            if frame.index:
                raise RuntimeError('interrupted pose')
            return np.zeros((48,64,3),np.uint8)
        scene.add('drawing',RepaintLayer(studio,fail))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'painting.mp4'
            output.write_bytes(b'previous complete film')
            with self.assertRaisesRegex(RuntimeError,'interrupted pose'):
                scene.write_mp4(output)
            self.assertEqual(output.read_bytes(),b'previous complete film')
            self.assertEqual(list(Path(folder).iterdir()),[output])

    def test_cloud_passage_eight_second_scene(self):
        path = Path(__file__).resolve().parents[1]/'examples'/'cloud_passage_animation'/'scene.py'
        loop = runpy.run_path(str(path))['build']()
        self.assertEqual((loop.count,loop.fps),(240,30))
        self.assertEqual(list(loop.layers),['reflection','blink','glints'])
        np.testing.assert_array_equal(loop.frame(240),loop.frame(0))
        for name, min_y in [('reflection',1235),('glints',1235)]:
            a,b = loop.frame(0,name),loop.frame(114,name)
            np.testing.assert_array_equal(a[:min_y],b[:min_y])
            self.assertGreater(np.count_nonzero(np.abs(a.astype(np.int16)-b).max(axis=2)>4),1000)
        blue = []
        for i in range(loop.count):
            frame = loop.frame(i,'blink')
            eye = frame[349:395,692:910].astype(np.int16)
            blue.append(np.count_nonzero((eye[...,2]-eye[...,0]>26)&(eye[...,1]-eye[...,0]>12)
                                         &(eye[...,2]>75)))
        closed = np.asarray(blue) < .35*np.median(blue)
        self.assertEqual(np.count_nonzero(np.diff(np.r_[False,closed,False])),2)
        self.assertLess(min(blue),blue[0]*.1)


if __name__=='__main__':
    unittest.main()
