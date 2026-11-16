"""Render the combined movie and a self-contained eight-second comparison player."""
import argparse
import base64
from pathlib import Path
import sys
import tempfile

import cv2
from numba import get_num_threads, set_num_threads

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scene import build


VIEWS = (('all','合成'),('reflection','1 · 分层倒影'),('blink','2 · 逐帧眨眼'),('glints','3 · 逐笔水光'))


def comparison(out, width):
    out.mkdir(parents=True, exist_ok=True)
    loop = build()
    buttons, videos = [], []
    with tempfile.TemporaryDirectory(dir=out) as folder:
        for i, (name, label) in enumerate(VIEWS):
            movie = out/'云渡-合成.mp4' if i == 0 else Path(folder)/(name+'.mp4')
            loop.write_mp4(movie, layers=None if i==0 else [name], width=width if i==0 else min(960,width))
            print(f'{label} · {loop.count} frames', flush=True)
            data = base64.b64encode(movie.read_bytes()).decode('ascii')
            state = 'true' if i==0 else 'false'
            buttons.append(f'<button data-mode="{name}" aria-pressed="{state}">{label}</button>')
            playback = 'autoplay' if i==0 else 'hidden'
            videos.append(f'<video id="{name}" aria-label="{label}" controls loop muted playsinline {playback} '
                          f'preload="auto" src="data:video/mp4;base64,{data}"></video>')
        template = Path(__file__).with_name('compare.html').read_text(encoding='utf-8')
        tmp = Path(folder)/'comparison.html'
        tmp.write_text(template.replace('<!-- views -->','\n'.join(videos))
                       .replace('<!-- modes -->','\n'.join(buttons)), encoding='utf-8')
        result = out/'云渡-三种画法.html'
        tmp.replace(result)
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'out'/'云渡-三种动画测试')
    parser.add_argument('--width', type=int, default=1600)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    set_num_threads(min(6,get_num_threads()))
    print(comparison(args.out,args.width).resolve())
