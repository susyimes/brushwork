# brushwork：给 agent 用的真实笔刷绘画工具

用 Python 模拟油画笔，一笔一笔地画。agent 决定画什么、怎么画；工具负责让颜料像颜料：鬃毛带料、飞白、湿画带色、厚涂起伏、布纹和打光。不读入任何图片，不调用任何生图模型。

这个工程最初是为画《五彩斑斓的黑》写的，现在画笔和作画流程已经和那幅画分开，可以画任何题材。

## 画过的三幅

仓库里不带成品图，运行例子后生成，命令见下面「重画三幅例子」。

| 运行后得到 | 题材 | 怎么画的 |
| --- | --- | --- |
| `out/astra_iridescent_black.png` | 爱式塔，3200×4000，49,317 笔 | `examples/astra_iridescent_black/`，手写底稿加专门的作画步骤 |
| `out/still_life.png` | 柠檬与青花碗，2000×1600 | `examples/still_life.py`，约百行，只用公开接口 |
| `replay --out` 指定的文件 | 落日海景，2000×1250 | `examples/seascape/*.json`，三十来条 JSON 指令，经命令行会话画成 |

`examples/brush_sheet.py` 生成笔触试样 `out/brush_sheet.jpg`；人物那幅加 `--frames` 会另存八个作画阶段的过程图。

## 给 agent 的说明

`skill/SKILL.md` 是写给 agent 读的完整用法：概念、流程、全部操作、经验。让哪个 agent 用这套工具，就让它先读这个文件。

装给本机的 agent：

```powershell
python brush.py install-skill
```

它把说明写到 `~/.agents/skills/brushwork-painting/SKILL.md`，并把其中的 `<工程目录>` 换成本机的实际路径，`--dest` 可以改位置。然后在 agent 的全局说明里加一句指向这个文件，比如 Claude Code 的 `~/.claude/CLAUDE.md`。改过 `skill/SKILL.md` 之后重跑一次。

## 两种用法

写脚本：

```python
import sys
sys.path.insert(0, r'<工程目录>')           # 本仓库的路径
from brushwork import Studio

st = Studio(width=1000, height=800)
sk = st.sk                                  # 底稿
sk.background('#3a3f3a', region='wall', direction={'angle': 70, 'spread': 35}, detail=5.5)
sk.rect(0, 470, 1000, 330, color='#8a6a48', region='table', direction=3, detail=4)
sk.ellipse((500, 520), 110, 80, color='#e6c022', region='lemon', direction=15, detail=1.6)
sk.form_shadow('lemon', '#a2780f', width=60)
st.paint(ground='#75604a')                  # 罩底、打底、铺色、修正
st.look('view.jpg')                         # 存一张图来看
st.save_image('lemon.png')
```

命令行会话，状态存在目录里，可以分多次画、可以撤销：

```powershell
python brush.py new   work\sea --width 1000 --height 625
python brush.py do    work\sea examples\seascape\01_sketch.json
python brush.py do    work\sea examples\seascape\02_paint.json --look view.jpg
python brush.py do    work\sea examples\seascape\03_accents.json
python brush.py undo  work\sea
python brush.py replay work\sea --width 3200 --out final.png
```

位置和粗细都用「单位」表示，画布永远宽 1000 单位，所以同一份脚本或指令既能出小稿也能出大图。

## 重画三幅例子

```powershell
python examples\astra_iridescent_black\paint.py            # 约 80 秒；--width 1000 出小稿，--frames 存过程图
python examples\still_life.py                              # --width 1000 出小稿
python examples\brush_sheet.py                             # 笔触试样
```

海景按上面命令行会话的六条命令画。人物那幅用同一个种子重画，输出文件逐字节相同。

## 画笔怎么模拟

一笔是一排鬃毛扫过的带子，每根鬃毛各带各的颜料和载料量：

- 载料沿笔画逐渐耗尽，之后只在画布纹理的高处留色，形成飞白。
- 经过未干的颜料时，鬃毛会把它带起来混进自己的颜料，再拖到后面。黄色笔拖过湿的蓝色会变绿。
- 每根鬃毛在颜料层上犁出自己的沟，笔画两侧挤出棱，提笔处留一小坨。
- 鬃毛不会走得笔直：笔道会游走、分叉，偶尔离开画布。

颜料有 13 管，没有黑色，按 Kubelka-Munk 模型在吸收和散射空间里混合，所以互补色混在一起会变黑而不是变灰。另有一层按薄膜干涉计算颜色的珠光介质，颜色只出现在笔痕迎光的坡面上。画完后把颜料层的起伏和布纹一起打光。

## 作画流程

底稿规定每个位置的颜色、所属区域、运笔方向和最小用笔。作画按画家的顺序走：罩底色，跨区域打底，从后往前逐区域铺色，沿轮廓切边，对照底稿把偏差大的地方用更小的笔重画，最后手工落笔勾线点高光。每一步之后都可以存图来看，再决定下一步。

## 目录

- `brush.py`：命令行入口。
- `brushwork/engine.py`：鬃毛画笔、画布、打光，numba 编译。
- `brushwork/color.py`：颜料、混色、配方查找、薄膜干涉色。
- `brushwork/sketch.py`：底稿，用形状和造型调用搭出来。
- `brushwork/studio.py`：作画的各个步骤，以及存图、存档。
- `brushwork/cli.py`：会话、JSON 指令、撤销、重放。
- `brushwork/geom.py`：曲线和栅格化工具。
- `examples/`：三幅画的源头和笔触试样。
- `skill/SKILL.md`：给 agent 的用法说明。
- `tests/check.py`：自检，约一分钟。

依赖：Python 3 与 numpy、scipy、numba、opencv-python。

## 自检

```powershell
python tests\check.py
```

四项：每个操作都能运行；存档再读取后继续画，结果与一次画完相同；命令行会话全流程可用；人物那幅与改造前逐像素一致。

## 关于《五彩斑斓的黑》

画面 58% 的面积比 L\*25 还暗，这些暗部的亮度中位数是 L\*11，其中中性灰不到 1%：它是黑的，也是彩色的。黑色区域只用品红、紫、蓝、青、碧这一圈冷色的珠光，金色只留给月蚀的光环和零星几笔。

人物造型是对照角色设定图手工定的坐标：白色短发、月牙发饰、灰色眼睛、15 号「黑曜哥特」的方领蕾丝和酒红蝴蝶结。虹膜里的彩色细丝和部分星星用的是官方九种虹膜色。程序运行时不读取任何设定文件或图片。
