---
name: brushwork-painting
description: "用 Python 模拟的真实油画笔作画（brushwork）：自己构底稿、铺色、细化、点高光，一笔一笔画出成品。"
version: 1.0.0
platforms: [windows, linux]
metadata:
  hermes:
    tags: [painting, brush, oil, python, 画笔, 油画, 绘画, 模拟笔刷]
    related_skills: []
---

# brushwork：用模拟油画笔作画

一个画布加一套画家的动作。你决定画什么、每一笔怎么走；它负责让颜料像颜料：鬃毛带料、飞白、湿画带色、厚涂起伏、布纹和打光。不读入图片，不调用任何生图模型。

## 何时使用

用户要求「用 Python 模拟真实画笔作画」「Painted fully in Python」，或点名 brushwork、这套画笔工具时使用。

不用于：照着现成图片转绘（工具不读图）、需要文字排版的画面、普通的 AI 生图请求（那走生图 skill）。

## 位置与两种用法

工程在 `<工程目录>`。依赖 numpy、scipy、numba、opencv-python。下文出现的相对路径（`brush.py`、`examples/`、`out/`、`tests/`）都相对这个目录。

<!-- repo-only -->
> `<工程目录>` 指本仓库的根目录，也就是 `brush.py` 所在的那一层。`python brush.py install-skill` 会把这份说明装到 `~/.agents/skills/brushwork-painting/`，并把 `<工程目录>` 换成实际路径。

<!-- /repo-only -->
**写脚本（推荐，表达力最强）**：

```python
import sys
sys.path.insert(0, r'<工程目录>')
from brushwork import Studio
```

**命令行会话（不想写脚本时）**：`python <工程目录>/brush.py <命令>`，每次调用输出一行 JSON。画的状态存在会话目录里，可以分多次画。

## 基本概念

- **单位**：画布永远宽 1000 单位，x 向右、y 向下，高度按比例（1200×1500 的画布高 1250 单位）。位置、笔的粗细、笔画长度都用单位，所以同一份指令既能画 1000 像素的小稿，也能画 3200 像素的成品。
- **颜色**：`'#rrggbb'` 或 `[r, g, b]`（0–255）。工具会把它换成真实颜料的配方，给出这些颜料调得出的最接近的颜色。默认调色盘是 25 管彩色加白，不含黑，深色是混出来的；见下面「调色盘」。
- **底稿（Sketch，`st.sk`）**：动笔前的意图。用形状从后往前搭：每个形状可以规定它覆盖处的颜色、所属区域（region）、运笔方向（direction）、最小用笔（detail）。再用几个造型调用把平涂变成有体积的东西。
- **作画（Studio，`st`）**：罩底 → 整体打底 → 逐区域铺色 → 切边 → 对照底稿修正 → 手工落笔。
- **看（look）**：把画面存成图片，自己打开看，再决定下一步。这一步不能省。

## 流程

1. 小尺寸起稿：`Studio(width=1000, height=1250)`。
2. 搭底稿，然后 `st.look('s.jpg', what='sketch')` 检查造型和颜色。底稿不对，后面怎么画都不对。
3. `st.paint(ground='#6b5548')` 一步走完常规流程；或者分步调用（见下）。
4. `st.look('a.jpg')` 看全貌，再用 `crop` 放大看每个重要部位。
5. 用 `st.strokes([...])` 补线条、高光和细节；哪里画坏了就在上面重画。
6. 满意后把宽度改成 2400–3200 重跑脚本（会话则用 `replay`），`st.save_image('final.png')`。

## 最小例子

```python
st = Studio(width=1000, height=800, seed=1)
sk = st.sk
sk.background({'linear': [[0, 0], [1000, 480]], 'colors': ['#676b60', '#2a2e2c']}, region='wall',
              direction={'angle': 70, 'spread': 35}, detail=5.5)
sk.rect(0, 470, 1000, 330, color='#8a6a48', region='table', direction=3, detail=4)
sk.ellipse((500, 520), 110, 80, rot=-10, color='#e6c022', region='lemon', detail=1.6,
           direction={'angle': 15, 'spread': 30, 'scale': 60})
sk.form_shadow('lemon', '#a2780f', light=(-1, -0.6), width=60, amount=0.85)   # 背光面
sk.form_light('lemon', '#fbe670', light=(-1, -0.6), width=50, amount=0.8)     # 受光面
sk.cast_shadow('lemon', 'table', '#35261c', offset=(44, 14), blur=5)          # 投影
st.paint(ground='#75604a')
st.strokes([dict(path=[[455, 488], [475, 482]], brush=7, color='#fffce6', style='dab', thick=2.6)])  # 高光
st.look('lemon.jpg')
st.save_image('lemon.png')
```

完整例子：`examples/still_life.py`（静物，约百行）、`examples/seascape/*.json`（海景，纯 JSON 指令）、`examples/astra_iridescent_black/`（人物，最复杂）。运行后成品写到 `out/`。

## 底稿操作（`st.sk.<名字>`，JSON 里 `{"op": "<名字>", ...}`）

形状，后画的盖住先画的：

| 操作 | 参数 |
| --- | --- |
| `background` | `color, region='background', direction, detail` 铺满画布，最先调用 |
| `polygon` / `spline` | `points`；spline 是过这些点的光滑闭合曲线 |
| `ellipse` | `center, rx, ry, rot`（度） |
| `rect` | `x, y, w, h` |
| `ribbon` | `path, width`；width 可为数、`[起, 止]` 或逐点列表 |

形状共有的可选参数：`color`、`region`（名字，自动登记）、`direction`、`detail`、`soft`（边缘虚化，单位）、`opacity`、`name`（把这块形状存成具名 mask），以及区域属性 `style`、`brush`、`length`、`vary`、`black`。只给 `name` 不给颜色和区域，就只是圈出一块地方供以后引用。

`color` 还可以是渐变：`{'linear': [[x0, y0], [x1, y1]], 'colors': [...]}` 或 `{'radial': [cx, cy, r], 'colors': [内, ..., 外]}`。

造型（`mask` 可以是区域名、具名 mask，或它们的列表）：

| 操作 | 作用 |
| --- | --- |
| `tint(mask, color, amount, blob, ramp, noise, soft)` | 往某色偏。`blob=[cx, cy, rx, ry, rot]` 限定成一团（高光、腮红）；`ramp=[[x0, y0], [x1, y1]]` 由无到有 |
| `form_shadow(mask, color, light, width, amount)` | 背光一侧变暗。`light` 是从物体指向光源的方向，默认左上 `(-1, -0.5)` |
| `form_light(mask, color, light, width, amount)` | 受光一侧变亮 |
| `cast_shadow(caster, onto, color, offset, blur, amount)` | 一个形状落在别的区域上的影子 |
| `line(path, width, color, amount, on)` | 底稿里的一条线 |
| `direction(mask, spec)` / `detail(mask, value)` | 事后改运笔方向、最小用笔 |
| `region_props(region, style, brush, length, vary, black)` | 改一个区域的画法 |
| `sheen_field(mask, amount, hue, noise)` | 哪里上珠光（干涉色）介质，什么色 |

`direction` 的写法：数字是角度（0 向右，90 向下）；`{'angle': a, 'spread': 度, 'scale': 单位}` 是带游移的角度；`{'along': [[x, y], ...]}` 平行于一条曲线；`{'contour': true}` 顺着形状自己的轮廓；`{'around': [x, y]}` 绕点；`{'radial': [x, y]}` 放射；`{'swirl': [x, y], 'pitch': 度}` 螺旋。

`detail` 是该处允许的最小笔半径：眼睛、小字符 0.75；脸、小物件 1.2–1.8；衣物 3；背景 4–6。它决定细化能做到多细。

## 作画操作（`st.<名字>`）

| 操作 | 作用 |
| --- | --- |
| `paint(ground, rounds=2)` | 常规全流程：罩底、打底、铺色、补漏、三档修正 |
| `tone_ground(color)` | 给白底布罩一层底色，选画面的平均色调 |
| `underpaint(regions)` | 跨区域的整体打底，保证接缝处不露底 |
| `block_in(regions)` | 逐区域用大小两号笔铺色，按给的顺序（默认按底稿登记顺序，即从后往前） |
| `cut_in(region, against)` | 用 region 自己的颜色沿它和 against 的交界重画一遍，把别人的轮廓收干净 |
| `mend(regions)` | 找出笔触之间漏出底色的针孔并补上 |
| `refine(regions, passes)` | 对照底稿，偏差大的地方用更小的笔重画。`passes` 为 `'broad'`、`'coarse'`、`'fine'` |
| `strokes(items, on)` | 手工落笔，见下 |
| `dabs(n, regions, brush, colors, stretch, angle, style, seed)` | 在区域或具名 mask 内随机点 n 笔：星星、水面反光、花瓣、肌理 |
| `glaze(regions, color, opacity)` | 透明罩染，只会变暗变色：加深阴影、统一色调 |
| `soften(regions, brush, rewet)` | 干净的笔拖过去，柔化区域内部的硬边 |
| `sheen(regions, strength)` | 按底稿的 sheen_field 罩珠光介质 |
| `dry(amount)` | 让颜料干。湿颜料会被后面的笔带起来混色 |
| `look(path, width, crop, zoom, lit, what)` | 存一张图来看。`crop=[x, y, w, h]`（单位）配 `zoom` 放大；`what` 为 `'canvas'`、`'sketch'`、`'both'`、`'error'` |
| `save_image(path)` | 输出全分辨率成品 |

`strokes` 的每一笔：

```python
dict(path=[[x, y], ...],      # 两个点是一个点触或一段直线
     brush=1.2,               # 笔半径，单位
     color='#fff8e0',         # 或 pigments={'alizarin': .5, 'white': .5}，或 black='violet'
     style='line',            # line 勾线 | dab 厚点 | hair 细长 | body | impasto 厚涂 | smooth 薄 | glaze | wash
     smooth=True,             # 把折线变成过各点的曲线
     sheen=0.8, hue='teal',   # 可选：珠光
     thick=2.4, opacity=0.9, taper=0.9)   # 可选：覆盖 style 的任何旋钮
```

`on=['区域', ...]` 把这批笔限制在这些区域内，出界处自动截断。

区域属性：`style` 画法；`vary` 色彩活泼程度，0 平整、0.4 默认、1 是印象派式的碎色；`brush` 最大笔半径；`length` 笔画长度范围；`black=True` 把这个区域当成「五彩斑斓的黑」，用互补色调出的黑加珠光来画。

## 调色盘

建画布时选定从哪些颜料管里调色：`Studio(palette=...)`，命令行是 `brush.py new ... --palette ...`。

| 调色盘 | 内容 | 什么时候用 |
| --- | --- | --- |
| `'full'`（默认） | 白加 25 管彩色，不含黑 | 一般的彩色画 |
| `'with_black'` | 再加一管象牙黑 | 墨线、水墨、黑白稿、平面风格，需要不带色相的黑和灰 |
| `'classic'` | 最初的 13 管 | 重画旧作；色域较窄，艳蓝艳紫调不出 |
| 颜料名的列表 | 只用这几管 | 限定调色盘，整幅画自动落在这几管调得出的颜色里 |

限定调色盘的例子：`['white', 'ivory_black']` 是单色素描，`['white', 'ochre', 'cad_red', 'ivory_black']` 是佐恩调色盘。列表里也可以写调色盘名，比如 `['classic', 'ivory_black']`。

颜料管：`white`；黄橙 `lemon_yellow` `cad_yellow` `naples_yellow` `ochre` `cad_orange`；红 `cad_red` `pyrrole_red` `alizarin` `rose` `magenta`；紫 `cobalt_violet` `purple` `violet`；蓝 `ultramarine` `cobalt_blue` `cerulean` `phthalo_blue`；青绿 `turquoise` `teal` `phthalo_green` `permanent_green` `sap_green`；土色 `sienna` `umber` `raw_umber`；黑 `ivory_black`。

- **要不要黑**：彩色画不需要。`full` 能调出 `#1a1a1d` 这样的中性深色，比它更深的近黑会带一点色相，而这正是油画暗部好看的原因。只有要纯正的中性黑和灰时才选 `with_black`。
- **点名永远有效**：不管选了哪个调色盘，`strokes` 里写 `pigments={'ivory_black': 1}` 都能直接用那一管。
- **颜料有边界**：屏幕上的纯蓝 `#0000ff`、纯品红 `#ff00ff`、纯绿 `#00ff00` 不是颜料调得出的颜色，会得到最接近的 `#2a3bc9`、`#e150c4`、`#78e73a`。纯黑 `#000000` 同理，最深是 `#0e0e0f`。
- **看颜料**：`python brush.py palette` 列出全部颜料管的色值、调色盘、style 的参数、chromatic black 的名字和珠光色名；`python examples/palette_chart.py` 画一张色卡到 `out/palette_chart.jpg`。

## 命令行会话

```
python brush.py new   <会话目录> --width 1000 --height 625 --seed 2 [--palette with_black]
python brush.py do    <会话目录> ops.json [--look view.jpg]     # ops.json 是 [{"op": ...}, ...]，- 表示从标准输入读
python brush.py look  <会话目录> --out v.jpg [--crop X Y W H --zoom 2] [--what both]
python brush.py undo  <会话目录>                                 # 撤销上一次 do，最多三步
python brush.py info  <会话目录>
python brush.py replay <会话目录> --width 3200 --out final.png   # 用同样的指令按别的尺寸重画
python brush.py crop  <图片> X Y W H [--zoom 2]                  # 从任何图片裁一块（像素）
```

`do` 出错时状态不变，错误信息在返回的 JSON 里。JSON 指令的参数和上面的方法参数同名。

## 经验

- **从后往前搭底稿**，让每个位置都属于某个区域。区域的登记顺序就是铺色顺序。
- **运笔方向跟着形体走**：桌面横、墙面斜、布料顺褶、头发顺发丝。小而圆的东西不要用 `around`，会画出靶心，用带 `spread` 的角度。
- **`cut_in` 只用于硬轮廓**，并用 `against` 指明对象。对云这类软边的东西切边会留下一圈印子。
- **`refine` 管不到 detail 太大的地方**：`broad` 管 detail ≤ 6，`coarse` ≤ 2.7，`fine` ≤ 1.7。想让某处更准，把那里的 detail 调小。
- **细线画两遍**：底稿里用 `line` 标出，最后再用 `strokes` 勾一次。高光、反光留到最后用 `dab` 厚点。
- **一定要放大看**：全图缩略看不出眼睛歪没歪、边缘漏没漏底。`look(crop=..., zoom=2)`。
- **哪里错了就地重画**：再画一个底稿形状、对那个区域 `block_in` 加 `refine`，或直接 `strokes` 盖上去。不必从头来。
- **自己用循环生成笔触时**（比如上千根发丝），颜色要取自这一笔真正落在的区域，别取到旁边物体上。
- **想要油画的碎色就调高 `vary`**，想要干净准确的颜色就调低。
- 速度参考：1000 像素宽的静物约 10 秒，2000 像素的海景约 5 秒，3200×4000 的人物约 80 秒。

## 自检

`python tests/check.py`，约一分钟，五项都应为 PASS。
