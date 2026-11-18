---
name: brushwork-painting
description: "用 brushwork 的 Python 画笔作画，按对象修改底稿或已有画作，检查局部前后对照。动画可用分层、逐帧重画或笔触运动。"
version: 1.3.0
platforms: [windows, linux]
metadata:
  hermes:
    tags: [painting, brush, oil, gouache, watercolor, pastel, charcoal, ink, python, 画笔, 油画, 水粉, 水彩, 色粉, 炭笔, 水墨, 绘画, 模拟笔刷]
    related_skills: []
---

# brushwork：用模拟的真实画笔和画材作画

你决定画什么、每一笔怎么走，工具模拟鬃毛、飞白、湿画混色、厚涂、布纹或纸纹和最后的打光。画材可选油画（默认）、丙烯、水粉、水彩、色粉、炭笔、水墨。可用形状从零搭底稿，也可读取已有画作，用具名对象修改局部。

## 何时使用

用户要求「用 Python 模拟真实画笔作画」「Painted fully in Python」，要用某种真实画材（水粉、水彩、色粉、炭笔、水墨……）一笔一笔画，或点名 brushwork、这套画笔工具时使用。

用户要给已有画作改眼神、移动局部对象或比较修改前后时，也可使用这套工具。普通 AI 生图请求走生图 skill。

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
- **画材（medium）**：建画布时选，`Studio(..., medium='gouache')`。它改变的是颜料和画在什么上面，作画的步骤和所有操作都一样；见下面「画材」。
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
| `refine(regions, passes)` | 对照底稿，偏差大的地方用更小的笔重画。`passes` 为 `'broad'`、`'coarse'`、`'fine'`、`'detail'`（眼、唇这类小五官，最小到 0.33 单位） |
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
     brush=1.2,               # 笔半径，单位；也可以写名字，如 'liner'（见下面「画笔粗细」）
     color='#fff8e0',         # 或 pigments={'alizarin': .5, 'white': .5}，或 black='violet'
     style='line',            # line 勾线 | dab 厚点 | hair 细长 | body | impasto 厚涂 | smooth 薄 | glaze | wash
     smooth=True,             # 把折线变成过各点的曲线
     sheen=0.8, hue='teal',   # 可选：珠光
     thick=2.4, opacity=0.9, taper=0.9)   # 可选：覆盖 style 的任何旋钮
```

`on='眼睛'` 或 `on=['左眼','右眼']` 按实际笔宽逐像素裁切颜料，可传区域名、具名遮罩或 Python 布尔遮罩。一条直线即使两个端点都在选区外，穿过选区的部分仍可着色。用 `lit=False` 或 CLI 的 `--unlit` 核对实际颜料边界。打光时，选区内的厚涂可能在邻近像素投下阴影。显式 `dry=True` 仍会干燥整幅画。

可编辑底稿上的笔触还可用 `on={"object":"left_iris"}` 选中一个对象的可见影响范围。即使左右虹膜使用同一个绘画区域，也可以只画其中一个。

区域属性：`style` 画法；`vary` 色彩活泼程度，0 平整、0.4 默认、1 是印象派式的碎色（只是每一笔在底稿颜色附近抖一点，见「经验」）；`brush` 最大笔半径；`length` 笔画长度范围；`black=True` 把这个区域当成「五彩斑斓的黑」，用互补色调出的黑加珠光来画。

## 按对象修改与局部预览

**底稿用画宽 1000 单位。** 将底稿指令按对象分组，按遮挡顺序放入 `Drawing`。每个对象有唯一 ID，可包含多条操作，区域名仍指绘画区域。`set_drawing` 接受 `Drawing` 或其 `to_dict()` 形式。以下示例先留一份修改前的 Studio，用同一份源在当前底稿上移动虹膜。

```python
from brushwork import Drawing, Studio

source = Drawing()
source.add('iris', [{'op': 'ellipse', 'center': [500, 350], 'rx': 24, 'ry': 30,
                     'color': '#72532c', 'region': 'eye'}])
previous = Studio(400, 300, sketch=source)
st = Studio(400, 300)
st.set_drawing(source)
st.edit_drawing('iris', 'move', dx=-10, dy=2)
report = st.inspect(object_id='iris', before=previous, what='sketch', path='iris-check.png')
```

`edit_drawing` 还支持 `replace`（传 `ops`）、`hide`、`show` 和 `add`。修改会按原顺序重建底稿，下一次铺色或细化使用新底稿。会话存档保留对象源。直接用 `st.sk` 改写这份底稿后，继续按 ID 修改须重新载入对象源。CLI 用 `do` 提交 `set_drawing` 或 `edit_drawing`，再用 `python brush.py inspect <会话> --object iris --what sketch --out iris-check.png` 与前一批操作比较。查看绘画区域时改用 `--region eye`。

**成品用原图像素。** `ObjectEditor(base)` 接受背景图片路径或 RGB(A) 数组。`add(id, tile, xy=(x,y), clip=mask, clip_xy=(cx,cy))` 把前景注册为对象。移动对象时 `clip` 固定在画布上，可用于保护眼睑。`move(id,dx,dy)`、`replace(id,tile)`、`set_visible(id,False)` 分别移动、替换和隐藏对象。对象移开后会露出下层，处理扁平图时须先准备这部分底色。`save` 和 `ObjectEditor.load` 用同一份 NPZ 档案恢复底图、对象顺序、位置和遮罩。

用 `from brushwork import inspect_edit` 导入预览接口。`report = inspect_edit(before, after, mask=mask, path='check.png')` 比较两张同尺寸的 8 位 RGB(A) 图，也可用 `bbox=(左,上,右,下)` 选定局部，右、下边界不包含在框内。打开对照图检查全貌和局部，读 `report.metadata` 中的 `changed_bbox`、`changed_pixels` 和 `changed_outside_selection`。CLI 的 `edit <档案> - --preview check.png` 从标准输入接收一批对象操作，需要聚焦某个对象时加 `--focus <ID>`。在 `<工程目录>/README.md` 的「实际改一次眼神」一节可以直接复现《星落之前》的修改。

## 画笔粗细

凡是要笔半径的地方（`strokes` 的 `brush`、`dabs`、`glaze`、`soften`、`underpaint`、`mend`、区域属性 `brush`），都可以写数字（单位），也可以写名字：

| 名字 | 半径（单位） | 1000 像素宽时 | 3200 像素宽时 | 用来画 |
| --- | --- | --- | --- | --- |
| `hairline` | 0.25 | 0.25 px | 0.8 px | 睫毛尖、发丝、最细的线 |
| `liner` | 0.4 | 0.4 px | 1.3 px | 勾线、眼睑、细发丝 |
| `fine` | 0.7 | 0.7 px | 2.2 px | 五官、小高光、细节 |
| `small` | 1.2 | 1.2 px | 3.8 px | 小物件、脸的明暗 |
| `medium` | 2.5 | 2.5 px | 8 px | 衣褶、手臂 |
| `large` | 5 | 5 px | 16 px | 大块颜色 |
| `broad` | 10 | 10 px | 32 px | 天空、墙面、底色 |
| `wide` | 18 | 18 px | 58 px | 大面积打底 |

- **细节要靠大画布**：笔小于一个像素就画不出比一个像素更细的东西。`hairline`、`liner` 的细节要在 2400 像素宽以上的画布上才出得来；小稿上看不出，不代表大图上画不出。
- **底稿也要给得够细**：`refine` 只在底稿 `detail` 不大于笔半径的地方用那支笔。眼睛、嘴唇想用 `'detail'` 这一档，就把那里的 `detail` 设到 0.33–0.6。
- **照着线稿画时**：细线用 `liner`/`hairline` 按线稿的粗细勾；睫毛这种粗而有形的黑色，用小笔把它的形状填满，比一笔粗线更像原稿。

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

## 画材

建画布时选定，之后不能换：`Studio(width, height, medium='watercolor')`，命令行是 `brush.py new <会话目录> --medium watercolor`。不写就是油画。

| 画材 | 画出来是什么样 | 适合 |
| --- | --- | --- |
| `oil` 油画（默认） | 厚涂起棱，湿的颜色互相带色，柔和的光泽，亚麻布纹 | 要颜料的分量、厚涂高光、深而透亮的暗部 |
| `acrylic` 丙烯 | 像油画，但干得快：混色少、颜色干脆、边缘利落、半哑光 | 色块鲜明、边缘清楚的画 |
| `gouache` 水粉 | 不透明、哑光、薄：平整色块，笔触只留在颜色上，纸纹隐约 | 插画、动画背景风格；脸和细节比油画干净 |
| `watercolor` 水彩 | 半透明淡彩，纸纹透出，边缘柔而不齐，叠一层深一层 | 清淡的风景、天空、花卉 |
| `pastel` 色粉 | 干，每笔一个颜色，只落在纸纹凸处，中间调的色纸从笔触间透出 | 柔和的肖像、风景 |
| `charcoal` 炭笔 | 只有黑白灰，干，在纸纹上碎开，可以揉开 | 素描、速写、明暗习作 |
| `ink` 水墨 | 墨分浓淡，干笔出飞白，宣纸纤维，只有墨色 | 写意、线描 |

- **画材带着自己的默认值**：底色（`primer`）、要不要先罩一层底色、调色盘。水彩、水墨、炭笔保留纸的白，不罩底色；色粉画在中间调的色纸上；炭笔和水墨默认只用白加黑。这些都可以在建画布时另外指定。
- **底稿照常用彩色画**：调色盘只有黑白时，颜色会自动落成相应深浅的灰。
- **透明的画材只能越叠越深**：水彩、炭笔、水墨的笔触带一部分透明罩染，同一处反复加笔会变深，修正也很难把它改亮。亮处在底稿里就留成纸色，从浅往深画，少在一个地方来回补。点高光的 `dab` 仍然是不透明的白。
- **色粉、炭笔不补针孔**：纸色从笔触间透出来就是它们的样子，`mend()` 对它们不起作用。
- **它们是近似**：水彩和水墨用半透明的笔触加纸纹来模拟，没有颜料在湿纸上的渗化、水渍。
- **看画材**：`python brush.py media` 列出全部画材和用途，`brush.py media gouache` 显示一份完整配置；`python examples/media_sheet.py` 把同一幅静物用每种画材各画一遍，存到 `out/media_sheet.jpg`。

**自己写一种画材**：画材是配置，不是代码。复制 `brushwork/media/` 里最接近的一份 JSON，改了以后把路径当画材传进去：`medium='my_gouache.json'`，或 `--medium my_gouache.json`。也可以直接传一个 dict。会话会把整份配置存下来，换机器 `replay` 也不需要原文件。可用的键：

| 键 | 含义 |
| --- | --- |
| `support` | 画在什么上面：`{"kind": "canvas"}` 布纹；`{"kind": "paper", "grain": 1.6, "coarse": 4.5, "broad": 14, "fibres": 0}` 纸纹三种尺寸（单位）和宣纸纤维 |
| `primer` / `ground` / `palette` | 纸或布的底色；`paint()` 先罩的底色，`false` 表示不罩；默认调色盘 |
| `scale` | 乘在每种笔法（style）原有参数上的系数 |
| `all` / `styles` | 直接替换参数：`all` 对每种笔法，`styles` 对某一种，如 `{"line": {"opacity": 0.9}}` |
| `body` | 颜料能堆多厚：最后乘在每一笔的厚度上，包括各步骤和你自己指定的（油画 1，水粉 0.3，水彩 0.06） |
| `render` | 成品怎么打光：`relief` 起伏、`spec` 光泽、`shadow` 投影、`weave_amp` 纸纹或布纹的强弱 |
| `mend` | `false` 表示纸色透出就是要的效果，不补针孔 |

笔触参数都可以写进 `all`、`styles`、`scale`：`opacity` `thick` `dry`（多早在纸纹上碎开） `pickup`（带起湿颜料） `unmix`（笔上颜色不匀） `furrow` `ridge` `blob`（起伏） `taper` `skip` `wobble` `ragged`（手的稳定程度） `mult`（0 不透明，1 只会把下面变深的透明罩染）。完整说明在 `brushwork/media.py` 开头。

## 让成画动起来

写一个带 `build()` 的 Python 场景，返回 `Animation(底画, seconds=8, fps=30)`。按从远到近的顺序用 `add(名字, 图层)` 叠上动作。底画使用本项目画出的 RGB 图，图层也可带透明通道。用下面的三条创作指令选择画法，一幅画可以同时使用三种。

| 画法 | 可以直接给画师的指令 | Python 接口 |
| --- | --- | --- |
| 1 分层 | 先画完整的水面，再画前景人物。固定裙摆接触线，让贴着裙边的倒影轻颤，下方波带逐渐横向错开。 | `PaintedLayer(image, origin=..., warp=..., clip=...)` |
| 2 逐帧重画 | 单独重画双眼。上眼睑盖过高光，闭合略快，睁开略慢。每张从同一底色和种子起笔。 | `RepaintLayer(factory, draw, origin=..., hold=1, active=...)` |
| 3 笔触运动 | 逐笔画出左侧暖光，记录颜料、路径和笔毛种子。让同一道波峰上的几笔先后亮起，波峰过去就淡去。 | `with st.capture_strokes(seed=...) as tape`，然后用 `StrokeLayer(factory, tape, motion, over=True)` |

**分层。** 把会露出的背景补画完整。带透明边缘的云或发梢可用 `origin(frame)` 位移。`warp(frame,x,y)` 返回到原图取样的偏移量，单位是本层像素。正的水平偏移会取到右边的颜色。`clip` 是固定在目标图层位置的遮罩，可用它保护人物轮廓。

**逐帧。** `factory()` 返回全新的 `Studio`，也可以在这里用 `Studio.load(...)` 恢复画好的底层。`draw(st, frame)` 重新下笔，默认输出这张画布，也可以返回自己的 RGB(A) 小画。画布取整幅尺寸即可从头重画整帧。局部动作按需要裁小，`hold=2` 让每张画占两帧，`active(frame)` 指定动作时间。把每个独立动作的种子固定，顺序播放和拖动时间轴应得到同一张画。

**逐笔。** 在动笔前开始捕获，实际落笔的路径、宽度、鬃毛种子、颜料配方、笔触参数、顺序和干湿操作都会进入记录。`motion(batch, frame)` 可用 `batch.ids` 认出同一笔，用 `batch.translate(dx,dy)` 平移，直接改 `batch.points` 弯曲路径，或改 `batch.parameter('opacity')[:]` 控制强度。笔触层每帧恢复同一张物理底画。用 `over=True` 把油彩带来的颜色和笔脊变化叠到正在运动的下层。用 `tape.save(path)` 和 `StrokeTape.load(path)` 保存并读回笔触。单独保存或重建同尺寸底画，便可复现原来的湿画混色。

`frame.index` 从零开始，`frame.seconds` 是秒，`frame.phase` 是一个循环内从 0 到 1 之前的进度。波形可用 `2 * pi * frame.phase`。检查看第零帧、第 239 帧及回绕后的第零帧，两端之间应只隔一次正常运动。

```powershell
python brush.py animate examples/cloud_passage_animation/scene.py --list-layers
python brush.py animate examples/cloud_passage_animation/scene.py --layers glints --out out/glints.mp4
python examples/cloud_passage_animation/compare.py
```

最后一条命令生成《云渡》的合成 MP4 和可切换三种画法的离线对照页。导出 MP4 使用系统 PATH 上带 libx264 的 FFmpeg。

## 回放作画过程

想让人看到这幅画是怎么一笔笔画出来的，就在动笔前打开记录，画完存一个网页：

```python
st = Studio(1200, 1500)
st.record()                                   # 从这里起记下每一笔和每个步骤结束时的画面
...                                           # 照常画
st.save_replay('process.html', title='标题')   # 一个独立的 HTML 文件，双击就能看
```

命令行会话存着全部指令，任何会话都能事后导出：`brush.py replay <会话目录> --width 1200 --out final.png --html process.html`，每次 `do` 是回放里的一步。

- **网页里能做什么**：播放、暂停、调速，拖进度条，点步骤名直接跳到那一步画完的样子，按住「底稿」看底稿。打开时先显示成品，按播放从空白画布开始画。
- **看到的是真画**：每一步里的笔按真实的落笔顺序、沿各自的路径把这一步画完的样子揭开，像素都是引擎画出来的，网页不重新模拟颜料。同一步里被后面的笔盖住的地方，会直接显示成被盖住之后的样子；步骤分得越细，回放越贴近真实过程。
- **记录不改变画**：打开记录画出来的画和不打开时逐像素相同，只是每个步骤结束时多渲染一张图（3200 宽时每张约 1 秒）。
- **大小**：步骤图默认 1200 宽（`record(frame_width=...)`）。2 万笔左右的画，网页约 1 MB；6 万笔、二十多步的大画约 5–10 MB。

## 命令行会话

```
python brush.py new   <会话目录> --width 1000 --height 625 --seed 2 [--medium gouache] [--palette with_black]
python brush.py do    <会话目录> ops.json [--look view.jpg]     # ops.json 是 [{"op": ...}, ...]，- 表示从标准输入读
python brush.py look  <会话目录> --out v.jpg [--crop X Y W H --zoom 2] [--what both]
python brush.py undo  <会话目录>                                 # 撤销上一次 do，最多三步
python brush.py info  <会话目录>
python brush.py replay <会话目录> --width 3200 --out final.png   # 用同样的指令按别的尺寸重画；加 --html p.html 另存回放网页
python brush.py media [名字]                                      # 列出画材，或显示一种的完整配置
python brush.py crop  <图片> X Y W H [--zoom 2]                  # 从任何图片裁一块（像素）
```

`do` 出错时状态不变，错误信息在返回的 JSON 里。JSON 指令的参数和上面的方法参数同名。

## 经验

- **从后往前搭底稿**，让每个位置都属于某个区域。区域的登记顺序就是铺色顺序。
- **运笔方向跟着形体走**：桌面横、墙面斜、布料顺褶、头发顺发丝。小而圆的东西不要用 `around`，会画出靶心，用带 `spread` 的角度。
- **`cut_in` 只用于硬轮廓**，并用 `against` 指明对象。对云这类软边的东西切边会留下一圈印子。
- **`refine` 管不到 detail 太大的地方**：`broad` 管 detail ≤ 6，`coarse` ≤ 2.7，`fine` ≤ 1.7，`detail` ≤ 0.65。想让某处更准，把那里的 detail 调小。
- **细线画两遍**：底稿里用 `line` 标出，最后再用 `strokes` 勾一次。高光、反光留到最后用 `dab` 厚点。
- **一定要放大看**：全图缩略看不出眼睛歪没歪、边缘漏没漏底。`look(crop=..., zoom=2)`。
- **哪里错了就地重画**：再画一个底稿形状、对那个区域 `block_in` 加 `refine`，或直接 `strokes` 盖上去。不必从头来。
- **自己用循环生成笔触时**（比如上千根发丝），颜色要取自这一笔真正落在的区域，别取到旁边物体上。
- **想要油画的碎色就调高 `vary`**，想要干净准确的颜色就调低。但 `vary` 只让每一笔的颜色在底稿颜色附近小幅抖动（OKLab 标准差 0.004–0.014），造不出比一笔更大的色彩变化。
- **小笔会照抄底稿，要活得先画进底稿**：大笔在一点取色、拖得很远，会盖过底稿的细节，色块的变化是顺带出来的；换成小笔短笔触，画出来就是底稿本身，底稿平，画面就平。比如倒影：底稿只是倒过来、模糊过的人，小笔画出来是一块灰板；底稿里先画上横向断开的波纹、缝里露出的水色和几道亮光，同样的小笔就画出了水。把某处换成小笔之前，先用 `look(crop=..., what='both')` 看看那里的底稿平不平。
- **小圆点要叠出来**：一笔比它自己还宽的短笔，会被两端的收笔吃掉，几乎什么都不留。瞳孔、圆的高光这类小点，用几道并排的短横笔叠成。
- **精细的小东西一层干了再画下一层**：画眼睛时每层之间 `st.cv.dry(1.0)`，笔触里写 `pickup=0`，否则后一笔会把前一层还湿的颜料带起来，瞳孔和虹膜的深色会被混掉。
- **薄云、卷云用底稿画**：在底稿里叠几条半透明、边缘虚化的 `ribbon`，让天空自己的笔触去画它们。用噪声阈值做会碎成一片小点，用 `strokes` 直接画会变成生硬的白弧。
- 速度参考：1000 像素宽的静物约 10 秒，2000 像素的海景约 5 秒，3200×4000 的人物约 80 秒。

## 自检

`python tests/check.py`，约一分钟，七项都应为 PASS。
