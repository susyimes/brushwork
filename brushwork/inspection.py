"""Compare two rendered canvases. All boxes are half-open (left, top, right, bottom) pixels."""
from dataclasses import dataclass
from operator import index
from pathlib import Path

import cv2
import numpy as np

from .animation import _image
from .studio import _imwrite


@dataclass
class Inspection:
    image: np.ndarray
    metadata: dict


def _measure(mask):
    row_counts = np.count_nonzero(mask, axis=1)
    rows = np.flatnonzero(row_counts)
    if rows.size == 0:
        return None, 0
    cols = np.flatnonzero(mask.any(axis=0))
    bbox = [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]
    return bbox, int(row_counts.sum())


def _change_mask(before, after):
    before, after = _image(before), _image(after)
    if before.shape != after.shape:
        raise ValueError('Before and after renders must have the same shape and channel count')
    return np.any(before != after, axis=2)


def changed_bounds(before, after):
    """Return (half-open pixel bbox, changed pixel count), or (None, 0) when identical."""
    return _measure(_change_mask(before, after))


def _clip_box(box, w, h):
    try:
        left, top, right, bottom = map(index, box)
    except (ValueError, TypeError) as exc:
        raise ValueError('bbox needs four integer pixel coordinates (left, top, right, bottom)') from exc
    result = [max(0, left), max(0, top), min(w, right), min(h, bottom)]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError('bbox must have a nonempty intersection with the canvas')
    return result


def _fit(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    size = max(1, round(w * scale)), max(1, round(h * scale))
    if image.shape[2] == 4:
        # Composite before resizing so hidden RGB values cannot fringe the preview.
        alpha = image[:, :, 3].astype(np.uint16)
        background = (255 - alpha) * 235
        rgb = np.empty((h, w, 3), np.uint8)
        for channel in range(3):
            rgb[:, :, channel] = (image[:, :, channel] * alpha + background + 127) // 255
        image = rgb
    mode = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    return cv2.resize(image, size, interpolation=mode)


def _draw_sheet(before, after, crop, changed_count, outside):
    width, gap, header, footer = 440, 14, 32, 34
    height, canvas_width = after.shape[:2]
    x0, y0, x1, y1 = crop
    rows = [[], []]
    for image in (before, after):
        full = _fit(image, width, 300)
        h, w = full.shape[:2]
        if crop != [0, 0, canvas_width, height]:
            # The overview locates the magnified crop. Detail panels stay unobstructed.
            left, top = x0 * w // canvas_width, y0 * h // height
            right = (x1 * w + canvas_width - 1) // canvas_width - 1
            bottom = (y1 * h + height - 1) // height - 1
            cv2.rectangle(full, (left, top), (right, bottom), (26, 32, 38), 3)
            cv2.rectangle(full, (left, top), (right, bottom), (248, 197, 74), 1)
        rows[0].append(full)
        rows[1].append(_fit(image[y0:y1, x0:x1], width, 320))

    row_heights = [row[0].shape[0] for row in rows]
    sheet = np.full((sum(row_heights) + 2 * header + footer + gap, 2 * width + 3 * gap, 3),
                    (246, 245, 242), np.uint8)
    y = 0
    for row, row_height, label in zip(rows, row_heights, ('canvas', 'detail')):
        for column, (image, when) in enumerate(zip(row, ('Before', 'After'))):
            x = gap + column * (width + gap) + (width - image.shape[1]) // 2
            cv2.putText(sheet, when + ' / ' + label, (x, y + 22), cv2.FONT_HERSHEY_SIMPLEX,
                        .54, (44, 49, 53), 1, cv2.LINE_AA)
            sheet[y + header:y + header + row_height, x:x + image.shape[1]] = image
        y += header + row_height + gap
    status = f'{changed_count:,} changed pixels' if changed_count else 'No changed pixels'
    if outside is not None:
        status += f'  |  {outside:,} outside selection'
    cv2.putText(sheet, status, (gap, sheet.shape[0] - 11), cv2.FONT_HERSHEY_SIMPLEX,
                .49, (44, 49, 53), 1, cv2.LINE_AA)
    return sheet


def inspect_edit(before, after, *, bbox=None, mask=None, padding=16, path=None):
    """Return full before/after previews, matching detail crops and exact pixel-change counts.

    Supply RGB or RGBA uint8 renders of equal shape. Select a focus with a canvas-pixel
    bbox (e.g. the union of an object's old and new bounds) or a canvas-sized boolean or
    numeric mask (nonzero coverage counts as selected). Padding adds context to the
    preview only. Without a selection, focus on changed pixels, or the whole unchanged
    canvas. Global changes outside the selection still count. RGBA differences include
    alpha and hidden RGB. Saving the comparison to `path` is optional.
    """
    before, after = np.asarray(before), np.asarray(after)
    changed = _change_mask(before, after)
    if bbox is not None and mask is not None:
        raise ValueError('Choose a bbox or a mask to select the detail')
    padding = index(padding)
    if padding < 0:
        raise ValueError('padding must be nonnegative')
    h, w = after.shape[:2]
    changed_bbox, count = _measure(changed)
    selection = None
    outside = None
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != (h, w) or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all():
            raise ValueError('mask must contain finite coverage values and match the canvas height and width')
        selected = mask != 0
        selection, _ = _measure(selected)
        outside = count - int(np.count_nonzero(changed & selected))
    elif bbox is not None:
        selection = _clip_box(bbox, w, h)
        left, top, right, bottom = selection
        outside = count - int(np.count_nonzero(changed[top:bottom, left:right]))
    left, top, right, bottom = selection or changed_bbox or [0, 0, w, h]
    crop = _clip_box((left - padding, top - padding, right + padding, bottom + padding), w, h)
    sheet = _draw_sheet(before, after, crop, count, outside)
    metadata = dict(canvas_size=[w, h], selection_bbox=selection, changed_bbox=changed_bbox,
                    crop_bbox=crop, changed_pixels=count, changed_outside_selection=outside, path=None)
    if path is not None:
        metadata['path'] = _imwrite(str(Path(path).resolve()), sheet)
    return Inspection(sheet, metadata)
