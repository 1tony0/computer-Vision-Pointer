"""Fallback pointer from the segmentation mask of the user's own arm.

In first-person video the pointing hand is often too close, cut off by the frame edge and seen
from the back, which makes MediaPipe's palm detector miss it. The YOLO segmenter still labels it
"person", so we recover the fingertip from the mask shape:

1. The user's arms are "person" masks that touch the bottom edge, or a side edge in the lower
   part of the frame (other people in the scene rarely do). When both arms are visible, the one
   whose tip is most finger-like (lowest width ratio, step 5) and closest to the previous
   frame's fingertip is the pointing arm.
2. The entry point is the centroid of mask pixels along the frame border.
3. The fingertip is the outline point farthest from the entry point (the arm's extremity).
4. Pairs of outline points at equal arc length on either side of the tip lie on opposite edges of
   the finger; their midpoints trace its centre line, and tip - mean(midpoints) is the direction.
5. The pose counts as pointing only when the end of the mask is a thin finger: the mask's width
   measured across the pointing direction just behind the tip (`finger_width`, 2.5% of the frame
   diagonal back) must be finger-sized. A fist, a relaxed hand or a hand holding something is much
   wider there. The chord/arc `width_ratio` near the tip is a coarser second check.
6. Only the chosen hand points (the right one by default). Which arm is which follows from where it
   enters the frame: the right arm comes in from the right part of the bottom edge or the right side.
"""

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIZE = 320  # masks are rasterised at this long side for speed


@dataclass
class MaskPointer:
    tip: np.ndarray        # fingertip, full-frame pixels
    direction: np.ndarray  # unit vector
    entry: np.ndarray      # where the arm enters the frame
    length: float          # entry -> tip distance (hand/arm scale)
    width_ratio: float     # outline chord / arc length near the tip; low = finger-like
    contour: np.ndarray
    finger_width: float = 0.0  # mask width just behind the tip / frame diagonal
    side_label: str | None = None  # set by arm_candidates when both arms are visible

    def is_pointing(self, max_width_ratio: float, max_finger_width: float = 0.06) -> bool:
        return self.width_ratio <= max_width_ratio and self.finger_width <= max_finger_width

    def side(self, frame_w: int, split: float = 0.45) -> str:
        """Which of the user's arms this is, from where it enters the frame."""
        if self.side_label is not None:
            return self.side_label
        return "right" if self.entry[0] >= split * frame_w else "left"

    def pointing_ray(self):
        # The mask gives no depth cue, so foreshortening is reported as 1 (use the ray).
        return self.tip.copy(), self.direction.copy(), 1.0

    @property
    def size(self) -> float:
        return self.length


def _rasterise(contour: np.ndarray, frame_size: tuple[int, int]):
    w, h = frame_size
    scale = WORK_SIZE / max(w, h)
    mw, mh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    mask = np.zeros((mh, mw), np.uint8)
    cv2.fillPoly(mask, [(contour * scale).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask, scale


def _border_pixels(mask: np.ndarray, side_from: float, margin: int = 2) -> np.ndarray:
    """Mask pixels on the bottom edge, or on the left/right edges below `side_from` x height.
    Segmentation outlines often stop a pixel or two short of the frame edge, so pixels within
    `margin` (at the work resolution) of an edge count as on it."""
    h, w = mask.shape
    y0 = int(h * side_from)
    ys, xs = np.nonzero(mask)
    on_bottom = ys >= h - 1 - margin
    on_side = (ys >= y0) & ((xs <= margin) | (xs >= w - 1 - margin))
    keep = on_bottom | on_side
    pts = np.stack([xs[keep], ys[keep]], 1).astype(np.float32)
    # Snap to the edge so the entry point lies on the border.
    pts[on_bottom[keep], 1] = h - 1
    side = on_side[keep] & ~on_bottom[keep]
    pts[side & (pts[:, 0] <= margin), 0] = 0
    pts[side & (pts[:, 0] >= w - 1 - margin), 0] = w - 1
    return pts


def arm_candidates(contours: list[np.ndarray], frame_size: tuple[int, int], side_from: float = 0.4,
                   arc_min: float = 0.08, arc_max: float = 0.22, min_border_px: int = 3,
                   min_length: float = 0.08) -> list[MaskPointer]:
    """The user's own arms: person masks attached to the frame edge, with their fingertip.

    With only one arm in view, its side comes from where it enters the frame. With both arms in
    view they are compared with each other: the one entering further right is the right arm. A
    fixed split fails when the left arm reaches across and enters near the middle. Masks entering
    close together are pieces of the same arm."""
    arms = [c for c in (_arm_pointer(contour, frame_size, side_from, arc_min, arc_max,
                                     min_border_px, min_length) for contour in contours) if c]
    if len(arms) >= 2:
        xs = np.array([a.entry[0] for a in arms])
        if xs.max() - xs.min() >= SAME_ARM * frame_size[0]:
            for a in arms:
                a.side_label = "right" if xs.max() - a.entry[0] < SAME_ARM * frame_size[0] else "left"
    return arms


SAME_ARM = 0.15  # x frame width: arm masks entering closer than this are treated as one arm


def find_mask_pointer(contours: list[np.ndarray], frame_size: tuple[int, int],
                      side_from: float = 0.4, arc_min: float = 0.08, arc_max: float = 0.22,
                      min_border_px: int = 3, min_length: float = 0.08,
                      prev_tip: np.ndarray | None = None, track_weight: float = 4.0,
                      hand: str = "any", arms: list[MaskPointer] | None = None) -> MaskPointer | None:
    """Both forearms often enter the frame. Keep only the arm on the chosen side (`hand`: "right",
    "left" or "any"), then pick the one whose tip looks most like a finger and, when the previous
    frame's fingertip is known, stays close to it (temporal consistency)."""
    candidates = arms if arms is not None else arm_candidates(contours, frame_size, side_from, arc_min,
                                                               arc_max, min_border_px, min_length)
    if hand != "any":
        candidates = [c for c in candidates if c.side(frame_size[0]) == hand]
    if not candidates:
        return None
    diag = float(np.hypot(*frame_size))

    def cost(c: MaskPointer) -> float:
        jump = 0.0 if prev_tip is None else float(np.linalg.norm(c.tip - prev_tip)) / diag
        return c.width_ratio + track_weight * jump

    return min(candidates, key=cost)


def _arm_pointer(contour, frame_size, side_from, arc_min, arc_max, min_border_px, min_length):
    mask, scale = _rasterise(contour, frame_size)
    border = _border_pixels(mask, side_from)
    if len(border) < min_border_px:
        return None  # not attached to the frame edge: someone else in the scene
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    entry = border.mean(axis=0)
    dist = np.linalg.norm(outline - entry, axis=1)
    i_tip = int(np.argmax(dist))
    tip = outline[i_tip]
    length = float(dist[i_tip])
    if length < max(5.0, min_length * max(mask.shape)):
        return None

    # Walk the outline the same arc length in both directions from the tip: the two points sit on
    # opposite edges of the finger, so their midpoints trace the finger's centre line.
    n = len(outline)
    offsets = np.arange(max(2, int(arc_min * length)), max(3, int(arc_max * length)) + 1)
    left = outline[(i_tip + offsets) % n]
    right = outline[(i_tip - offsets) % n]
    mids = (left + right) / 2
    axis = tip - mids.mean(axis=0)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return None
    # Chord across the outline at the largest offset, relative to that arc length:
    # ~2 for a round blob (fist), much smaller for a finger with parallel edges.
    width_ratio = float(np.linalg.norm(left[-1] - right[-1]) / offsets[-1])
    direction = axis / norm
    diag = float(np.hypot(*mask.shape))
    finger_width = _width_across(mask, tip - direction * FINGER_BACK * diag, direction) / diag

    return MaskPointer(tip=tip / scale, direction=direction, entry=entry / scale,
                       length=length / scale, width_ratio=width_ratio, contour=contour,
                       finger_width=finger_width)


FINGER_BACK = 0.025  # where the finger is measured: this fraction of the frame diagonal behind the tip


def _width_across(mask: np.ndarray, centre: np.ndarray, direction: np.ndarray) -> float:
    """Length of the run of mask pixels through `centre`, perpendicular to `direction`."""
    perp = np.array([-direction[1], direction[0]])
    h, w = mask.shape
    run = 0
    for sign, start in ((1, 0), (-1, 1)):
        for k in range(start, w + h):
            x, y = (centre + sign * perp * k).astype(int)
            if not (0 <= x < w and 0 <= y < h) or not mask[y, x]:
                break
            run += 1
    return float(run)
