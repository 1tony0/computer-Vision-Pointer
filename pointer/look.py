"""Look mode: name whatever the wearer keeps looking at, without using the hands.

The glasses' camera moves with the head, so "looking at something" means the middle of the frame
stays on it while the head is still. Head motion is measured between frames with phase
correlation (the image shift that best aligns two small greyscale frames).

Rules (thresholds measured on the station clip glasses_1369, in frame widths per second):
- deliberate looking: 0.01-0.06; turning or walking: 0.2-2.0. Steady = smoothed speed < 0.12.
- the view must stay steady for 1.5 s on the same target before it is named. Deliberate
  looks in the station clip lasted 1.7-4.2 s; incidental still moments 1.1-1.3 s.
- after naming something, the head has to move on (0.15 frame widths in total) before the next
  look counts, so staring at one thing doesn't repeat it every 2 s.
"""

import cv2
import numpy as np

from .detector import Detection, contour_contains


class HeadMotion:
    def __init__(self, width: int = 192, smoothing: float = 0.3, steady_speed: float = 0.12,
                 min_response: float = 0.3):
        self.width = width
        self.smoothing = smoothing
        self.steady_speed = steady_speed      # frame widths per second
        self.min_response = min_response      # below this the shift estimate is unreliable
        self.speed = 1.0                      # smoothed; starts "moving" until measured
        self._prev = None
        self._prev_t = None
        self._window = None

    def update(self, frame_bgr: np.ndarray, now_s: float) -> float:
        """Returns how far the view moved since the previous frame (in frame widths)."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (self.width, max(8, int(self.width * h / w))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        moved = 0.0
        if self._prev is not None and self._prev.shape == gray.shape and now_s > self._prev_t:
            if self._window is None or self._window.shape != gray.shape:
                self._window = cv2.createHanningWindow(gray.shape[::-1], cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(self._prev, gray, self._window)
            moved = float(np.hypot(dx, dy)) / self.width
            speed = moved / (now_s - self._prev_t)
            if response < self.min_response:
                speed = max(speed, 2 * self.steady_speed)  # blur or a big change: treat as moving
            self.speed += self.smoothing * (speed - self.speed)
        self._prev, self._prev_t = gray, now_s
        return moved

    @property
    def steady(self) -> bool:
        return self.speed < self.steady_speed


def look_point(frame_size: tuple[int, int], crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Centre of what the wearer sees (of the zoomed view, when zoomed)."""
    if crop is not None:
        x, y, cw, ch = crop
        return np.array([x + cw / 2, y + ch / 2], np.float32)
    w, h = frame_size
    return np.array([w / 2, h / 2], np.float32)


def center_box(center: np.ndarray, frame_size: tuple[int, int], frac: float = 0.35) -> np.ndarray:
    w, h = frame_size
    side = frac * min(w, h)
    x1 = float(np.clip(center[0] - side / 2, 0, w - side))
    y1 = float(np.clip(center[1] - side / 2, 0, h - side))
    return np.array([x1, y1, x1 + side, y1 + side], np.float32)


def look_target(detections: list[Detection], center: np.ndarray, frame_size: tuple[int, int],
                near_frac: float = 0.06) -> tuple[Detection, str]:
    """What is in the middle of the view: the smallest detected object under the centre, else the
    detection nearest the centre (within `near_frac` of the diagonal), else the central region
    itself (named by the language model). Returns (target, mode)."""
    under = [d for d in detections if contour_contains(d.contour, center)]
    if under:
        return min(under, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1])), "look"
    diag = float(np.hypot(*frame_size))
    if detections:
        nearest = min(detections, key=lambda d: np.linalg.norm(d.center - center))
        if np.linalg.norm(nearest.center - center) <= near_frac * diag:
            return nearest, "look"
    box = center_box(center, frame_size)
    x1, y1, x2, y2 = box
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return Detection("object", 0.0, box, rect), "region"


class LookGate:
    """After something has been named, wait until the head has moved on before naming again."""

    def __init__(self, move_on: float = 0.15):
        self.move_on = move_on   # frame widths of total head movement
        self.blocked = False
        self._moved = 0.0

    def announced(self):
        self.blocked, self._moved = True, 0.0

    def update(self, moved: float):
        if self.blocked:
            self._moved += moved
            if self._moved >= self.move_on:
                self.blocked = False
