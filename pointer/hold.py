"""Hold-to-identify: the user must keep pointing at the same object for `hold_s` seconds.

Time comes from the video timestamps, not the wall clock, so results on a recorded clip are the
same no matter how fast the machine processes it.
"""

from dataclasses import dataclass

import numpy as np

from .detector import Detection


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class HoldEvent:
    label: str
    confidence: float
    time_s: float
    started_s: float
    description: str | None = None  # filled in by the vision-language model, if enabled
    crop_box: object = None         # region sent to the model (x1, y1, x2, y2)
    vlm_score: float | None = None  # the model's confidence in its answer (0..1)


class HoldSelector:
    def __init__(self, hold_s: float = 2.0, grace_s: float = 0.4, same_iou: float = 0.3,
                 switch_s: float = 0.2):
        self.hold_s = hold_s
        self.grace_s = grace_s      # tolerate brief hand/detection dropouts
        self.switch_s = switch_s    # a different target must persist this long to take over
        self.same_iou = same_iou    # a new box counts as the same object above this IoU
        self.reset()

    def reset(self):
        self.current: Detection | None = None
        self.started_s = 0.0
        self.last_seen_s = 0.0
        self.fired = False
        self.confidences: list[float] = []
        self.pending: Detection | None = None
        self.pending_since = 0.0

    def _same(self, det: Detection) -> bool:
        return (self.current is not None and det.label == self.current.label
                and box_iou(det.box, self.current.box) >= self.same_iou)

    def progress(self, now_s: float) -> float:
        if self.current is None:
            return 0.0
        return min(1.0, (now_s - self.started_s) / self.hold_s)

    def update(self, target: Detection | None, now_s: float) -> HoldEvent | None:
        """Feed the current target; returns an event the moment the hold completes."""
        if target is None:
            if self.current is not None and now_s - self.last_seen_s > self.grace_s:
                self.reset()
            return None
        if self.current is None:
            self.reset()
            self.current, self.started_s = target, now_s
        elif self._same(target):
            self.current = target  # follow the box as it moves
            self.pending = None
        else:
            # Debounce: a one-frame flicker to another object must not restart the timer.
            if self.pending is None or not (target.label == self.pending.label
                                            and box_iou(target.box, self.pending.box) >= self.same_iou):
                self.pending, self.pending_since = target, now_s
            else:
                self.pending = target
            if now_s - self.pending_since < self.switch_s:
                self.last_seen_s = now_s
                return None
            since = self.pending_since
            self.reset()
            self.current, self.started_s = target, since
        self.last_seen_s = now_s
        self.confidences.append(target.confidence)
        if not self.fired and now_s - self.started_s >= self.hold_s:
            self.fired = True
            return HoldEvent(target.label, float(np.mean(self.confidences)), now_s, self.started_s)
        return None
