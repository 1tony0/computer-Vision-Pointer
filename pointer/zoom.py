"""Digital zoom controlled by hand gestures.

- Start: an OK sign (thumb+index pinch, other three fingers open) held for `confirm_s`
  (one-frame tracking dropouts shorter than `grace_s` don't reset the timer).
- Zoom: while that hand stays up (middle, ring and pinky still open), opening the gap between
  thumb and index zooms in. The change is proportional to how far the gap opened since the start,
  so a fully opened hand gives about `gain` x 1.7 more zoom. Closing the fingers again does not
  zoom out (zoom only goes in); the open palm is the only way back to 1x.
  Opening the hand fully looks like an open palm; during a zoom session that is just "zoom in".
- End: lower the hand (or curl the three fingers) for `end_s`; the zoom stays where it is.
- Reset to 1x: an open palm held for `reset_s`, but only once no zoom session is running and
  `cooldown_s` has passed since the last one ended.
- Zoom only works while the system is armed and snaps back to 1x when it disarms.

The zoom crops the frame around the point where the gesture started; the detector runs on that
crop (upscaled by the model), which helps with small or distant objects.
"""

import numpy as np


class ZoomController:
    def __init__(self, max_zoom: float = 4.0, smoothing: float = 0.3, reset_s: float = 0.6,
                 confirm_s: float = 0.3, gain: float = 2.0, end_s: float = 0.3, cooldown_s: float = 1.0,
                 grace_s: float = 0.2):
        self.max_zoom = max_zoom
        self.smoothing = smoothing
        self.reset_s = reset_s
        self.confirm_s = confirm_s
        self.gain = gain
        self.end_s = end_s
        self.cooldown_s = cooldown_s
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.center: np.ndarray | None = None
        self.active = False            # a zoom session is running
        self._start_dist = 0.0
        self._start_zoom = 1.0
        self._last_seen = float("-inf")
        self._ended_at = float("-inf")
        self._wrist: np.ndarray | None = None
        self._ok_since: float | None = None
        self._ok_last = float("-inf")
        self.grace_s = grace_s
        self._palm_since: float | None = None

    @staticmethod
    def _three_open(hand) -> bool:
        return all(hand.is_extended(f) for f in ("middle", "ring", "pinky"))

    def update(self, hands, now_s: float, frame_size: tuple[int, int], enabled: bool = True) -> None:
        if not enabled:
            hands = []
            self.target_zoom = 1.0
            self.active = False

        if self.active:
            # Follow the same hand (nearest wrist) as long as its three fingers stay open.
            cands = [h for h in hands if self._three_open(h)]
            hand = min(cands, key=lambda h: np.linalg.norm(h.points[0] - self._wrist), default=None)
            if hand is not None:
                self._last_seen, self._wrist = now_s, hand.points[0].copy()
                wanted = self._start_zoom + self.gain * (hand.pinch_distance - self._start_dist)
                self.target_zoom = float(np.clip(max(self.target_zoom, wanted), 1.0, self.max_zoom))
            elif now_s - self._last_seen > self.end_s:
                self.active, self._ended_at = False, now_s
        else:
            ok = next((h for h in hands if h.is_ok_sign()), None)
            if ok is not None:
                if self._ok_since is None or now_s - self._ok_last > self.grace_s:
                    self._ok_since = now_s  # a new OK sign (gaps up to grace_s are dropouts)
                self._ok_last = now_s
            elif now_s - self._ok_last > self.grace_s:
                self._ok_since = None
            if ok is not None and now_s - self._ok_since >= self.confirm_s:
                self.active, self._last_seen, self._wrist = True, now_s, ok.points[0].copy()
                self._start_dist, self._start_zoom = ok.pinch_distance, self.target_zoom
                self._ok_since = None
                if self.center is None or self.target_zoom <= 1.01:
                    self.center = (ok.points[4] + ok.points[8]) / 2

            palm = next((h for h in hands if h.is_open_palm()), None)
            if palm is not None and ok is None and now_s - self._ended_at >= self.cooldown_s:
                self._palm_since = self._palm_since if self._palm_since is not None else now_s
                if now_s - self._palm_since >= self.reset_s:
                    self.target_zoom = 1.0
            else:
                self._palm_since = None

        self.zoom += self.smoothing * (self.target_zoom - self.zoom)
        if abs(self.zoom - self.target_zoom) < 0.01:
            self.zoom = self.target_zoom
        if self.zoom <= 1.0 and not self.active:
            self.center = None

    def crop(self, frame_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """Crop rectangle (x, y, w, h) for the current zoom, or None at 1x."""
        if self.zoom <= 1.01:
            return None
        w, h = frame_size
        cw, ch = int(w / self.zoom), int(h / self.zoom)
        cx, cy = self.center if self.center is not None else (w / 2, h / 2)
        x = int(np.clip(cx - cw / 2, 0, w - cw))
        y = int(np.clip(cy - ch / 2, 0, h - ch))
        return x, y, cw, ch
