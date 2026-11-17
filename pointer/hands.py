"""Hand landmark tracking (MediaPipe GestureRecognizer) and simple gesture rules.

GestureRecognizer runs the same hand landmark model as HandLandmarker and adds a trained
classifier for canned gestures (Thumb_Up, Open_Palm, Pointing_Up, ...). Thumb_Up arms the system.

Landmark indices follow the MediaPipe hand model:
0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky.
"""

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP = 9
# (pip, tip) for index, middle, ring, pinky
FINGER_JOINTS = {"index": (6, 8), "middle": (10, 12), "ring": (14, 16), "pinky": (18, 20)}


@dataclass
class Hand:
    points: np.ndarray  # (21, 2) pixel coordinates in the full frame
    handedness: str
    score: float
    gesture: str = "None"
    gesture_score: float = 0.0

    def is_thumbs_up(self, min_score: float = 0.6) -> bool:
        return self.gesture == "Thumb_Up" and self.gesture_score >= min_score

    def is_peace(self, min_score: float = 0.6) -> bool:
        """Peace sign: switches to look mode. MediaPipe's "Victory" label, or the landmarks: index and
        middle fingers far out and apart, ring and pinky curled. The landmark test is needed because
        the glasses see the back of the hand, which the classifier labels "None" (clip 1451).
        Thresholds (fingertip-to-wrist / hand size): real peace signs 2.0-2.8 for index and middle;
        the only near-miss in ten clips was 1.7-1.8, for 0.5 s."""
        if self.gesture == "Victory" and self.gesture_score >= min_score:
            return True
        p, size = self.points, self.size
        reach = lambda tip: float(np.linalg.norm(p[tip] - p[WRIST])) / size
        return (reach(8) >= 1.9 and reach(12) >= 1.9 and reach(16) <= 1.45 and reach(20) <= 1.35
                and float(np.linalg.norm(p[8] - p[12])) / size >= 0.25)

    def is_thumbs_down(self, min_score: float = 0.6) -> bool:
        """MediaPipe's Thumb_Down label, confirmed by the landmarks: the thumb tip must be the
        lowest point of the hand and well below the wrist. Without this check, pointing down at
        something on a desk is often labelled Thumb_Down."""
        if self.gesture != "Thumb_Down" or self.gesture_score < min_score:
            return False
        p = self.points
        return int(p[:, 1].argmax()) == THUMB_TIP and p[THUMB_TIP, 1] - p[WRIST, 1] >= 0.25 * self.size

    @property
    def size(self) -> float:
        """Wrist to middle-finger knuckle distance; a scale reference for thresholds."""
        return float(np.linalg.norm(self.points[MIDDLE_MCP] - self.points[WRIST])) + 1e-6

    def is_extended(self, finger: str) -> bool:
        # A finger is extended when its tip is farther from the wrist than its middle joint.
        # Distances (not y-coordinates) keep this independent of hand rotation.
        pip, tip = FINGER_JOINTS[finger]
        wrist = self.points[WRIST]
        return np.linalg.norm(self.points[tip] - wrist) > 1.15 * np.linalg.norm(self.points[pip] - wrist)

    @property
    def pinch_distance(self) -> float:
        """Thumb-tip to index-tip distance, normalised by hand size."""
        return float(np.linalg.norm(self.points[THUMB_TIP] - self.points[INDEX_TIP])) / self.size

    def is_pointing(self) -> bool:
        return (self.is_extended("index") and not self.is_extended("middle")
                and not self.is_extended("ring") and not self.is_extended("pinky"))

    def is_ok_sign(self, max_pinch: float = 0.35) -> bool:
        """Zoom gesture: thumb and index tips touching while the other three fingers stay open.
        Real OK signs measured 0.03-0.22 hand sizes between the tips; a hand holding something, or
        a pointing hand seen from behind, often has three "open" fingers too but a gap of 0.4-0.7."""
        return (self.is_extended("middle") and self.is_extended("ring") and self.is_extended("pinky")
                and self.pinch_distance < max_pinch)

    def is_open_palm(self) -> bool:
        return all(self.is_extended(f) for f in FINGER_JOINTS) and self.pinch_distance > 0.9

    def pointing_ray(self):
        """Returns (origin, unit direction, foreshortening ratio).

        The direction runs from the index knuckle through the fingertip. The ratio compares the
        on-screen finger length with the hand size; a small ratio means the finger points into
        the scene (towards the camera's viewing direction), so its 2-D direction is unreliable.
        """
        base, tip = self.points[INDEX_MCP], self.points[INDEX_TIP]
        v = tip - base
        length = float(np.linalg.norm(v))
        direction = v / length if length > 1e-6 else np.array([0.0, -1.0])
        return tip.copy(), direction, length / self.size


def two_thumbs_up(hands: list[Hand], min_score: float = 0.6, min_sep: float = 1.0) -> bool:
    """Arming gesture: thumbs-up on both hands at once. A single thumbs-up is not enough, because
    a pointing hand seen from behind is often classified as Thumb_Up. The two hands must be at
    least `min_sep` hand sizes apart, so one hand detected twice doesn't count."""
    ups = [h for h in hands if h.is_thumbs_up(min_score)]
    return any(np.linalg.norm(a.points[0] - b.points[0]) >= min_sep * max(a.size, b.size)
               for i, a in enumerate(ups) for b in ups[i + 1:])


class TwoThumbsUp:
    """two_thumbs_up() with a short memory: each hand's thumbs-up counts for `memory_s` seconds,
    so the two hands don't have to be classified as Thumb_Up in exactly the same frames (a hand
    near the frame edge often flickers between Thumb_Up and None)."""

    def __init__(self, memory_s: float = 0.5, min_score: float = 0.6, min_sep: float = 1.0):
        self.memory_s, self.min_score, self.min_sep = memory_s, min_score, min_sep
        self._recent: list[tuple[float, np.ndarray, float]] = []  # (time, wrist, hand size)

    def update(self, hands: list[Hand], now_s: float) -> bool:
        self._recent = [r for r in self._recent if now_s - r[0] <= self.memory_s]
        self._recent += [(now_s, h.points[0].copy(), h.size) for h in hands if h.is_thumbs_up(self.min_score)]
        # Two remembered thumbs-ups far enough apart to be two different hands, one seen now.
        now = [r for r in self._recent if r[0] == now_s]
        return any(np.linalg.norm(a[1] - b[1]) >= self.min_sep * max(a[2], b[2])
                   for a in now for b in self._recent if b is not a)


class HandTracker:
    def __init__(self, model_path: str, num_hands: int = 2):
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        # VIDEO mode needs strictly increasing timestamps.
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.recognize_for_video(image, timestamp_ms)
        hands = []
        gestures = result.gestures or [[] for _ in result.hand_landmarks]
        for landmarks, handedness, gesture in zip(result.hand_landmarks, result.handedness, gestures):
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)
            top = gesture[0] if gesture else None
            hands.append(Hand(pts, handedness[0].category_name, handedness[0].score,
                              top.category_name if top else "None", top.score if top else 0.0))
        return hands

    def close(self):
        self._landmarker.close()
