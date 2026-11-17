"""Object detection + instance segmentation with an Ultralytics YOLO segmentation model."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    label: str
    confidence: float
    box: np.ndarray      # x1, y1, x2, y2 in full-frame pixels
    contour: np.ndarray  # (N, 2) outline polygon in full-frame pixels
    track_id: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


# A dark sleeve reaching into the frame is often classified as an animal instead of "person"
# (station clip glasses_1369: the user's black-sleeved arm was "dog" at up to 0.86 and "horse").
ARM_LOOKALIKES = {"dog", "horse", "cow", "bear", "cat", "sheep", "elephant"}


def looks_like_arm(label: str, box: np.ndarray, frame_size: tuple[int, int], min_area: float = 0.04,
                   edge: float = 0.02) -> bool:
    """A big "animal" that enters from the bottom or a side edge of the frame is the user's arm."""
    if label not in ARM_LOOKALIKES:
        return False
    w, h = frame_size
    x1, y1, x2, y2 = box
    touches = x1 <= edge * w or x2 >= (1 - edge) * w or y2 >= (1 - edge) * h
    return touches and (x2 - x1) * (y2 - y1) >= min_area * w * h


class ObjectDetector:
    def __init__(self, weights: str, conf: float = 0.35, device: str | None = None,
                 imgsz: int = 640, ignore: tuple[str, ...] = ("person",), person_conf: float = 0.2):
        self.model = YOLO(weights)
        self.conf = conf
        if device is None:
            # Apple GPU is ~3x faster than CPU for the nano model on an M-series Mac.
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.imgsz = imgsz
        # The user's own hand/arm is detected as "person"; ignore it so it is never selected.
        # Ignored masks are kept in `last_ignored` (at a lower threshold) for the mask pointer.
        self.ignore = set(ignore)
        self.person_conf = person_conf
        self.last_ignored: list[Detection] = []

    def detect(self, frame_bgr: np.ndarray, crop: tuple[int, int, int, int] | None = None) -> list[Detection]:
        """Run the detector, optionally on a crop (x, y, w, h) used for digital zoom.

        Results are always returned in full-frame coordinates.
        """
        ox, oy = 0, 0
        image = frame_bgr
        if crop is not None:
            ox, oy, cw, ch = crop
            image = frame_bgr[oy:oy + ch, ox:ox + cw]
        self.last_ignored = []
        result = self.model.predict(image, conf=min(self.conf, self.person_conf), imgsz=self.imgsz,
                                    device=self.device, verbose=False, retina_masks=False)[0]
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)
        offset = np.array([ox, oy], dtype=np.float32)
        for box, conf, cls, poly in zip(boxes, confs, classes, polygons):
            label = self.model.names[cls]
            ignored = label in self.ignore
            if not ignored and conf < self.conf:
                continue
            if poly is None or len(poly) < 3:
                x1, y1, x2, y2 = box
                poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            det = Detection(label, float(conf), box + np.tile(offset, 2), np.asarray(poly, dtype=np.float32) + offset)
            if looks_like_arm(label, det.box, (frame_bgr.shape[1], frame_bgr.shape[0])):
                det.label, ignored = "person", True
            (self.last_ignored if ignored else detections).append(det)
        return detections


def contour_contains(contour: np.ndarray, point: np.ndarray) -> bool:
    return cv2.pointPolygonTest(contour.reshape(-1, 1, 2), (float(point[0]), float(point[1])), False) >= 0
