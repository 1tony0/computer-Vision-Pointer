"""One frame in, annotated frame + events out. Shared by the live app and the evaluation script."""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import overlay
from .arming import ArmController
from .describer import HELD_PROMPT, PROMPT, centered_crop, highlight, object_crop, region_crop, sharpness
from .detector import Detection, ObjectDetector
from .hand_mask import arm_candidates, find_mask_pointer
from .hands import HandTracker, TwoThumbsUp
from .hold import HoldEvent, HoldSelector
from .targeting import TargetingParams, angular_deviation, select_target
from .look import HeadMotion, LookGate, look_point, look_target
from .zoom import ZoomController


@dataclass
class FrameInfo:
    time_s: float
    hand_ms: float
    detect_ms: float
    total_ms: float
    num_hands: int
    num_objects: int
    pointing: bool
    mode: str
    target: str | None
    zoom: float
    source: str = "none"  # which method found the pointing hand: landmarks, mask or none
    event: HoldEvent | None = None      # a hold completed this frame
    described: HoldEvent | None = None  # the model's name for an event arrived this frame
    armed: bool = True
    just_armed: bool = False
    just_disarmed: bool = False
    extra: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, hand_model: str, det_weights: str, hold_s: float = 2.0, conf: float = 0.35,
                 device: str | None = None, detect_every: int = 1, imgsz: int = 640,
                 banner_s: float = 2.5, targeting: TargetingParams = TargetingParams(),
                 pointer_source: str = "hybrid", max_width_ratio: float = 1.6,
                 arming: ArmController | None = None, describer=None, describe_wait: bool = False,
                 hand: str = "right", max_finger_width: float = 0.06,
                 look: bool = True, look_s: float = 1.5):
        """pointer_source: "landmarks" (MediaPipe only), "mask" (arm segmentation only) or
        "hybrid" (MediaPipe first, segmentation when MediaPipe finds no hand).
        arming: two-thumbs-up gate; None means always armed.
        describer: optional vision-language model (pointer.describer.Describer) that names the
        target when a hold completes, and lets the user point at things YOLO can't detect.
        hand: which hand may point ("right", "left" or "any"); the other hand is ignored.
        max_finger_width: the mask fallback only counts a thin index finger (fraction of the
        frame diagonal), so fists and relaxed hands don't point.
        look: allow look mode (peace sign): name whatever stays in the middle of the view while
        the head is steady for look_s seconds (pointer/look.py). Two thumbs-up = point mode."""
        self.hands = HandTracker(hand_model)
        self.detector = ObjectDetector(det_weights, conf=conf, device=device, imgsz=imgsz)
        self.hold = HoldSelector(hold_s)
        self.hold_s, self.look_s, self.look_enabled = hold_s, look_s, look
        self.motion, self.look_gate = HeadMotion(), LookGate()
        self._looking = False
        self.zoom = ZoomController()
        self.targeting = targeting
        self.detect_every = max(1, detect_every)
        self.banner_s = banner_s
        self.pointer_source = pointer_source
        self.max_width_ratio = max_width_ratio
        self.hand = hand
        self.max_finger_width = max_finger_width
        self._frame_idx = 0
        self._detections = []
        self._last_event: HoldEvent | None = None
        self._banner_since = 0.0
        self.arming = arming or ArmController(always_armed=True)
        self.two_thumbs = TwoThumbsUp()
        self.describer = describer
        self.describe_wait = describe_wait
        self.last_target = None
        self._prev_tip, self._prev_tip_t = None, 0.0
        self._best = None  # (hold start, sharpness, crop box, crop image, mode) for the current hold
        self._point_since, self._point_last = None, float("-inf")
        self._held_t = float("-inf")   # last time the target was the other hand
        self._held_det = None          # (detection, time) of the last detected object in that hand
        self.held_memory_s = 0.6       # how long that is remembered when the ray jitters off it
        self.held_stick = 0.12         # x frame diagonal: fingertip this close to the hand keeps it
        self.held_cone_deg = 10.0      # a detected object in the other hand must lie this close to the ray
        self.steady_point_s = 0.3  # pointing must last this long to keep the system armed

    def process(self, frame: np.ndarray, time_s: float, draw: bool = True):
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        hands = self.hands.detect(frame, int(time_s * 1000))
        self.look_gate.update(self.motion.update(frame, time_s))
        t1 = time.perf_counter()

        just_armed = self.arming.update(self.two_thumbs.update(hands, time_s), time_s,
                                        thumbs_down=any(hd.is_thumbs_down() for hd in hands),
                                        peace=self.look_enabled and any(hd.is_peace() for hd in hands))
        armed = self.arming.is_armed(time_s)
        look_mode = self.look_enabled and self.arming.mode == "look"
        self.zoom.update(hands, time_s, (w, h), enabled=armed)
        crop = self.zoom.crop((w, h))

        detect_ms = 0.0
        if self._frame_idx % self.detect_every == 0:
            if crop is not None:
                # The arm is usually outside the zoomed crop: find it in the full frame first.
                self.detector.detect(frame)
                people = self.detector.last_ignored
                self._detections = self.detector.detect(frame, crop)
                self.detector.last_ignored = people
            else:
                self._detections = self.detector.detect(frame)
            detect_ms = (time.perf_counter() - t1) * 1000
        self._frame_idx += 1

        # Only the chosen hand points. The user's arms are the person masks attached to the frame
        # edge; each one's side follows from where it enters the frame.
        arms = arm_candidates([d.contour for d in self.detector.last_ignored], (w, h))
        mine = [hd for hd in hands if self._on_side(hd, arms, w, h)]
        # Zoom gestures are not pointing: ignore the hands while an OK sign or zoom is going on.
        zooming = self.zoom.active or any(hd.is_ok_sign() for hd in hands)
        pointer, source, arm = None, "none", None
        if self.pointer_source != "mask" and not zooming:
            pointer = next((hd for hd in mine if hd.is_pointing()), None)
            source = "landmarks" if pointer is not None else "none"
        # Seen from behind (typical first-person pointing) MediaPipe often finds the hand but places
        # the fingers wrongly, so it reports "not pointing". Fall back to the arm mask in that case
        # too, unless MediaPipe confidently recognises some other gesture (fist, palm, thumbs-up...)
        # on the pointing hand.
        # A Thumb_Down label that fails the landmark check is usually a finger pointing down.
        other_gesture = any(hd.gesture not in ("None", "Pointing_Up", "Thumb_Down") and hd.gesture_score >= 0.6
                            or hd.is_thumbs_down() for hd in mine)
        use_mask = self.pointer_source == "mask" or (self.pointer_source == "hybrid" and not other_gesture)
        if pointer is None and use_mask and not zooming:
            arm = find_mask_pointer([], (w, h), prev_tip=self._prev_tip, hand=self.hand, arms=arms)
            if arm is not None and arm.is_pointing(self.max_width_ratio, self.max_finger_width):
                pointer, source = arm, "mask"
        target, mode, origin, ray_end = None, "none", None, None
        if pointer is not None:
            origin, direction, ratio = pointer.pointing_ray()
            ray_end = origin + direction * 0.15 * float(np.hypot(w, h))
            self._prev_tip, self._prev_tip_t = origin, time_s
            # Only steady pointing keeps it armed: stray one-frame detections while the hands are
            # down must not stop it from switching off.
            if self._point_since is None or time_s - self._point_last > 0.15:
                self._point_since = time_s
            self._point_last = time_s
            if time_s - self._point_since >= self.steady_point_s:
                self.arming.keep_alive(time_s)
        elif time_s - self._prev_tip_t > 0.5:
            self._prev_tip = None  # lost the hand for a while: don't bias the next choice
        if pointer is not None and armed and not look_mode:
            others = self._other_hands(pointer, arm, origin, arms, w)
            result = select_target(self._detections + others, origin, direction, ratio, pointer.size,
                                   (w, h), self.targeting)
            target, mode, ray_end = result.detection, result.mode, result.ray_end
            if target is not None and (target.label == "held" or any(self._in_hand(target, o, w, h, origin, direction) for o in others)):
                self._held_t = time_s  # the other hand, or a detected object in it (e.g. a phone)
            elif time_s - self._held_t < self.held_memory_s:
                # The finger direction jitters; while the ray still passes close to the other hand,
                # keep treating it as pointing at what that hand holds (not the laptop behind it).
                diag = float(np.hypot(w, h))
                ray = origin[None, :] + np.linspace(0, self.targeting.held_frac * diag, 12)[:, None] * direction[None, :]
                for o in others:
                    poly = o.contour.reshape(-1, 1, 2).astype(np.float32)
                    gaps = [-cv2.pointPolygonTest(poly, (float(x), float(y)), True) for x, y in ray]
                    k = int(np.argmin(gaps))
                    if gaps[k] <= self.held_stick * diag:
                        target, mode, ray_end, self._held_t = o, "ray", ray[k].copy(), time_s
                        break
            if target is not None and target.label == "held":
                # The ray reaches the user's other hand first: they're pointing at what it holds.
                # Centre the crop on the part of that hand nearest the ray, where the object is.
                c = target.contour
                nearest = c[int(np.argmin(np.linalg.norm(c - ray_end[None, :], axis=1)))]
                held_det = [d for d in self._detections if self._in_hand(d, target, w, h, origin, direction)]
            else:
                held_det = []
            if held_det:
                # The detector recognises the held object itself: use it (better name and outline).
                target = min(held_det, key=lambda d: np.linalg.norm(d.center - nearest))
                mode, ray_end = "ray", nearest
                self._held_det = (target, time_s)
            elif target is not None and target.label == "held" and self._held_det is not None \
                    and time_s - self._held_det[1] < self.held_memory_s:
                # The detector missed the held object this frame: keep its last detection so the
                # hold timer doesn't restart.
                target, mode = self._held_det[0], "ray"
            elif target is not None and target.label == "held":
                box = centered_crop(nearest, (w, h))
                x1, y1, x2, y2 = box
                rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                target, mode = Detection("object", 0.0, box, rect), "held"
            if target is None and self.describer is not None:
                # Nothing the detector knows: let the language model look just ahead of the finger.
                box = region_crop(origin, direction, (w, h))
                x1, y1, x2, y2 = box
                rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                target, mode = Detection("object", 0.0, box, rect), "region"

        # Look mode: no hand pointing, so name what stays in the middle of the view.
        # Look mode (peace sign): name what stays in the middle of the view; hands are ignored.
        looking = look_mode and armed and not zooming
        if looking:
            self.arming.keep_alive(time_s)  # stays on until a thumbs-down
            target, mode = None, "none"
            if self.motion.steady and not self.look_gate.blocked:
                target, mode = look_target(self._detections, look_point((w, h), crop), (w, h))
        self._looking = looking
        self.hold.hold_s = self.look_s if looking else self.hold_s

        self.last_target = target
        event = self.hold.update(target, time_s)
        # Only frames of the object being held count (not a pending switch candidate).
        if self.describer is not None and target is not None and target is self.hold.current \
                and (event is not None or not self.hold.fired):
            self._keep_sharpest(frame, target, mode)
        described = None
        if event is not None:
            self._last_event, self._banner_since = event, time_s
            if looking:
                self.look_gate.announced()
            if self.describer is not None and self._best is not None:
                _, _, event.crop_box, crop_img, best_mode = self._best
                prompt = HELD_PROMPT if best_mode == "held" else PROMPT
                answer = self.describer.submit(event, crop_img, wait=self.describe_wait, prompt=prompt)
                if answer is not None:
                    (event.description, event.vlm_score), described = answer, event
        if self.describer is not None:
            for ev, answer in self.describer.poll():
                (ev.description, ev.vlm_score), described = answer, ev
                if ev is self._last_event:
                    self._banner_since = time_s
        total_ms = (time.perf_counter() - t0) * 1000

        info = FrameInfo(time_s, (t1 - t0) * 1000, detect_ms, total_ms, len(hands), len(self._detections),
                         pointer is not None, mode, target.label if target else None, self.zoom.zoom,
                         source, event, described, armed, just_armed, self.arming.just_disarmed)
        if not draw:
            return None, info

        view = overlay.View(frame, crop)
        for det in self._detections:
            if det is not target:
                overlay.draw_outline(view, det.contour, overlay.GRAY, 1)
        for hd in hands:
            overlay.draw_hand(view, hd, hd is pointer)
        if arm is not None:
            overlay.draw_arm(view, arm, arm is pointer)
        if pointer is not None and armed and not look_mode:
            overlay.draw_pointing(view, origin, ray_end)
        if looking:
            overlay.draw_reticle(view, look_point((w, h), crop), self.motion.steady and not self.look_gate.blocked)
        if target is not None:
            overlay.draw_outline(view, target.contour, overlay.RED, 3)
            x1, y1 = view.pt(target.box[:2])
            chip = {"region": "looking here" if looking else "pointing here", "held": "held object"}.get(mode, f"{target.label} {target.confidence:.2f}")
            overlay.draw_label(view.image, chip, (x1, y1 - 6), overlay.RED)
            if not self.hold.fired:
                overlay.draw_progress(view, target, self.hold.progress(time_s))
        if self._last_event is not None:
            age = time_s - self._banner_since
            if age < self.banner_s:
                title, subtitle = banner_text(self._last_event, self.describer is not None)
                overlay.draw_banner(view.image, title, subtitle, alpha=min(1.0, (self.banner_s - age) / 0.5))
        overlay.draw_arm_badge(view.image, armed, self.arming.remaining(time_s), self.arming.window_s,
                               self.arming.always_armed, self.arming.mode if self.look_enabled else "point")
        fps = 1000 / total_ms if total_ms > 0 else 0
        overlay.draw_hud(view.image, [f"{fps:4.1f} FPS  hand {info.hand_ms:.0f} ms  det {detect_ms:.0f} ms",
                                      f"zoom {self.zoom.zoom:.1f}x  mode {mode}  hand {source}  head {self.motion.speed:.2f}"])
        return view.image, info

    def _in_hand(self, det, hand, w, h, origin, direction) -> bool:
        """Is this a small object touching the other hand (probably held) that the finger points
        towards? The direction check stops e.g. a mug on the desk just below the hand counting."""
        if angular_deviation(det.contour, origin, direction) > self.held_cone_deg:
            return False
        if det.label == "held":
            return False
        x1, y1, x2, y2 = det.box
        if (x2 - x1) * (y2 - y1) > 0.08 * w * h:
            return False  # big things (laptop, table) are behind the hand, not in it
        gap = -cv2.pointPolygonTest(hand.contour.reshape(-1, 1, 2).astype(np.float32),
                                    (float(det.center[0]), float(det.center[1])), True)
        return gap <= 0.04 * float(np.hypot(w, h))

    def _on_side(self, hand, arms, w, h) -> bool:
        """Is this MediaPipe hand the one allowed to point? Its left/right label is unreliable when
        the hand is seen from behind, so use the arm its wrist lies on (or, with no arm mask, which
        half of the frame the wrist is in)."""
        if self.hand == "any":
            return True
        wrist = (float(hand.points[0][0]), float(hand.points[0][1]))
        tol = self.targeting.touch_frac * float(np.hypot(w, h))
        for a in arms:
            if cv2.pointPolygonTest(a.contour.reshape(-1, 1, 2).astype(np.float32), wrist, True) >= -tol:
                return a.side(w) == self.hand
        return (wrist[0] >= 0.5 * w) == (self.hand == "right")

    def _other_hands(self, pointer, arm, origin, arms, w) -> list[Detection]:
        """The user's other hand(s) as ray obstacles, so pointing at a hand-held object works.
        Needs the language model, since the held object itself is usually not detected."""
        if self.describer is None:
            return []
        if self.hand != "any":
            # Only the opposite arm. The segmenter sometimes splits the pointing arm into several
            # masks, and those pieces must not count as "the other hand".
            return [Detection("held", 1.0, np.concatenate([a.contour.min(0), a.contour.max(0)]).astype(np.float32),
                              a.contour) for a in arms if a.side(w) != self.hand]
        others = []
        for person in self.detector.last_ignored:
            if pointer is arm and person.contour is arm.contour:
                continue  # the pointing arm itself (mask pointer)
            if pointer is not arm:
                # Landmark pointer: its own arm is the mask containing its wrist. (Testing the
                # fingertip instead would also drop the other hand when the finger nearly touches it.)
                wrist = (float(pointer.points[0][0]), float(pointer.points[0][1]))
                gap = -cv2.pointPolygonTest(person.contour.reshape(-1, 1, 2), wrist, True)
                if gap <= self.targeting.touch_frac * float(np.hypot(*self._frame_size)):
                    continue
            others.append(Detection("held", person.confidence, person.box, person.contour))
        return others

    def _keep_sharpest(self, frame, target, mode):
        h, w = frame.shape[:2]
        box = object_crop(target.box, (w, h)) if mode not in ("region", "held") else target.box
        x1, y1, x2, y2 = box.astype(int)
        crop = frame[y1:y2, x1:x2]
        score = sharpness(crop)
        key = self.hold.started_s
        if self._best is None or self._best[0] != key or score > self._best[1]:
            contour = target.contour if mode not in ("region", "held") else None
            self._best = (key, score, box, highlight(crop, contour, (x1, y1)), mode)

    def close(self):
        self.hands.close()
        if self.describer is not None:
            self.describer.close()


def with_article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


# Words the language model may use for a COCO class (the detector's label).
SYNONYMS = {
    "tv": {"tv", "television", "monitor", "screen", "display"},
    "cell phone": {"phone", "smartphone", "iphone", "cellphone", "mobile"},
    "cup": {"cup", "mug", "glass", "tumbler"},
    "laptop": {"laptop", "computer", "notebook", "macbook"},
    "dining table": {"table", "desk"},
    "couch": {"couch", "sofa"},
    "potted plant": {"plant"},
    "remote": {"remote", "controller"},
    "car": {"car", "suv", "vehicle", "sedan", "van", "minivan", "jeep", "hatchback", "coupe"},
    "truck": {"truck", "pickup", "lorry", "suv", "vehicle"},
}
SURE_SCORE = 0.6       # model confidence needed to name something without the detector's backing
DETECTOR_SURE = 0.5    # detector confidence needed when there is no language model


def labels_agree(label: str, description: str) -> bool:
    """Same kind of object? Compares the description's main noun (its last word: "computer
    monitor" -> monitor, but "mouse pad" -> pad) with the detector label and its synonyms."""
    words = description.lower().split(" that says")[0].replace("-", " ").split()
    if not words:
        return False
    head = words[-1].rstrip("s") if len(words[-1]) > 3 else words[-1]
    names = SYNONYMS.get(label, set()) | set(label.split())
    return head in names or words[-1] in names


def is_sure(event: HoldEvent) -> bool:
    """Confident when the detector and the language model name the same thing, or (for objects
    the detector doesn't know) when the language model itself is confident."""
    if not event.description:
        return event.label != "object" and event.confidence >= DETECTOR_SURE
    if event.label != "object" and labels_agree(event.label, event.description):
        return True
    return (event.vlm_score or 0.0) >= SURE_SCORE and event.label == "object"


def banner_text(event: HoldEvent, uses_model: bool) -> tuple[str, str | None]:
    """(title, subtitle). The title is also what gets spoken."""
    detected = None if event.label == "object" else f"detector: {event.label} {event.confidence:.2f}"
    name = event.description or (None if uses_model else event.label)
    if name and name.split()[0] in ("no", "nothing", "none", "empty"):
        return "I'm not sure what that is", detected  # the model saw nothing in particular
    if name is None:
        return ("Looking..." if event.label == "object" else f"{event.label.capitalize()}..."), detected
    if is_sure(event):
        return f"This is {with_article(name)}", detected
    return f"I don't know, but it looks like {with_article(name)}", detected
