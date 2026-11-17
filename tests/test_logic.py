import cv2
import numpy as np
import pytest

from pointer.detector import Detection
from pointer.hands import Hand
from pointer.hold import HoldSelector
from pointer.targeting import select_target
from pointer.zoom import ZoomController

FRAME = (640, 480)


def rect(label, x1, y1, x2, y2, conf=0.9):
    contour = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
    return Detection(label, conf, np.array([x1, y1, x2, y2], np.float32), contour)


UP = np.array([0.0, -1.0])


def test_ray_prefers_smaller_object_on_a_surface():
    table = rect("dining table", 0, 150, 640, 400)
    cup = rect("cup", 300, 180, 340, 230)
    res = select_target([table, cup], np.array([320.0, 420.0]), UP, 1.0, 80, FRAME)
    assert res.detection is cup and res.mode == "ray"


def test_ray_selects_surface_when_nothing_on_it():
    table = rect("dining table", 0, 150, 640, 400)
    cup = rect("cup", 500, 180, 540, 230)
    res = select_target([table, cup], np.array([320.0, 420.0]), UP, 1.0, 80, FRAME)
    assert res.detection is table


def test_ray_prefers_nearer_object_that_occludes_another():
    bottle = rect("bottle", 280, 150, 360, 300)
    cup = rect("cup", 300, 60, 340, 120)
    res = select_target([cup, bottle], np.array([320.0, 420.0]), UP, 1.0, 80, FRAME)
    assert res.detection is bottle


def test_touched_object_wins():
    mug = rect("cup", 330, 380, 380, 440)          # fingertip right at its edge
    laptop = rect("laptop", 250, 50, 400, 200)
    res = select_target([laptop, mug], np.array([325.0, 400.0]), UP, 1.0, 80, FRAME)
    assert res.detection is mug


def test_cone_fallback_and_limit():
    near_miss = rect("bottle", 335, 50, 360, 100)   # ~5 deg off the ray
    far_off = rect("book", 500, 50, 560, 100)
    res = select_target([near_miss, far_off], np.array([320.0, 400.0]), UP, 1.0, 80, FRAME)
    assert res.detection is near_miss and res.mode == "cone"
    res = select_target([far_off], np.array([320.0, 400.0]), UP, 1.0, 80, FRAME)
    assert res.detection is None


def test_objects_behind_the_finger_are_ignored():
    behind = rect("cup", 300, 440, 340, 470)
    res = select_target([behind], np.array([320.0, 400.0]), UP, 1.0, 80, FRAME)
    assert res.detection is None


def test_foreshortened_finger_uses_tip():
    mouse = rect("mouse", 300, 300, 360, 340)
    laptop = rect("laptop", 100, 50, 600, 360)
    res = select_target([mouse, laptop], np.array([330.0, 320.0]), UP, 0.1, 80, FRAME)
    assert res.detection is mouse and res.mode == "tip"


def test_hold_fires_once_after_hold_time():
    sel = HoldSelector(hold_s=1.5)
    cup = rect("cup", 100, 100, 150, 150)
    events = [sel.update(cup, t / 30) for t in range(0, 90)]
    fired = [e for e in events if e is not None]
    assert len(fired) == 1 and fired[0].label == "cup"
    assert fired[0].time_s == pytest.approx(1.5, abs=1 / 30)


def test_hold_survives_short_dropout_but_resets_on_long_one():
    cup = rect("cup", 100, 100, 150, 150)
    sel = HoldSelector(hold_s=1.5, grace_s=0.25)
    sel.update(cup, 0.0)
    sel.update(None, 0.1)
    assert sel.update(cup, 1.6) is not None  # 0.1 s gap < grace: timer kept
    sel = HoldSelector(hold_s=1.5, grace_s=0.25)
    sel.update(cup, 0.0)
    sel.update(cup, 0.5)
    sel.update(None, 1.0)                     # 0.5 s gap > grace: timer reset
    assert sel.update(cup, 1.6) is None


def test_hold_resets_when_target_changes():
    sel = HoldSelector(hold_s=1.5, switch_s=0.2)
    sel.update(rect("cup", 100, 100, 150, 150), 0.0)
    assert sel.update(rect("book", 400, 100, 450, 150), 1.0) is None
    assert sel.update(rect("book", 400, 100, 450, 150), 1.3) is None  # switch confirmed, started 1.0
    assert sel.update(rect("book", 400, 100, 450, 150), 2.4) is None
    assert sel.update(rect("book", 400, 100, 450, 150), 2.5) is not None


def test_one_frame_flicker_does_not_restart_hold():
    cup, book = rect("cup", 100, 100, 150, 150), rect("book", 400, 100, 450, 150)
    sel = HoldSelector(hold_s=1.5, switch_s=0.2)
    fired = []
    for i in range(60):
        t = i / 30
        fired.append(sel.update(book if i == 20 else cup, t))
    events = [e for e in fired if e]
    assert len(events) == 1 and events[0].label == "cup" and events[0].time_s == pytest.approx(1.5, abs=1 / 30)


def make_hand(extended: set[str], pinch: float = 1.5):
    """Synthetic upright hand: wrist at origin, fingers point up (negative y)."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = [0, 0]
    pts[9] = [0, -100]  # middle MCP -> hand size 100
    columns = {"index": (5, 6, 7, 8, -30), "middle": (9, 10, 11, 12, 0),
               "ring": (13, 14, 15, 16, 25), "pinky": (17, 18, 19, 20, 50)}
    for name, (mcp, pip, dip, tip, x) in columns.items():
        pts[mcp] = [x, -100]
        pts[pip] = [x, -140]
        if name in extended:
            pts[dip], pts[tip] = [x, -170], [x, -200]
        else:
            pts[dip], pts[tip] = [x, -120], [x, -105]
    pts[4] = pts[8] + [pinch * 100, 0]
    return Hand(pts, "Right", 1.0)


def test_gestures():
    assert make_hand({"index"}).is_pointing()
    assert not make_hand({"index", "middle"}).is_pointing()
    assert make_hand({"middle", "ring", "pinky"}, pinch=0.3).is_ok_sign()
    # A hand holding something: three fingers open but thumb and index apart is not an OK sign.
    assert not make_hand({"middle", "ring", "pinky"}, pinch=0.6).is_ok_sign()
    assert make_hand({"index", "middle", "ring", "pinky"}).is_open_palm()
    origin, direction, ratio = make_hand({"index"}).pointing_ray()
    assert np.allclose(direction, [0, -1]) and ratio == pytest.approx(1.0)


def test_zoom_follows_pinch_spread_and_palm_resets():
    z = ZoomController(smoothing=1.0, confirm_s=0.3, gain=2.0)
    ok = make_hand({"middle", "ring", "pinky"}, pinch=0.3)
    for t in (0.0, 0.15, 0.35):                        # held 0.35 s: session starts
        z.update([ok], t, FRAME)
    z.update([make_hand({"middle", "ring", "pinky"}, pinch=0.8)], 0.4, FRAME)
    assert z.active and z.zoom == pytest.approx(2.0)   # 1 + 2 x (0.8 - 0.3)
    x, y, w, h = z.crop(FRAME)
    assert (w, h) == (320, 240) and 0 <= x <= 320 and 0 <= y <= 240
    palm = make_hand({"index", "middle", "ring", "pinky"}, pinch=1.3)
    for t in (0.5, 0.8, 1.2):                          # opening fully = zoom in, not a reset
        z.update([palm], t, FRAME)
    assert z.zoom == pytest.approx(3.0)
    z.update([make_hand({"middle", "ring", "pinky"}, pinch=0.4)], 1.25, FRAME)
    assert z.zoom == pytest.approx(3.0)                # closing again does not zoom out
    z.update([], 1.3, FRAME)
    z.update([], 1.7, FRAME)                           # hand lowered: session ends, zoom stays
    assert not z.active and z.zoom == pytest.approx(3.0)
    for t in (3.0, 3.7):                               # later, open palm for 0.7 s: reset
        z.update([palm], t, FRAME)
    assert z.zoom == 1.0 and z.crop(FRAME) is None


def test_zoom_ok_sign_survives_one_frame_dropout():
    ok = make_hand({"middle", "ring", "pinky"}, pinch=0.3)
    z = ZoomController(smoothing=1.0, confirm_s=0.3, grace_s=0.2)
    for t in (0.0, 0.1):
        z.update([ok], t, FRAME)
    z.update([], 0.15, FRAME)                 # one-frame dropout
    z.update([ok], 0.2, FRAME)
    z.update([ok], 0.31, FRAME)
    assert z.active


def test_zoom_ignores_brief_ok_sign_and_needs_arming():
    ok = make_hand({"middle", "ring", "pinky"}, pinch=0.3)
    wide = make_hand({"middle", "ring", "pinky"}, pinch=0.6)
    z = ZoomController(smoothing=1.0, confirm_s=0.3)
    z.update([ok], 0.0, FRAME)
    z.update([wide], 0.1, FRAME)          # one misread frame: no zoom
    assert z.zoom == 1.0
    z = ZoomController(smoothing=1.0, confirm_s=0.3)
    for t in (0.0, 0.35):
        z.update([ok], t, FRAME, enabled=False)
    z.update([wide], 0.4, FRAME, enabled=False)
    assert z.zoom == 1.0                  # disarmed: never zooms


def test_mask_pointer_prefers_finger_and_tracks_previous_tip():
    from pointer.hand_mask import find_mask_pointer
    size = (640, 480)
    # Right arm: thin finger sticking up from a forearm at the bottom-right edge.
    finger = np.array([[440, 479], [440, 300], [470, 300], [470, 200], [485, 200], [485, 300],
                       [520, 300], [520, 479]], np.float32)
    # Left fist: a blob at the bottom-left edge.
    fist = cv2.ellipse2Poly((120, 430), (90, 70), 0, 0, 360, 10).astype(np.float32)
    best = find_mask_pointer([fist, finger], size)
    assert best is not None and best.contour is finger
    assert np.allclose(best.direction, [0, -1], atol=0.25)  # within ~15 degrees
    # With the previous fingertip next to the fist, a small finger-score gap is outweighed.
    near_fist = find_mask_pointer([fist, finger], size, prev_tip=np.array([120.0, 360.0]), track_weight=40.0)
    assert near_fist.contour is fist


def test_mask_pointer_only_uses_chosen_arm_and_rejects_fist():
    from pointer.hand_mask import find_mask_pointer
    size = (640, 480)
    # Left arm with a pointing finger, entering at the bottom-left.
    left_finger = np.array([[100, 479], [100, 300], [130, 300], [130, 200], [145, 200], [145, 300],
                            [180, 300], [180, 479]], np.float32)
    # Right arm ending in a fist, entering at the bottom-right.
    right_fist = np.vstack([np.array([[560, 479], [480, 479]], np.float32),
                            cv2.ellipse2Poly((520, 330), (60, 60), 0, 180, 360, 10).astype(np.float32)])
    assert find_mask_pointer([left_finger, right_fist], size, hand="right").contour is right_fist
    assert find_mask_pointer([left_finger], size, hand="right") is None
    assert find_mask_pointer([left_finger], size, hand="left").is_pointing(1.6)
    assert not find_mask_pointer([right_fist], size, hand="right").is_pointing(1.6)


def test_other_hand_in_front_of_finger_wins_over_touched_background():
    laptop = rect("laptop", 100, 50, 600, 400)        # fingertip is over the laptop
    other_hand = rect("held", 300, 200, 360, 260)     # the user's other hand, just ahead
    res = select_target([laptop, other_hand], np.array([330.0, 330.0]), UP, 1.0, 80, FRAME)
    assert res.detection is other_hand
    far_hand = rect("held", 300, 0, 360, 20)          # far along the ray: normal rules apply
    res = select_target([laptop, far_hand], np.array([330.0, 330.0]), UP, 1.0, 80, FRAME)
    assert res.detection is laptop
    beside = rect("held", 360, 200, 420, 260)         # ray passes just beside the other hand
    res = select_target([laptop, beside], np.array([330.0, 330.0]), UP, 1.0, 80, FRAME)
    assert res.detection is beside


def test_arm_sides_are_relative_when_both_arms_visible():
    from pointer.hand_mask import arm_candidates
    size = (640, 480)
    # Left arm reaching across: enters at the bottom near the middle (x ~ 330).
    left = np.array([[300, 479], [300, 300], [330, 300], [330, 200], [345, 200], [345, 300],
                     [360, 300], [360, 479]], np.float32)
    right = np.array([[560, 479], [560, 300], [590, 300], [590, 200], [605, 200], [605, 300],
                      [620, 300], [620, 479]], np.float32)
    sides = {id(a.contour): a.side(size[0]) for a in arm_candidates([left, right], size)}
    assert sides[id(left)] == "left" and sides[id(right)] == "right"
    (alone,) = arm_candidates([left], size)
    assert alone.side(size[0]) == "right"   # one arm only: absolute position decides


def test_dark_sleeve_classified_as_animal_counts_as_arm():
    from pointer.detector import looks_like_arm
    frame = (1520, 2032)
    assert looks_like_arm("dog", np.array([835, 871, 1520, 1995.0]), frame)       # arm from the right edge
    assert not looks_like_arm("dog", np.array([600, 600, 900, 900.0]), frame)     # a dog in the scene
    assert not looks_like_arm("dog", np.array([1450, 1900, 1520, 2032.0]), frame)  # too small
    assert not looks_like_arm("cup", np.array([835, 871, 1520, 1995.0]), frame)


def test_peace_sign_from_landmarks():
    hand = make_hand({"index", "middle"})
    hand.points[8] = [-60, -200]   # index and middle spread into a V
    hand.points[12] = [20, -210]
    assert hand.is_peace()
    assert not make_hand({"index"}).is_peace()          # pointing
    assert not make_hand({"index", "middle", "ring", "pinky"}).is_peace()  # open palm
