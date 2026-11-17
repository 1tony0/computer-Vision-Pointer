import cv2
import numpy as np
import pytest

from pointer.arming import ArmController
from pointer.describer import clean_answer, highlight, object_crop, region_crop, sharpness
from pointer.hold import HoldEvent
from pointer.pipeline import banner_text

FPS = 30


def feed(ctrl, pattern, start=0.0):
    """pattern: list of booleans, one per frame. Returns frames (times) on which it armed."""
    armed_at = []
    for i, thumb in enumerate(pattern):
        t = start + i / FPS
        if ctrl.update(thumb, t):
            armed_at.append(t)
    return armed_at


def test_not_armed_until_thumbs_up_is_held():
    ctrl = ArmController(window_s=10, confirm_s=0.4)
    assert not ctrl.is_armed(0.0)
    assert feed(ctrl, [True] * 6) == []          # 0.17 s: too short
    ctrl = ArmController(window_s=10, confirm_s=0.4)
    armed = feed(ctrl, [True] * 15)              # 0.5 s
    assert armed == [pytest.approx(0.4, abs=1 / FPS)]
    assert ctrl.is_armed(5.0) and not ctrl.is_armed(10.5)


def test_short_dropout_does_not_reset_confirmation():
    ctrl = ArmController(window_s=10, confirm_s=0.4, grace_s=0.2)
    pattern = [True] * 6 + [False] * 3 + [True] * 6   # 0.1 s gap
    assert len(feed(ctrl, pattern)) == 1


def test_thumbs_up_again_extends_window():
    ctrl = ArmController(window_s=10, confirm_s=0.4)
    feed(ctrl, [True] * 15)
    feed(ctrl, [False] * 30, start=1.0)
    feed(ctrl, [True] * 15, start=8.0)           # re-arm at ~8.4 s
    assert ctrl.is_armed(15.0)


def test_stays_armed_while_pointing_and_times_out_when_idle():
    ctrl = ArmController(window_s=15, confirm_s=0.4)
    feed(ctrl, [True] * 15)                      # armed at ~0.4 s
    for t in range(1, 40):                       # pointing every second until 39 s
        ctrl.keep_alive(float(t))
    assert ctrl.is_armed(50.0)                   # 11 s after the last pointing
    assert not ctrl.is_armed(54.5)               # 15.5 s idle: disarmed


def test_pointing_does_not_arm_by_itself():
    ctrl = ArmController(window_s=15)
    for t in range(10):
        ctrl.keep_alive(float(t))
    assert not ctrl.is_armed(5.0)


def test_thumbs_down_disarms_but_not_a_brief_one():
    ctrl = ArmController(window_s=5, confirm_s=0.4)
    feed(ctrl, [True] * 15)                      # armed
    for i in range(6):                           # 0.2 s thumbs-down: ignored
        ctrl.update(False, 1.0 + i / FPS, thumbs_down=True)
    assert ctrl.is_armed(1.5)
    for i in range(15):                          # 0.5 s thumbs-down: off
        ctrl.update(False, 2.0 + i / FPS, thumbs_down=True)
    assert not ctrl.is_armed(2.6)
    ctrl.keep_alive(3.0)                         # pointing doesn't re-arm it
    assert not ctrl.is_armed(3.1)


def test_disarms_after_idle_window():
    ctrl = ArmController(window_s=5, confirm_s=0.4)
    feed(ctrl, [True] * 15)
    ctrl.keep_alive(2.0)
    assert ctrl.is_armed(6.9) and not ctrl.is_armed(7.1)


def test_thumbs_down_needs_thumb_below_hand():
    from pointer.hands import Hand
    pts = np.zeros((21, 2), np.float32)
    pts[9] = [0, -100]                           # hand size 100, fingers up
    pts[5:21] = [[0, -100 - 5 * k] for k in range(16)]
    pts[4] = [0, 60]                             # thumb tip well below the wrist
    assert Hand(pts, "Right", 1.0, "Thumb_Down", 0.8).is_thumbs_down()
    pts[4] = [0, -50]                            # thumb not lowest: e.g. a finger pointing down
    assert not Hand(pts, "Right", 1.0, "Thumb_Down", 0.8).is_thumbs_down()


def test_two_thumbs_up_needs_two_separate_hands():
    from pointer.hands import Hand, two_thumbs_up
    def hand(x, gesture="Thumb_Up"):
        pts = np.zeros((21, 2), np.float32)
        pts[:, 0] = x
        pts[9] = [x, -100]                       # hand size 100
        return Hand(pts, "Right", 1.0, gesture, 0.8)
    assert two_thumbs_up([hand(0), hand(300)])
    assert not two_thumbs_up([hand(0)])                    # one thumbs-up: not enough
    assert not two_thumbs_up([hand(0), hand(30)])          # same hand detected twice
    assert not two_thumbs_up([hand(0), hand(300, "Pointing_Up")])


def test_two_thumbs_up_tolerates_flicker_between_hands():
    from pointer.hands import Hand, TwoThumbsUp
    def hand(x, gesture="Thumb_Up"):
        pts = np.zeros((21, 2), np.float32)
        pts[:, 0] = x
        pts[9] = [x, -100]
        return Hand(pts, "Right", 1.0, gesture, 0.8)
    tt = TwoThumbsUp(memory_s=0.5)
    assert tt.update([hand(0), hand(300)], 0.0)
    assert tt.update([hand(0), hand(300, "None")], 0.1)   # right hand flickers: still both
    assert not tt.update([hand(0)], 0.7)                    # right gone for > 0.5 s
    assert not TwoThumbsUp().update([hand(0)], 0.0)


def test_labels_agree_uses_main_noun():
    from pointer.pipeline import labels_agree
    assert labels_agree("tv", "computer monitor")
    assert labels_agree("bottle", "yeti water bottle")
    assert labels_agree("cup", "coffee mugs")
    assert labels_agree("cell phone", "phone")
    assert not labels_agree("mouse", "mouse pad")
    assert not labels_agree("laptop", "keyboard")


def test_always_armed():
    assert ArmController(always_armed=True).is_armed(0.0)


def test_crops_stay_inside_frame_and_are_square():
    w, h = 1080, 1920
    for box in ([0, 0, 50, 900], [1000, 1800, 1080, 1920], [300, 400, 700, 600]):
        x1, y1, x2, y2 = object_crop(np.array(box, np.float32), (w, h))
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
        assert x2 - x1 == pytest.approx(y2 - y1, abs=1)
    x1, y1, x2, y2 = region_crop(np.array([540.0, 1500.0]), np.array([0.0, -1.0]), (w, h))
    assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and y1 < 1500


def test_sharpness_detects_blur():
    rng = np.random.default_rng(0)
    img = (rng.random((200, 200, 3)) * 255).astype(np.uint8)
    assert sharpness(img) > 5 * sharpness(cv2.GaussianBlur(img, (21, 21), 8))


def test_highlight_keeps_object_and_fades_background():
    crop = np.full((100, 100, 3), 255, np.uint8)
    contour = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], np.float32) + [10, 20]
    out = highlight(crop, contour, (10, 20))
    assert (out[50, 50] == 255).all() and out[5, 5, 0] < 200


def test_clean_answer():
    assert clean_answer("A coffee mug.") == "coffee mug"
    assert clean_answer("The main object in this image is a laptop.\nMore text") == "laptop"
    assert clean_answer(" Water bottle ") == "water bottle"


def test_banner_text():
    e = HoldEvent("cup", 0.82, 1.5, 0.0)
    assert banner_text(e, False)[0] == "This is a cup"
    assert banner_text(e, True)[0] == "Cup..."
    e.description, e.vlm_score = "orange mug", 0.3        # detector and model agree: sure
    assert banner_text(e, True) == ("This is an orange mug", "detector: cup 0.82")
    e.description, e.vlm_score = "keyboard", 0.9          # they disagree: not sure
    assert banner_text(e, True)[0] == "I don't know, but it looks like a keyboard"
    region = HoldEvent("object", 0.0, 1.5, 0.0)
    assert banner_text(region, True) == ("Looking...", None)
    region.description, region.vlm_score = "glasses case", 0.4
    assert banner_text(region, True)[0] == "I don't know, but it looks like a glasses case"
    region.vlm_score = 0.8                                 # no detector, but a confident model
    assert banner_text(region, True)[0] == "This is a glasses case"
    weak = HoldEvent("book", 0.3, 1.5, 0.0)               # detector only, low confidence
    assert banner_text(weak, False)[0] == "I don't know, but it looks like a book"


def test_peace_sign_starts_look_mode_and_thumbs_up_switches_back():
    ctrl = ArmController(window_s=5, confirm_s=0.4)
    fired = [ctrl.update(False, t / 10, peace=True) for t in range(6)]
    assert any(fired) and ctrl.mode == "look" and ctrl.is_armed(0.5)
    fired = [ctrl.update(True, 1 + t / 10) for t in range(6)]
    assert fired[-1] and ctrl.mode == "point" and ctrl.just_switched  # look -> point
    ctrl.update(False, 2.0, thumbs_down=True)
    for t in range(1, 6):
        ctrl.update(False, 2.0 + t / 10, thumbs_down=True)
    assert not ctrl.is_armed(2.6)


def test_lock_mode_ignores_other_mode_gesture():
    a = ArmController(always_armed=True, mode="look", lock_mode=True)
    t = 0.0
    while t < 1.0:
        a.update(True, t)          # two thumbs-up held for 1 s
        t += 1 / 30
    assert a.mode == "look"
