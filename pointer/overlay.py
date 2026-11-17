"""Drawing. Everything is computed in full-frame coordinates and mapped to the (possibly zoomed)
display with a View transform."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_FILES = ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial Unicode.ttf", "DejaVuSans.ttf")

RED = (0, 0, 255)
GREEN = (80, 220, 80)
YELLOW = (0, 220, 255)
GRAY = (170, 170, 170)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class View:
    def __init__(self, frame: np.ndarray, crop: tuple[int, int, int, int] | None):
        h, w = frame.shape[:2]
        if crop is None:
            self.offset, self.scale = np.zeros(2, np.float32), 1.0
            self.image = frame.copy()
        else:
            x, y, cw, ch = crop
            self.offset, self.scale = np.array([x, y], np.float32), w / cw
            self.image = cv2.resize(frame[y:y + ch, x:x + cw], (w, h), interpolation=cv2.INTER_LINEAR)

    def map(self, pts) -> np.ndarray:
        return (np.asarray(pts, np.float32) - self.offset) * self.scale

    def pt(self, p) -> tuple[int, int]:
        q = self.map(p)
        return int(round(q[0])), int(round(q[1]))


def draw_outline(view: View, contour: np.ndarray, color, thickness: int):
    poly = view.map(contour).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(view.image, [poly], True, color, thickness, cv2.LINE_AA)


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.7):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 6), (x + tw + 10, y + 2), color, -1)
    cv2.putText(img, text, (x + 5, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 2, cv2.LINE_AA)


def draw_hand(view: View, hand, pointing: bool):
    color = YELLOW if pointing else GRAY
    for i in (0, 5, 6, 7, 8):
        cv2.circle(view.image, view.pt(hand.points[i]), 4, color, -1, cv2.LINE_AA)
    cv2.circle(view.image, view.pt(hand.points[8]), 8, color, 2, cv2.LINE_AA)


def draw_arm(view: View, arm, pointing: bool):
    color = YELLOW if pointing else GRAY
    draw_outline(view, arm.contour, color, 1)
    cv2.circle(view.image, view.pt(arm.tip), 8, color, 2, cv2.LINE_AA)


def draw_pointing(view: View, origin: np.ndarray, end: np.ndarray):
    cv2.line(view.image, view.pt(origin), view.pt(end), YELLOW, 2, cv2.LINE_AA)


def draw_reticle(view: View, center: np.ndarray, steady: bool):
    """Crosshair in the middle of the view for look mode: green while the head is steady."""
    c = view.pt(center)
    color = GREEN if steady else GRAY
    r = max(14, view.image.shape[1] // 40)
    cv2.circle(view.image, c, r, color, 2, cv2.LINE_AA)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        cv2.line(view.image, (c[0] + dx * r // 2, c[1] + dy * r // 2), (c[0] + dx * r * 3 // 2, c[1] + dy * r * 3 // 2),
                 color, 2, cv2.LINE_AA)


def draw_progress(view: View, det, progress: float):
    x1, y1 = view.pt(det.box[:2])
    center = (x1 + 22, y1 + 22)
    cv2.circle(view.image, center, 16, BLACK, -1, cv2.LINE_AA)
    cv2.ellipse(view.image, center, (14, 14), -90, 0, int(360 * progress), GREEN, 4, cv2.LINE_AA)


@lru_cache(maxsize=16)
def _font(size: int):
    for path in FONT_FILES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(text: str, size: int) -> tuple[int, int]:
    left, top, right, bottom = _font(size).getbbox(text)
    return right - left, bottom - top


def blit_text(img: np.ndarray, text: str, org: tuple[int, int], size: int, color=WHITE, alpha: float = 1.0):
    """Anti-aliased text in a system font, alpha-blended onto a BGR image at top-left `org`."""
    font = _font(size)
    left, top, right, bottom = font.getbbox(text)
    w, h = right - left + 2, bottom - top + 2
    patch = Image.new("L", (w, h), 0)
    ImageDraw.Draw(patch).text((-left + 1, -top + 1), text, font=font, fill=255)
    a = np.asarray(patch, np.float32)[..., None] / 255.0 * alpha
    x, y = org
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    a = a[y0 - y:y1 - y, x0 - x:x1 - x]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def rounded_box(img: np.ndarray, p1, p2, radius: int, color, opacity: float):
    x1, y1 = p1
    x2, y2 = p2
    radius = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    layer = img.copy()
    cv2.rectangle(layer, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(layer, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(layer, (cx, cy), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(layer, opacity, img, 1 - opacity, 0, img)


def wrap_text(text: str, size: int, max_w: int) -> list[str]:
    """Split text into lines no wider than max_w pixels (at font size `size`)."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and text_size(trial, size)[0] > max_w:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + [line] if line else lines


def draw_banner(img: np.ndarray, title: str, subtitle: str | None = None, alpha: float = 1.0):
    """Result card at the bottom of the frame: big title (wrapped to fit), small grey subtitle."""
    h, w = img.shape[:2]
    big, small = max(28, w // 22), max(16, w // 50)
    pad = big // 2
    lines = wrap_text(title, big, int(0.9 * w) - 2 * pad)
    sizes = [text_size(t, big) for t in lines]
    tw, lh = max(s[0] for s in sizes), max(s[1] for s in sizes)
    gap = pad // 3
    th = len(lines) * lh + (len(lines) - 1) * gap
    sw, sh = text_size(subtitle, small) if subtitle else (0, 0)
    bw = max(tw, sw) + 2 * pad
    bh = th + (sh + pad // 2 if subtitle else 0) + 2 * pad
    x1, y2 = (w - bw) // 2, h - max(24, h // 25)
    y1 = y2 - bh
    rounded_box(img, (x1, y1), (x1 + bw, y2), pad, BLACK, 0.7 * alpha)
    for i, (t, (lw, _)) in enumerate(zip(lines, sizes)):
        blit_text(img, t, ((w - lw) // 2, y1 + pad + i * (lh + gap)), big, WHITE, alpha)
    if subtitle:
        blit_text(img, subtitle, ((w - sw) // 2, y1 + pad + th + pad // 2), small, (200, 200, 200), alpha)


def draw_arm_badge(img: np.ndarray, armed: bool, remaining: float, window: float, always: bool,
                   mode: str = "point"):
    """Top-right status pill: grey with the two gestures, or green 'POINT' (with a countdown bar)
    or blue 'LOOK'."""
    h, w = img.shape[:2]
    size = max(16, w // 45)
    if not armed:
        text = "Thumbs-up x2: point   Peace: look"
    elif mode == "look":
        text = "LOOK MODE" + (" (always on)" if always else "")
    else:
        text = "POINT MODE" + (" (always on)" if always else f"  {remaining:4.1f}s")
    tw, th = text_size(text, size)
    pad = size // 2
    x2, y1 = w - pad * 2, pad * 2
    x1, y2 = x2 - tw - 2 * pad, y1 + th + 2 * pad
    color = ((200, 120, 40) if mode == "look" else (60, 170, 60)) if armed else (70, 70, 70)
    rounded_box(img, (x1, y1), (x2, y2), (y2 - y1) // 2, color, 0.85)
    blit_text(img, text, (x1 + pad, y1 + pad), size, WHITE)
    if armed and not always and window > 0 and mode != "look":
        bar = int((x2 - x1 - 2 * pad) * remaining / window)
        cv2.line(img, (x1 + pad, y2 + 6), (x1 + pad + bar, y2 + 6), (60, 170, 60), 4, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str]):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (12, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 4, cv2.LINE_AA)
        cv2.putText(img, line, (12, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
