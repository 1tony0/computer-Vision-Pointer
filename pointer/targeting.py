"""Decides which detected object the user is pointing at.

Rule (documented for the report):
1. Cast a 2-D ray from the index fingertip along the knuckle->tip direction.
2. Every object whose mask the ray crosses is a candidate, ordered by where the ray first enters
   it (objects the fingertip touches count as hit at the tip). The nearest one wins, because it
   occludes what is behind it - except when a later object lies inside the nearer object's box:
   then the nearer one is a supporting surface (table, bed, couch) and the object on it is meant.
3. If the ray crosses nothing, fall back to the object with the smallest angular deviation from
   the ray, as long as it is inside a cone of `cone_deg` degrees.
   Exception: when the ray passes close to the user's other hand ("held") soon after leaving the
   fingertip, the other hand wins over everything,
   since holding something up and pointing at it is deliberate (the held object itself is usually
   not detected, and the finger often overlaps the background, e.g. a laptop, behind it).
4. If the finger is strongly foreshortened (pointing into the scene), its 2-D direction is noise,
   so pick the smallest object under or right next to the fingertip instead.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .detector import Detection, contour_contains


@dataclass
class TargetingParams:
    cone_deg: float = 8.0          # fallback cone half-angle
    step_px: float = 4.0           # ray sampling step
    foreshortened_ratio: float = 0.35
    tip_radius: float = 0.6        # x hand size, used in foreshortened mode
    touch_frac: float = 0.02       # x frame diagonal: fingertip this close to a mask "touches" it
    inside_frac: float = 0.7       # share of a box inside another box to count as "on" it
    held_frac: float = 0.25        # x frame diagonal: how far along the ray the other hand counts
    held_near: float = 0.08        # x frame diagonal: how close the ray must pass to the other hand


def contour_area(contour: np.ndarray) -> float:
    return float(abs(cv2.contourArea(contour.reshape(-1, 1, 2))))


def box_inside(inner: np.ndarray, outer: np.ndarray) -> float:
    """Fraction of `inner` box area that lies inside `outer`."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1) / area) if area > 0 else 0.0


def ray_samples(origin: np.ndarray, direction: np.ndarray, frame_size: tuple[int, int], step: float) -> np.ndarray:
    """Points along the ray from origin until it leaves the frame."""
    w, h = frame_size
    ts = []
    for axis, limit in ((0, w), (1, h)):
        d = direction[axis]
        if abs(d) > 1e-9:
            for bound in (0.0, float(limit)):
                t = (bound - origin[axis]) / d
                if t > 0:
                    ts.append(t)
    t_max = min(ts) if ts else 0.0
    t = np.arange(0.0, t_max + step, step)
    return origin[None, :] + t[:, None] * direction[None, :]


def first_hit_distance(contour: np.ndarray, samples: np.ndarray, step: float) -> float | None:
    poly = contour.reshape(-1, 1, 2)
    for i, p in enumerate(samples):
        if cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0:
            return i * step
    return None


def angular_deviation(contour: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    """Smallest angle (degrees) between the ray and any contour point in front of the fingertip."""
    v = contour - origin[None, :]
    t = v @ direction
    ahead = t > 0
    if not np.any(ahead):
        return 180.0
    v, t = v[ahead], t[ahead]
    perp = np.abs(v[:, 0] * direction[1] - v[:, 1] * direction[0])
    return float(np.degrees(np.min(np.arctan2(perp, t))))


@dataclass
class TargetResult:
    detection: Detection | None
    mode: str          # "ray", "cone", "tip", or "none"
    ray_end: np.ndarray  # where to stop drawing the pointing line


def select_target(detections: list[Detection], origin: np.ndarray, direction: np.ndarray,
                  foreshortening: float, hand_size: float, frame_size: tuple[int, int],
                  params: TargetingParams = TargetingParams()) -> TargetResult:
    samples = ray_samples(origin, direction, frame_size, params.step_px)
    ray_end = samples[-1] if len(samples) else origin

    if foreshortening < params.foreshortened_ratio:
        radius = params.tip_radius * hand_size
        near = []
        for det in detections:
            if contour_contains(det.contour, origin):
                near.append(det)
                continue
            dist = cv2.pointPolygonTest(det.contour.reshape(-1, 1, 2), (float(origin[0]), float(origin[1])), True)
            if -dist <= radius:
                near.append(det)
        if near:
            return TargetResult(min(near, key=lambda d: contour_area(d.contour)), "tip", origin)
        return TargetResult(None, "none", origin)

    diag = float(np.hypot(*frame_size))
    near_px = params.held_near * diag
    for i, p in enumerate(samples[:int(params.held_frac * diag / params.step_px) + 1]):
        for det in detections:
            if det.label == "held" and cv2.pointPolygonTest(
                    det.contour.reshape(-1, 1, 2), (float(p[0]), float(p[1])), True) >= -near_px:
                return TargetResult(det, "ray", p.copy())

    touch_px = params.touch_frac * diag
    crossed = []
    for det in detections:
        gap = -cv2.pointPolygonTest(det.contour.reshape(-1, 1, 2), (float(origin[0]), float(origin[1])), True)
        hit = 0.0 if gap <= touch_px else first_hit_distance(det.contour, samples, params.step_px)
        if hit is not None:
            crossed.append((hit, det))
    if crossed:
        crossed.sort(key=lambda c: c[0])
        hit, det = crossed[0]
        for later_hit, later in crossed[1:]:
            if box_inside(later.box, det.box) >= params.inside_frac:
                hit, det = later_hit, later
        return TargetResult(det, "ray", origin + direction * hit)

    best, best_angle = None, params.cone_deg
    for det in detections:
        angle = angular_deviation(det.contour, origin, direction)
        if angle <= best_angle:
            best, best_angle = det, angle
    if best is not None:
        t = float(np.dot(best.center - origin, direction))
        return TargetResult(best, "cone", origin + direction * max(t, 0.0))
    return TargetResult(None, "none", ray_end)
