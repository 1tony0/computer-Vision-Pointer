"""Evaluate the pointing recognizer on annotated clips.

Ground truth is a CSV with one row per trial (one object pointed at and held):
    clip,start_s,end_s,label
    clips/kitchen.mp4,2.0,5.5,cup
start_s/end_s bound the time the finger is on the object. label is the COCO class name; list
alternatives with "|" (e.g. cup|mug|coffee mug). The detector is right when its label is one of
them; the vision-language model (--vlm) is right when its answer contains one of them.

For every detector model the clips are processed once; the per-frame targets are then replayed
through the hold timer for each hold duration, so the hold sweep costs nothing extra.

Metrics per (model, hold):
  accuracy       trials whose first identification in the window has the right label
  wrong label    trials identified as a different object
  missed         trials with no identification
  false triggers identifications outside every trial window
  time to ID     identification time - start_s, for correct trials
  vlm_accuracy   same as accuracy, judged on the model's answer (only at the pipeline's hold time)
  FPS, hand ms, detector ms  processing speed

Clips recorded with the two-thumbs-up gesture can be evaluated with --require-arm; by default the
gate is off so older clips still count.

Example:
  python evaluate.py --gt data/ground_truth.csv --models models/yolo26n-seg.pt models/yolo11n-seg.pt \
      --holds 1.5 2.0 2.5 3.0 --out results
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pointer.arming import ArmController
from pointer.hold import HoldSelector
from pointer.pipeline import Pipeline

ROOT = Path(__file__).parent
SLACK_S = 0.5  # an identification may land slightly after end_s (the hold completes late)


def load_gt(path):
    trials = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            labels = tuple(a.strip().lower() for a in row["label"].split("|") if a.strip())
            trials[row["clip"]].append((float(row["start_s"]), float(row["end_s"]), labels))
    return trials


def run_clip(pipeline, clip, save_dir=None):
    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, idx, writer, pipeline_events = [], 0, None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image, info = pipeline.process(frame, idx / fps, draw=save_dir is not None)
        if save_dir is not None:
            if writer is None:
                h, w = image.shape[:2]
                writer = cv2.VideoWriter(str(save_dir / (Path(clip).stem + ".mp4")),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(image)
        frames.append((info, pipeline.last_target))
        if info.event is not None:
            pipeline_events.append(info.event)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
    return frames, pipeline_events


def replay(frames, hold_s):
    """Re-run the hold timer on stored per-frame targets."""
    selector, events = HoldSelector(hold_s), []
    for info, target in frames:
        event = selector.update(target, info.time_s)
        if event is not None:
            events.append(event)
    return events


def vlm_match(description: str | None, labels: tuple[str, ...]) -> bool:
    d = (description or "").lower()
    return any(a in d or (d and d in a) for a in labels)


def score(events_by_clip, trials_by_clip, judge=lambda e, labels: e.label in labels):
    correct = wrong = missed = false_trig = 0
    delays = []
    for clip, trials in trials_by_clip.items():
        events = events_by_clip.get(clip, [])
        used = set()
        for start, end, labels in trials:
            hits = [(i, e) for i, e in enumerate(events) if start <= e.time_s <= end + SLACK_S and i not in used]
            if not hits:
                missed += 1
                continue
            i, e = hits[0]
            used.add(i)
            if judge(e, labels):
                correct += 1
                delays.append(e.time_s - start)
            else:
                wrong += 1
        false_trig += len(events) - len(used)
    n = correct + wrong + missed
    return {"trials": n, "accuracy": correct / n if n else 0.0, "correct": correct, "wrong_label": wrong,
            "missed": missed, "false_triggers": false_trig,
            "time_to_id_s": float(np.mean(delays)) if delays else float("nan")}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True)
    p.add_argument("--models", nargs="+", default=[str(ROOT / "models/yolo26n-seg.pt")])
    p.add_argument("--holds", nargs="+", type=float, default=[2.0])
    p.add_argument("--hand-model", default=str(ROOT / "models/gesture_recognizer.task"))
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--save-videos", action="store_true", help="write annotated clips for figures")
    p.add_argument("--pointer", default="hybrid", choices=["hybrid", "landmarks", "mask"])
    p.add_argument("--hand", default="right", choices=["right", "left", "any"])
    p.add_argument("--require-arm", action="store_true", help="only count holds after the two-thumbs-up")
    p.add_argument("--vlm", action="store_true", help="also score the vision-language model's answers")
    p.add_argument("--pipeline-hold", type=float, default=2.0, help="hold time used for the --vlm score")
    args = p.parse_args()

    trials = load_gt(args.gt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    describer = None
    if args.vlm:
        from pointer.describer import Describer
        describer = Describer()
    for weights in args.models:
        name = Path(weights).stem
        save_dir = out / name if args.save_videos else None
        if save_dir:
            save_dir.mkdir(exist_ok=True)
        frames_by_clip, pipeline_events = {}, {}
        for clip in trials:
            # A fresh pipeline per clip so tracking state and timestamps start clean.
            pipeline = Pipeline(args.hand_model, weights, hold_s=args.pipeline_hold, conf=args.conf,
                                device=args.device, pointer_source=args.pointer, hand=args.hand,
                                arming=ArmController(always_armed=not args.require_arm),
                                describer=describer, describe_wait=True)
            frames_by_clip[clip], pipeline_events[clip] = run_clip(pipeline, clip, save_dir)
            pipeline.hands.close()
            print(f"{name}: processed {clip} ({len(frames_by_clip[clip])} frames)")
        infos = [info for frames in frames_by_clip.values() for info, _ in frames]
        total = np.array([i.total_ms for i in infos])
        speed = {"fps": 1000 / total.mean(), "hand_ms": np.mean([i.hand_ms for i in infos]),
                 "detect_ms": np.mean([i.detect_ms for i in infos]),
                 "hand_found_pct": 100 * np.mean([i.num_hands > 0 for i in infos]),
                 "pointing_pct": 100 * np.mean([i.pointing for i in infos])}
        for hold_s in args.holds:
            events = {clip: replay(frames, hold_s) for clip, frames in frames_by_clip.items()}
            row = {"model": name, "pointer": args.pointer, "hold_s": hold_s, **score(events, trials)}
            if describer is not None:
                vlm = score(pipeline_events, trials, judge=lambda e, labels: vlm_match(e.description, labels))
                row["vlm_accuracy"] = vlm["accuracy"] if hold_s == args.pipeline_hold else float("nan")
            rows.append({**row, **speed})
        with open(out / f"events_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip", "time_s", "label", "confidence", "description"])
            for clip, evs in pipeline_events.items():
                for e in evs:
                    w.writerow([clip, f"{e.time_s:.3f}", e.label, f"{e.confidence:.3f}", e.description or ""])

    keys = list(rows[0].keys())
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    table = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    table += ["| " + " | ".join(fmt(r[k]) for k in keys) + " |" for r in rows]
    (out / "results.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
