"""Point at an object, hold for 2 s, and the program names it on screen and out loud.

Recognition is off until you give a thumbs-up with both hands (it stays armed while you keep pointing; press "a" in the window to arm
from the keyboard). A Hugging Face vision-language model names what you point at, including
objects the detector doesn't know.

Examples:
  python run.py --source 0                      # Mac webcam
  python run.py --source clips/kitchen.mp4      # clip recorded on the Ray-Ban Meta glasses
  python run.py --source clips/kitchen.mp4 --save out.mp4 --log events.csv --no-display
  python run.py --source clips/old.mov --always-armed --no-vlm   # original behaviour
"""

import argparse
import csv
from pathlib import Path

import cv2

from pointer.arming import ArmController
from pointer.describer import DEFAULT_MODEL
from pointer.pipeline import Pipeline, banner_text
from pointer.speech import Speaker, add_speech_track

ROOT = Path(__file__).parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index or video file path")
    p.add_argument("--model", default=str(ROOT / "models/yolo26n-seg.pt"), help="YOLO segmentation weights")
    p.add_argument("--hand-model", default=str(ROOT / "models/gesture_recognizer.task"))
    p.add_argument("--hold", type=float, default=2.0, help="seconds to hold before identifying")
    p.add_argument("--conf", type=float, default=0.35, help="detector confidence threshold")
    p.add_argument("--device", default=None, help="cpu or mps (default: mps when available)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--detect-every", type=int, default=1, help="run the detector every N frames")
    p.add_argument("--pointer", default="hybrid", choices=["hybrid", "landmarks", "mask"],
                   help="how to find the pointing hand (default: MediaPipe, then arm mask as fallback)")
    p.add_argument("--hand", default="right", choices=["right", "left", "any"],
                   help="which hand may point (default right index finger; the other hand is ignored)")
    p.add_argument("--always-armed", action="store_true", help="skip the two-thumbs-up gate")
    p.add_argument("--arm-window", type=float, default=5.0, help="seconds without pointing before it disarms")
    p.add_argument("--no-look", action="store_true", help="disable look mode (the peace sign)")
    p.add_argument("--mode", default="point", choices=["point", "look"], help="starting mode (used with --always-armed)")
    p.add_argument("--lock-mode", action="store_true", help="stay in --mode; ignore the other mode's gesture")
    p.add_argument("--look-time", type=float, default=1.5, help="seconds of steady looking before naming it")
    p.add_argument("--no-vlm", action="store_true", help="name objects with YOLO labels only")
    p.add_argument("--vlm-model", default=DEFAULT_MODEL, help="Hugging Face vision-language model")
    p.add_argument("--save", help="write the annotated video here")
    p.add_argument("--log", help="write identification events (CSV) here")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--mute", action="store_true", help="don't speak results out loud")
    p.add_argument("--no-audio", action="store_true", help="don't add the speech to the --save video")
    p.add_argument("--max-height", type=int, default=900, help="display window height limit")
    return p.parse_args()


def main():
    args = parse_args()
    is_camera = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_camera else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    describer = None
    if not args.no_vlm:
        from pointer.describer import Describer
        print(f"Loading {args.vlm_model} ...")
        describer = Describer(args.vlm_model)
    arming = ArmController(window_s=args.arm_window, always_armed=args.always_armed, mode=args.mode,
                           lock_mode=args.lock_mode)
    # Recorded clips wait for the model (deterministic output); the webcam never blocks.
    pipeline = Pipeline(args.hand_model, args.model, hold_s=args.hold, conf=args.conf,
                        device=args.device, detect_every=args.detect_every, imgsz=args.imgsz,
                        pointer_source=args.pointer, hand=args.hand, look=not args.no_look, look_s=args.look_time, arming=arming, describer=describer,
                        describe_wait=not is_camera)
    speaker = Speaker(enabled=not args.mute)
    writer = None
    events = []
    frame_idx = 0
    was_armed = arming.is_armed(0.0)
    start_tick = cv2.getTickCount()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if is_camera:
                frame = cv2.flip(frame, 1)  # mirror so pointing feels natural on a webcam
                time_s = (cv2.getTickCount() - start_tick) / cv2.getTickFrequency()
            else:
                time_s = frame_idx / fps  # video time keeps results reproducible
            frame_idx += 1

            image, info = pipeline.process(frame, time_s)
            if info.just_armed:
                how = ("thumbs-down to turn off" if arming.mode == "look" else
                       f"off after {args.arm_window:.0f} s without pointing, or thumbs-down")
                verb = "switched to" if arming.just_switched else "armed:"
                print(f"[{time_s:7.2f}s] {verb} {arming.mode} mode ({how})")
            elif info.just_disarmed:
                print(f"[{time_s:7.2f}s] disarmed (thumbs-down)")
            elif was_armed and not info.armed:
                print(f"[{time_s:7.2f}s] disarmed (no pointing for {args.arm_window:.0f} s)")
            was_armed = info.armed
            if info.event is not None:
                e = info.event
                events.append(e)
                if describer is None:
                    text = banner_text(e, False)[0]
                    print(f"[{e.time_s:7.2f}s] {text}  (detector: {e.label} {e.confidence:.2f})")
                    speaker.say(text, time_s)
            if info.described is not None:
                e = info.described
                text = banner_text(e, True)[0]
                print(f"[{e.time_s:7.2f}s] {text}  (detector: {e.label} {e.confidence:.2f}, "
                      f"model score {e.vlm_score or 0:.2f})")
                speaker.say(text, time_s)

            if args.save:
                if writer is None:
                    h, w = image.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(image)

            if not args.no_display:
                h, w = image.shape[:2]
                if h > args.max_height:
                    image = cv2.resize(image, (int(w * args.max_height / h), args.max_height))
                cv2.imshow("Pointing recognizer (q to quit, space to pause)", image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)
                if key == ord("a"):
                    arming.arm(time_s)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        pipeline.close()

    if args.save and not args.no_audio and speaker.log:
        # Put the spoken results into the saved video too, at the moments they were said.
        tmp = Path(args.save).with_suffix(".noaudio.mp4")
        Path(args.save).rename(tmp)
        if add_speech_track(str(tmp), speaker.log, args.save):
            tmp.unlink()
            print(f"Added speech to {args.save}")
        else:
            tmp.rename(args.save)

    if args.log:
        with open(args.log, "w", newline="") as f:
            out = csv.writer(f)
            out.writerow(["source", "time_s", "started_s", "label", "confidence", "description"])
            for e in events:
                out.writerow([args.source, f"{e.time_s:.3f}", f"{e.started_s:.3f}", e.label,
                              f"{e.confidence:.3f}", e.description or ""])
    print(f"Processed {frame_idx} frames, {len(events)} identifications.")


if __name__ == "__main__":
    main()
