# Pointing Recognizer

Give a thumbs-up with both hands to arm it, point at an object in first-person video, hold for 2 s, and the
program outlines it, names it on screen and says it out loud. A Hugging Face vision-language
model (SmolVLM2) does the naming, so it also works for things the detector doesn't know. Built for CPS 843 (Computer Vision) using video recorded on
Ray-Ban Meta glasses; it also runs live on a Mac webcam.

## Pipeline

```
frame ─┬─> MediaPipe HandLandmarker ──> gestures (point / OK-sign zoom / open palm)
       │                                   │
       │                  ZoomController ◄─┘ (crop for digital zoom)
       │                          │
       └─> YOLO segmentation on (cropped) frame ──> object masks
                                          │
             pointing ray (index knuckle → tip) ──> target selection ──> 2 s hold ──> label + speech
```

| Stage | File | What it does |
|---|---|---|
| Hand tracking | `pointer/hands.py` | MediaPipe GestureRecognizer: 21 landmarks per hand + canned gestures (Thumb_Up); rotation-invariant finger-extension test; pointing ray |
| Arming | `pointer/arming.py` | Thumbs-up on **both hands** held 0.4 s arms recognition (one thumbs-up is too easily confused with pointing seen from behind); it stays armed while you keep pointing and disarms after 5 s without pointing or on a thumbs-down; nothing is targeted while disarmed |
| Arm-mask fallback | `pointer/hand_mask.py` | When MediaPipe misses the hand (close, cut off, seen from the back — typical first-person pointing), use the YOLO "person" mask touching the frame edge: fingertip = farthest outline point from the entry, direction from the finger's outline centre line |
| Detection | `pointer/detector.py` | YOLO26n-seg (COCO, 80 classes); ignores `person` (the user's own arm) |
| Targeting | `pointer/targeting.py` | First object the ray hits (touched objects first); a surface is skipped for an object lying on it; 8° cone fallback; fingertip mode when the finger is foreshortened |
| Hold | `pointer/hold.py` | Same object (label + IoU ≥ 0.3) for 2 s of **video time**, 0.4 s dropout grace, 0.2 s debounce before switching targets |
| Zoom | `pointer/zoom.py` | OK sign: spread/close thumb–index to zoom 1–4×; open palm 0.6 s resets |
| Naming | `pointer/describer.py` | When a hold completes, the sharpest crop of the hold (variance of the Laplacian), with the background outside the object's outline faded, goes to SmolVLM2-500M (Hugging Face) on a worker thread. With no detected object, a square just ahead of the fingertip is used instead |
| Drawing | `pointer/overlay.py` | Yellow pointing line, red outline on target, hold progress ring, rounded result card, armed badge |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Model weights are in `models/` (MediaPipe `gesture_recognizer.task`, `yolo26n-seg.pt`, `yolo11n-seg.pt`).

## Run

```bash
.venv/bin/python run.py --source 0                          # webcam
.venv/bin/python run.py --source clips/kitchen.mp4           # recorded glasses clip
.venv/bin/python run.py --source clips/kitchen.mp4 --save out.mp4 --log events.csv
```

`--pointer hybrid|landmarks|mask` picks how the hand is found (default hybrid: MediaPipe, then
the arm mask). Comparing the three is a ready-made experiment.

Only the **right index finger** points (`--hand left|any` to change it). The other hand is
ignored, so it can hold things or rest on the desk. Which arm is which comes from where it enters
the frame (the right arm comes in from the right), since MediaPipe's left/right label is
unreliable for a hand seen from behind. In the mask fallback the end of the arm must also be
finger-thin, so fists and relaxed hands don't count as pointing.

Pointing at something in your **other hand** works too: when the ray passes close to the other
hand, the target is what it holds. If YOLO recognises the held object (e.g. a phone) that
detection is used; otherwise the vision-language model is asked what the hand is holding. With
both arms in view, the arm entering further right is the right arm.

Recognition starts **disarmed**: give a thumbs-up with both hands (or press `a` in the window) to arm it. It
stays armed while you keep pointing. It switches off when you give a thumbs-down (hold ~0.5 s) or after 5 s without pointing (`--arm-window`).
`--always-armed` skips the gate, `--no-vlm` names objects with YOLO labels only. The first run
downloads the vision-language model (~2 GB) into `~/.cache/huggingface`.

Results are shown and **spoken**. When the detector and the language model agree (e.g. YOLO
"tv", model "computer monitor"), or the model is confident (score ≥ 0.6, the geometric-mean
probability of its answer tokens) about something YOLO doesn't know, it says "This is a …".
Otherwise it says "I don't know, but it looks like a …". With `--save`, the speech is also mixed
into the saved video at the right moments (`--no-audio` to skip; needs ffmpeg).

Keys: `q` quit, `space` pause, `a` arm. `--mute` disables speech, `--hold 1.0` changes the hold time.

## Two modes

| Start with | Mode | What gets named |
|---|---|---|
| Thumbs-up with **both** hands | **Point** | What your right index finger points at (hold 2 s). Off after 5 s without pointing. |
| **Peace sign** ✌ (one hand) | **Look** | Whatever stays in the middle of your view while your head is still for 1.5 s. Hands not needed; stays on until a thumbs-down. |

Making the other mode's gesture switches modes; a thumbs-down turns either one off. The badge in the
top-right shows the mode (green POINT, blue LOOK).

Look mode measures head motion between frames with phase correlation (`pointer/look.py`). After
something is named you have to look elsewhere before the next one. Signs are read out ("a train
station sign that says 'No smoking'"). The 1.5 s comes from the station clip: deliberate looks kept
the head still for 1.7-4.2 s, incidental still moments lasted 1.1-1.3 s. The peace sign was never
detected by accident in any of the nine test clips. `--no-look` disables look mode; with
`--always-armed`, `--mode look` starts in look mode, and `--lock-mode` keeps that mode (for clips recorded before the mode gestures existed).

## Gestures

| Gesture | Effect |
|---|---|
| Thumbs-up with both hands (hold ~0.5 s) | Arm recognition (badge turns green); stays on while you keep pointing, off after 5 s with hands down |
| Thumbs-down (hold ~0.5 s) | Disarm straight away |
| Right index finger out, other fingers curled | Point: yellow line, red outline on the target, hold 2 s to identify |
| OK sign (thumb+index pinch, other three open), held 0.3 s | Starts zooming: open thumb and index apart to zoom in (opening the whole hand is fine); zoom never goes back out by itself; lower the hand to keep the zoom |
| Open palm for 0.6 s (at least 1 s after a zoom) | Reset zoom to 1× |

## Evaluate

1. Put clips in `clips/`.
2. Annotate trials in a CSV like `data/ground_truth_example.csv`: `clip,start_s,end_s,label`,
   where the window is when your finger is on the object and `label` is the COCO class name.
3. Run:

```bash
.venv/bin/python evaluate.py --gt data/ground_truth.csv \
    --models models/yolo26n-seg.pt models/yolo11n-seg.pt \
    --holds 1.5 2.0 2.5 3.0 --out results --save-videos
```

Add `--vlm` to also score the model's answers (`vlm_accuracy`), `--pointer` to compare hand
methods, and `--require-arm` for clips that include the two-thumbs-up. Labels can list alternatives:
`cup|mug|coffee mug`.

Outputs `results/results.csv` and `results/results.md` (accuracy, wrong labels, misses, false
triggers, time to identify, FPS, per-stage latency) and annotated videos for figures.

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

## Third-party code

MediaPipe Gesture Recognizer (Google, Apache-2.0), Ultralytics YOLO (AGPL-3.0), OpenCV (Apache-2.0),
SmolVLM2 (Hugging Face, Apache-2.0) via `transformers` (Apache-2.0).
Cite these in the report.
