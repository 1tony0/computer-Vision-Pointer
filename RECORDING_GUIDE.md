# Recording clips on the Ray-Ban Meta glasses

Goal: a small, varied, annotated dataset for the Experiments section.

Start every clip with a **thumbs-up on both hands** (both in view, ~1 s) to arm recognition,
and end with a thumbs-down or both hands down for 5 s.

## What to record

Each clip is 20–60 s with 3–6 **trials**. A trial is: point at one object with your **right** index finger,
hold still for ~3 s, then drop your hand for ~2 s before the next one.

Use objects the detector knows (COCO classes), e.g. cup, bottle, laptop, keyboard, mouse,
cell phone, book, chair, couch, tv, remote, scissors, clock, vase, potted plant, bowl, banana,
apple, backpack, umbrella, microwave, sink, refrigerator.

Aim for roughly 60 trials total, covering:

| Condition | Examples |
|---|---|
| Lighting | bright daylight, indoor lamp, dim room |
| Distance | object at 0.5 m, 1–2 m, 3 m+ |
| Clutter | object alone vs. surrounded by others |
| Pointing style | finger clearly visible vs. finger pointing "into" the scene (foreshortened) |
| Zoom | a few trials on small/far objects using the OK-sign zoom |
| Negatives | 2–3 clips of just moving around without pointing (measures false triggers) |

## Tips

- Keep your hand in the lower half of the view; the glasses camera sits at the left temple.
- Hold your head fairly still during each hold.
- Start the video with a clap or say the clip name, then write down trial times afterwards.

## Getting clips to the Mac

Captures sync to the Meta AI app on your phone → save to Photos → AirDrop to the Mac → put them
in `clips/`.

## Annotating

Open each clip in QuickTime, note when your finger lands on and leaves each object, and add a row
to `data/ground_truth.csv`:

```
clip,start_s,end_s,label
clips/desk_day.mp4,3.2,6.8,laptop
```
