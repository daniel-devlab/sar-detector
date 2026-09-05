# SAR Small-Person Detector

A search-and-rescue-style aerial detector: finds small/distant people in
drone video, tracks them across frames so the same person isn't
double-counted, and keeps an "intel log" of every new person spotted
(timestamp, location in frame, and a saved snapshot crop).

Built on the same stack as the rest of this portfolio — `ultralytics` YOLO
+ `supervision` — plus two pieces that are new to this project:

- **Tiled inference** (`sv.InferenceSlicer`): drone footage is high
  resolution with tiny subjects; running YOLO on the whole frame at once
  would downscale people to almost nothing. The slicer runs the model on
  overlapping tiles instead and merges the results.
- **Multi-object tracking** (`ByteTrackTracker`, from the `trackers`
  package): assigns a stable ID to each person across frames, so "new
  person spotted" fires once per person, not once per frame.

## Example

Real output on a high-altitude drone frame (a stock, non-fine-tuned
`yolo11n.pt` restricted to the "person" class, `--slice-wh 320 --overlap-wh
60 --confidence 0.1`). Each green box is one detection with its track ID —
note how small the people are at this altitude, which is exactly the
problem tiled inference and fine-tuning both exist to solve:

![Example detections on an aerial frame](example_detection.jpg)

## Project layout

- `main.py` — the local inference pipeline: loads a model, slices+tracks a
  video file, writes an annotated output video, and saves the intel log.
- `detector_utils.py` — the two pieces of logic worth unit-testing on their
  own: filtering detections down to classes you care about, and deciding
  whether a tracked ID is genuinely new.
- `intel_log.py` — records each new-person event and exports it to CSV.
- `train_colab.py` — a cell-by-cell script (paste into Google Colab) for
  fine-tuning a YOLO model on an aerial person-detection dataset, optionally
  re-tiling the training images first with `TrainingSlicer`.
- `example_detection.jpg` — the screenshot above.

## Quickstart (works today, no fine-tuning required)

This runs end-to-end with a stock pretrained YOLO model restricted to the
"person" class, so you can see the whole pipeline — tiling, tracking,
intel logging — before investing time in fine-tuning.

```powershell
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt

python main.py --source drone_video.mp4 --output annotated.mp4 --classes person
```

Outputs:
- `annotated.mp4` — the video with green boxes, track IDs, and a running
  "in view / total found" counter drawn on it.
- `intel_log.csv` — one row per newly-spotted person.
- `snapshots/` — one cropped image per newly-spotted person.

## Improving accuracy: fine-tune on aerial data

A stock YOLO model was trained on ground-level photos, so it under-performs
on the aerial viewpoint and tiny subjects a SAR drone actually sees. Open
`train_colab.py` in Google Colab (free GPU) to fine-tune on a public aerial
person-detection dataset from [Roboflow Universe](https://universe.roboflow.com),
then point `main.py` at the resulting `best.pt`:

```powershell
python main.py --source drone_video.mp4 --model best.pt --classes person
```

## Useful flags

| Flag | Default | Purpose |
| --- | --- | --- |
| `--slice-wh` | 640 | Tile size for `InferenceSlicer`. Smaller tiles help with more distant/smaller people, at the cost of more inference calls per frame. |
| `--overlap-wh` | 100 | Overlap between tiles, so a person straddling a tile boundary is still detected whole in the neighboring tile. |
| `--stride` | 1 | Process every Nth frame. Raise this (e.g. `15`-`60`) on CPU to keep up with long or high-resolution footage — the tracker's `timestamp` handling keeps counts and timing correct even when frames are skipped. |
| `--confidence` | 0.25 | Minimum detection confidence. |
| `--device` | cpu | `cuda` or `cuda:0` if you have a GPU available locally (check with `nvidia-smi`). |

## Troubleshooting: nothing gets detected

If `intel_log.csv` comes back empty and `annotated.mp4` has no boxes at
all, the most likely cause is tile size vs. footage: at the default
`--slice-wh 640`, a video that's high-altitude, high-resolution, or
already narrower than 640px effectively gets **zero tiling** — the model
sees the whole frame at once, and people at real SAR altitudes are only a
handful of pixels, well below what a stock detector can recognize.

Fixes, cheapest first:
- Lower `--confidence` (e.g. `0.1`) — costs nothing extra, sometimes enough
  on its own.
- Shrink `--slice-wh` (try `320`, then `160`) — each tile gets upscaled to
  the model's input size, so a tiny person occupies far more of what the
  model actually sees. This is slower: smaller tiles mean more inference
  calls per frame.
- Always pair a small `--slice-wh` with `--stride 15` or higher on CPU —
  otherwise a single 4K video can take well over an hour to process.
- Expect some false positives (tree canopy, rooftop clutter) once tiles get
  small — a stock model wasn't trained on this viewpoint. That gap is
  exactly what `train_colab.py`'s fine-tuning step is for.

## How "new person spotted" is decided

`ByteTrackTracker` gives every detection a `tracker_id`. A brand new track
starts at `tracker_id == -1` ("unconfirmed") and only gets a real ID once it
has matched for `minimum_consecutive_frames` (default 2) in a row — this
avoids treating single-frame false positives as sightings. `main.py` only
logs and draws confirmed tracks, and logs each one exactly once, the first
frame it becomes confirmed.
