"""SAR small-person detector: tiled inference + tracking + intel logging.

Runs a YOLO model over a (typically high-resolution, small-subjects) drone
video using `sv.InferenceSlicer` so small/distant people are not missed the
way they would be by running the model on the whole downscaled frame at
once. Detections are fed through `ByteTrackTracker` (from the `trackers`
package) so the same person isn't re-reported every frame, and every
newly-confirmed track is written to an "intel log" (CSV + saved snapshot
crop) via `intel_log.IntelLog` — the running record of "who was spotted,
when, and where in the frame".

Works out of the box with a stock pretrained YOLO model (e.g.
`yolo11n.pt`) restricted to the "person" class via `--classes person`, so
the whole pipeline can be exercised on ordinary video before a model
fine-tuned on aerial/small-person data (see `train_colab.py`) exists. Once
you have a fine-tuned model, point `--model` at its `best.pt` weights
instead.

Usage:
    python main.py --source drone_flyover.mp4 --output annotated.mp4 \
        --model yolo11n.pt --classes person --slice-wh 640 --overlap-wh 100

Press Ctrl+C to stop early; the intel log and video written so far are
still saved.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from detector_utils import filter_by_class_names, find_new_confirmed_track_ids
from intel_log import IntelLog
from trackers import ByteTrackTracker
from ultralytics import YOLO

import supervision as sv


def make_callback(
    model: YOLO, confidence: float, device: str, class_names: list[str]
) -> Callable[[np.ndarray], sv.Detections]:
    """Build the per-tile inference callback `InferenceSlicer` will call.

    Args:
        model: A loaded `ultralytics.YOLO` model.
        confidence: Minimum detection confidence passed to the model.
        device: `"cpu"`, `"cuda"`, `"cuda:0"`, or `"mps"`.
        class_names: Class names to keep (see `filter_by_class_names`); pass
            an empty list to keep every class the model predicts.
    """

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, conf=confidence, device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        return filter_by_class_names(detections, model.names, class_names)

    return callback


def draw_overlay(frame: np.ndarray, in_view: int, total_found: int) -> np.ndarray:
    """Draw the running "in view / total found" counters in the corner."""
    text = f"In view: {in_view}   Total found: {total_found}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 420), 40), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def save_snapshot(
    frame: np.ndarray, xyxy: np.ndarray, snapshot_dir: Path, track_id: int
) -> str:
    """Crop `xyxy` out of `frame` and save it under `snapshot_dir`."""
    x_min, y_min, x_max, y_max = (max(0, int(v)) for v in xyxy)
    crop = frame[y_min:y_max, x_min:x_max]
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"person_{track_id:04d}.jpg"
    if crop.size > 0:
        cv2.imwrite(str(path), crop)
        return str(path)
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--output", default="annotated.mp4", help="Path to write the annotated video."
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Path or name of the YOLO weights (stock or fine-tuned).",
    )
    parser.add_argument(
        "--classes",
        default="person",
        help='Comma-separated class names to keep, e.g. "person". Empty string keeps all.',
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default="cpu", help='"cpu", "cuda", or "cuda:0".')
    parser.add_argument("--slice-wh", type=int, default=640)
    parser.add_argument("--overlap-wh", type=int, default=100)
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold InferenceSlicer uses to merge duplicate detections "
        "from overlapping tiles.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every Nth frame (raise this on CPU to keep up with long footage).",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="snapshots",
        help="Directory to save a crop of each newly-spotted person.",
    )
    parser.add_argument(
        "--intel-csv", default="intel_log.csv", help="Where to write the intel log CSV."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    callback = make_callback(
        model=model,
        confidence=args.confidence,
        device=args.device,
        class_names=class_names,
    )
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=args.slice_wh,
        overlap_wh=args.overlap_wh,
        iou_threshold=args.iou_threshold,
    )

    video_info = sv.VideoInfo.from_video_path(args.source)
    tracker = ByteTrackTracker(frame_rate=video_info.fps or 30.0)

    # A fixed green box (rather than the default per-track color palette)
    # keeps every detection visually consistent -- "green box = person found".
    box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=3)
    label_annotator = sv.LabelAnnotator(
        color=sv.Color.GREEN, text_color=sv.Color.BLACK
    )
    trace_annotator = sv.TraceAnnotator(color=sv.Color.GREEN)

    intel_log = IntelLog()
    seen_ids: set[int] = set()
    snapshot_dir = Path(args.snapshot_dir)

    fps = video_info.fps or 30.0
    frame_source = sv.get_video_frames_generator(args.source, stride=args.stride)

    print(f"Processing {args.source} ({video_info.width}x{video_info.height} @ {fps:.1f}fps)")
    start_time = time.monotonic()

    try:
        with sv.VideoSink(args.output, video_info) as sink:
            for frame_index, frame in enumerate(frame_source):
                timestamp_sec = (frame_index * args.stride) / fps

                detections = slicer(frame)
                # `frame=` is intentionally omitted: this tracker ignores it
                # (no camera-motion compensation) and warns if it's passed.
                detections = tracker.update(detections, timestamp=timestamp_sec)

                new_ids = find_new_confirmed_track_ids(detections.tracker_id, seen_ids)
                for track_id in new_ids:
                    row = np.where(detections.tracker_id == track_id)[0][0]
                    snapshot_path = save_snapshot(
                        frame, detections.xyxy[row], snapshot_dir, track_id
                    )
                    intel_log.record(
                        track_id=track_id,
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        xyxy=detections.xyxy[row],
                        confidence=float(detections.confidence[row])
                        if detections.confidence is not None
                        else 0.0,
                        snapshot_path=snapshot_path,
                    )
                    print(
                        f"[{timestamp_sec:6.1f}s] new person spotted -> track_id={track_id}"
                    )
                seen_ids.update(new_ids)

                # Only draw confirmed tracks (tracker_id != -1): unconfirmed
                # detections are single-frame blips that haven't passed
                # `minimum_consecutive_frames` yet and would just clutter the
                # video with boxes that vanish a frame later.
                confirmed = detections[detections.tracker_id != -1]
                labels = [f"#{tid}" for tid in confirmed.tracker_id]
                annotated = trace_annotator.annotate(frame.copy(), confirmed)
                annotated = box_annotator.annotate(annotated, confirmed)
                annotated = label_annotator.annotate(annotated, confirmed, labels=labels)
                annotated = draw_overlay(
                    annotated, in_view=len(confirmed), total_found=len(seen_ids)
                )

                sink.write_frame(annotated)
    except KeyboardInterrupt:
        print("\nInterrupted — saving the intel log and video written so far...")
    finally:
        elapsed = time.monotonic() - start_time
        intel_log.save_csv(args.intel_csv)
        print(f"\nDone in {elapsed:.1f}s. {intel_log.summary()}")
        print(f"Annotated video saved to {args.output}")
        print(f"Intel log saved to {args.intel_csv}")
        if intel_log.events:
            print(f"Snapshots saved to {snapshot_dir}/")


if __name__ == "__main__":
    main()
