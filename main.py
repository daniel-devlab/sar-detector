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
import csv
import time
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from detector_utils import filter_by_class_names, find_new_confirmed_track_ids
from geo import CameraPose
from intel_log import IntelLog
from scout_zoom import detect_with_scout
from trackers import ByteTrackTracker
from ultralytics import YOLO

import supervision as sv


@dataclass
class TelemetrySample:
    timestamp_sec: float
    lat: float
    lon: float
    agl_m: float
    heading_deg: float
    gimbal_pitch_deg: float
    gimbal_yaw_deg: float
    hfov_deg: float

    def to_pose(self, frame_w: int, frame_h: int) -> CameraPose:
        return CameraPose(
            lat=self.lat,
            lon=self.lon,
            agl_m=self.agl_m,
            heading_deg=self.heading_deg,
            gimbal_pitch_deg=self.gimbal_pitch_deg,
            gimbal_yaw_deg=self.gimbal_yaw_deg,
            hfov_deg=self.hfov_deg,
            frame_w=frame_w,
            frame_h=frame_h,
        )


def load_telemetry(path: str | None, default_hfov_deg: float) -> list[TelemetrySample]:
    if not path:
        return []

    samples: list[TelemetrySample] = []
    with Path(path).open(newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            samples.append(
                TelemetrySample(
                    timestamp_sec=float(row["timestamp_sec"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    agl_m=float(row["agl_m"]),
                    heading_deg=float(row["heading_deg"]),
                    gimbal_pitch_deg=float(row.get("gimbal_pitch_deg") or 0.0),
                    gimbal_yaw_deg=float(row.get("gimbal_yaw_deg") or 0.0),
                    hfov_deg=float(row.get("hfov_deg") or default_hfov_deg),
                )
            )

    samples.sort(key=lambda sample: sample.timestamp_sec)
    return samples


def make_callback(
    model: YOLO,
    confidence: float,
    device: str,
    class_names: list[str],
    imgsz: int,
    zoom_factor: float,
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
        model_input = image_slice
        if zoom_factor != 1.0:
            model_input = cv2.resize(
                image_slice,
                None,
                fx=zoom_factor,
                fy=zoom_factor,
                interpolation=cv2.INTER_CUBIC,
            )
        result = model(
            model_input,
            conf=confidence,
            device=device,
            imgsz=imgsz,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = filter_by_class_names(detections, model.names, class_names)
        if zoom_factor != 1.0 and len(detections) > 0:
            detections.xyxy = detections.xyxy / zoom_factor
        return detections

    return callback


def draw_overlay(frame: np.ndarray, in_view: int, total_found: int) -> np.ndarray:
    """Draw the running "in view / total found" counters in the corner."""
    text = f"In view: {in_view}   Total found: {total_found}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 500), 76), (0, 0, 0), -1)
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
    frame: np.ndarray,
    xyxy: np.ndarray,
    snapshot_dir: Path,
    track_id: int,
    context_scale: float,
    upscale_factor: float,
) -> str:
    """Save a padded, upsampled crop around `xyxy` for later review."""
    frame_h, frame_w = frame.shape[:2]
    x_min, y_min, x_max, y_max = (int(v) for v in xyxy)
    box_w = max(1, x_max - x_min)
    box_h = max(1, y_max - y_min)
    side = int(max(box_w, box_h) * context_scale)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half_side = max(1, side // 2)

    crop_x_min = max(0, cx - half_side)
    crop_y_min = max(0, cy - half_side)
    crop_x_max = min(frame_w, cx + half_side)
    crop_y_max = min(frame_h, cy + half_side)
    crop = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_dir / f"person_{track_id:04d}.jpg"
    if crop.size > 0:
        if upscale_factor != 1.0:
            crop = cv2.resize(
                crop,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_CUBIC,
            )
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
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Model input size passed through to YOLO inference.",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=3.0,
        help="Explicit cubic upsample factor applied before YOLO sees a tile or ROI.",
    )
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
        "--snapshot-context-scale",
        type=float,
        default=8.0,
        help="How much surrounding context to keep around a tracked box when saving snapshots.",
    )
    parser.add_argument(
        "--snapshot-upscale",
        type=float,
        default=4.0,
        help="Explicit upsample factor applied to saved snapshots.",
    )
    parser.add_argument(
        "--intel-csv", default="intel_log.csv", help="Where to write the intel log CSV."
    )
    parser.add_argument(
        "--telemetry",
        help="Optional CSV with timestamp_sec, lat, lon, agl_m, heading_deg, and optional gimbal/fov columns.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=70.0,
        help="Default horizontal field of view in degrees when telemetry rows omit hfov_deg.",
    )
    parser.add_argument(
        "--geojson",
        help="Optional GeoJSON output path for pinned detections. Defaults beside the intel CSV when telemetry is provided.",
    )
    parser.add_argument(
        "--gpx",
        help="Optional GPX output path for pinned detections. Defaults beside the intel CSV when telemetry is provided.",
    )
    parser.add_argument(
        "--scout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use motion-gated scout ROIs before falling back to full sliced inference.",
    )
    parser.add_argument(
        "--scout-min-area",
        type=int,
        default=8,
        help="Minimum motion-blob area in pixels to treat as a scout ROI.",
    )
    parser.add_argument(
        "--scout-max-area",
        type=int,
        default=900,
        help="Maximum motion-blob area in pixels to treat as a scout ROI.",
    )
    parser.add_argument(
        "--scout-threshold",
        type=int,
        default=12,
        help="Binary threshold applied to frame differencing during the scout pass.",
    )
    parser.add_argument(
        "--scout-pad-ratio",
        type=float,
        default=1.8,
        help="How much padding to add around each scout ROI before zoomed inference.",
    )
    parser.add_argument(
        "--scout-max-rois",
        type=int,
        default=24,
        help="Fall back to full sliced inference when motion produces more than this many ROIs.",
    )
    parser.add_argument(
        "--scout-fallback-interval",
        type=int,
        default=30,
        help="Run one full sliced pass every N processed frames as a safety net; 0 disables it.",
    )
    parser.add_argument(
        "--tracker-min-frames",
        type=int,
        default=3,
        help="Frames a track must survive before it is treated as confirmed.",
    )
    parser.add_argument(
        "--tracker-activation-threshold",
        type=float,
        default=None,
        help="Minimum detection confidence ByteTrack uses to activate a new track. "
        "Defaults to max(0.15, --confidence + 0.05) so it never silently sits "
        "above --confidence and blocks every track from activating.",
    )
    parser.add_argument(
        "--tracker-high-conf-threshold",
        type=float,
        default=0.5,
        help="High-confidence threshold ByteTrack uses during matching.",
    )
    parser.add_argument(
        "--tracker-min-iou",
        type=float,
        default=0.05,
        help="Minimum IoU ByteTrack uses when associating tiny detections across frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [c.strip() for c in args.classes.split(",") if c.strip()]
    telemetry = load_telemetry(args.telemetry, args.hfov)
    telemetry_index = 0
    if args.tracker_activation_threshold is None:
        args.tracker_activation_threshold = max(0.15, args.confidence + 0.05)

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    callback = make_callback(
        model=model,
        confidence=args.confidence,
        device=args.device,
        class_names=class_names,
        imgsz=args.imgsz,
        zoom_factor=args.zoom_factor,
    )
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=args.slice_wh,
        overlap_wh=args.overlap_wh,
        iou_threshold=args.iou_threshold,
    )

    video_info = sv.VideoInfo.from_video_path(args.source)
    tracker = ByteTrackTracker(
        frame_rate=video_info.fps or 30.0,
        track_activation_threshold=args.tracker_activation_threshold,
        minimum_consecutive_frames=args.tracker_min_frames,
        minimum_iou_threshold=args.tracker_min_iou,
        high_conf_det_threshold=args.tracker_high_conf_threshold,
    )

    # A fixed green box (rather than the default per-track color palette)
    # keeps every detection visually consistent -- "green box = person found".
    box_annotator = sv.BoxAnnotator(color=sv.Color.GREEN, thickness=3)
    label_annotator = sv.LabelAnnotator(color=sv.Color.GREEN, text_color=sv.Color.BLACK)
    trace_annotator = sv.TraceAnnotator(color=sv.Color.GREEN)

    intel_log = IntelLog()
    seen_ids: set[int] = set()
    snapshot_dir = Path(args.snapshot_dir)

    fps = video_info.fps or 30.0
    frame_source = sv.get_video_frames_generator(args.source, stride=args.stride)
    prev_gray: np.ndarray | None = None

    # The frame generator only yields every `stride`-th frame, so the sink
    # must be written at fps/stride or the output plays back sped up.
    sink_fps = max(1.0, fps / max(args.stride, 1))
    total_frames = getattr(video_info, "total_frames", None)
    sink_video_info = sv.VideoInfo(
        width=video_info.width,
        height=video_info.height,
        fps=sink_fps,
        total_frames=None
        if not total_frames
        else max(1, total_frames // max(args.stride, 1)),
    )

    print(
        f"Processing {args.source} ({video_info.width}x{video_info.height} @ {fps:.1f}fps)"
    )
    if telemetry:
        print(f"Loaded {len(telemetry)} telemetry sample(s) from {args.telemetry}")
    start_time = time.monotonic()

    try:
        with sv.VideoSink(args.output, sink_video_info) as sink:
            for frame_index, frame in enumerate(frame_source):
                timestamp_sec = (frame_index * args.stride) / fps
                pose: CameraPose | None = None
                if telemetry:
                    while (
                        telemetry_index + 1 < len(telemetry)
                        and telemetry[telemetry_index + 1].timestamp_sec
                        <= timestamp_sec
                    ):
                        telemetry_index += 1
                    pose = telemetry[telemetry_index].to_pose(
                        frame_w=frame.shape[1], frame_h=frame.shape[0]
                    )

                if args.scout:
                    detections, prev_gray, roi_count, used_fallback = detect_with_scout(
                        frame,
                        prev_gray,
                        model=model,
                        class_names=class_names,
                        confidence=args.confidence,
                        device=args.device,
                        imgsz=args.imgsz,
                        zoom_factor=args.zoom_factor,
                        fallback_fn=slicer,
                        frame_index=frame_index,
                        min_area=args.scout_min_area,
                        max_area=args.scout_max_area,
                        threshold=args.scout_threshold,
                        pad_ratio=args.scout_pad_ratio,
                        max_rois=args.scout_max_rois,
                        fallback_interval=args.scout_fallback_interval,
                        merge_iou_threshold=args.iou_threshold,
                    )
                else:
                    detections = slicer(frame)
                    roi_count = 0
                    used_fallback = True

                # `frame=` is intentionally omitted: this tracker ignores it
                # (no camera-motion compensation) and warns if it's passed.
                detections = tracker.update(detections, timestamp=timestamp_sec)

                new_ids = find_new_confirmed_track_ids(detections.tracker_id, seen_ids)
                for track_id in new_ids:
                    row = np.where(detections.tracker_id == track_id)[0][0]
                    snapshot_path = save_snapshot(
                        frame,
                        detections.xyxy[row],
                        snapshot_dir,
                        track_id,
                        context_scale=args.snapshot_context_scale,
                        upscale_factor=args.snapshot_upscale,
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
                        pose=pose,
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
                annotated = label_annotator.annotate(
                    annotated, confirmed, labels=labels
                )
                annotated = draw_overlay(
                    annotated, in_view=len(confirmed), total_found=len(seen_ids)
                )
                if args.scout:
                    mode_text = f"Scout ROIs: {roi_count}   Fallback: {'yes' if used_fallback else 'no'}"
                    cv2.putText(
                        annotated,
                        mode_text,
                        (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                sink.write_frame(annotated)
    except KeyboardInterrupt:
        print("\nInterrupted — saving the intel log and video written so far...")
    finally:
        elapsed = time.monotonic() - start_time
        intel_log.save_csv(args.intel_csv)
        if telemetry:
            csv_path = Path(args.intel_csv)
            geojson_path = (
                Path(args.geojson) if args.geojson else csv_path.with_suffix(".geojson")
            )
            gpx_path = Path(args.gpx) if args.gpx else csv_path.with_suffix(".gpx")
            intel_log.save_geojson(geojson_path)
            intel_log.save_gpx(gpx_path)
        print(f"\nDone in {elapsed:.1f}s. {intel_log.summary()}")
        print(f"Annotated video saved to {args.output}")
        print(f"Intel log saved to {args.intel_csv}")
        if telemetry:
            print(f"GeoJSON pins saved to {geojson_path}")
            print(f"GPX pins saved to {gpx_path}")
        if intel_log.events:
            print(f"Snapshots saved to {snapshot_dir}/")


if __name__ == "__main__":
    main()
