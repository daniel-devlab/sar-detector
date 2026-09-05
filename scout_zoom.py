"""Motion-guided scout/zoom helpers for tiny aerial target detection.

The pipeline here is intentionally simple:

- Scout for compact moving regions on the native-resolution frame.
- Run explicit zoomed inference only on those chips.
- Periodically fall back to full-frame sliced inference as a safety net.

That makes high-elevation streams cheaper than tiling every empty patch of
terrain while giving tiny people more pixels before they reach YOLO.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from detector_utils import filter_by_class_names


type Roi = tuple[int, int, int, int]


def _merge_detections(
    detections_list: list[sv.Detections], iou_threshold: float
) -> sv.Detections:
    non_empty = [detections for detections in detections_list if len(detections) > 0]
    if not non_empty:
        return sv.Detections.empty()
    merged = sv.Detections.merge(non_empty)
    if len(merged) <= 1:
        return merged
    return merged.with_nms(threshold=iou_threshold, class_agnostic=True)


def scout_rois(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    *,
    min_area: int = 8,
    max_area: int = 900,
    threshold: int = 12,
    pad_ratio: float = 1.8,
    blur_kernel: int = 5,
    dilate_iterations: int = 2,
) -> tuple[list[Roi], np.ndarray]:
    """Find compact motion blobs worth zooming into on the next stage."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if prev_gray is None:
        return [], gray

    delta = cv2.absdiff(prev_gray, gray)
    blurred = cv2.GaussianBlur(delta, (blur_kernel, blur_kernel), 0)
    mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.dilate(mask, None, iterations=dilate_iterations)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois: list[Roi] = []
    frame_h, frame_w = frame.shape[:2]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area or area > max_area:
            continue

        pad = int(max(w, h) * pad_ratio)
        rois.append(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(frame_w, x + w + pad),
                min(frame_h, y + h + pad),
            )
        )

    rois.sort(key=lambda roi: (roi[2] - roi[0]) * (roi[3] - roi[1]), reverse=True)
    return rois, gray


def infer_zoomed(
    model: YOLO,
    frame: np.ndarray,
    roi: Roi,
    *,
    zoom_factor: float,
    confidence: float,
    device: str,
    imgsz: int,
    class_names: list[str],
) -> sv.Detections:
    """Run YOLO on an explicitly upsampled ROI and map boxes back."""
    x0, y0, x1, y1 = roi
    chip = frame[y0:y1, x0:x1]
    if chip.size == 0:
        return sv.Detections.empty()

    if zoom_factor != 1.0:
        chip = cv2.resize(
            chip,
            None,
            fx=zoom_factor,
            fy=zoom_factor,
            interpolation=cv2.INTER_CUBIC,
        )

    result = model(chip, conf=confidence, device=device, imgsz=imgsz, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = filter_by_class_names(detections, model.names, class_names)
    if len(detections) == 0:
        return detections

    if zoom_factor != 1.0:
        detections.xyxy = detections.xyxy / zoom_factor
    detections.xyxy[:, [0, 2]] += x0
    detections.xyxy[:, [1, 3]] += y0
    return detections


def detect_with_scout(
    frame: np.ndarray,
    prev_gray: np.ndarray | None,
    *,
    model: YOLO,
    class_names: list[str],
    confidence: float,
    device: str,
    imgsz: int,
    zoom_factor: float,
    fallback_fn: Callable[[np.ndarray], sv.Detections],
    frame_index: int,
    min_area: int,
    max_area: int,
    threshold: int,
    pad_ratio: float,
    max_rois: int,
    fallback_interval: int,
    merge_iou_threshold: float,
) -> tuple[sv.Detections, np.ndarray, int, bool]:
    """Run motion-gated zoom inference, with periodic sliced fallback."""
    rois, gray = scout_rois(
        frame,
        prev_gray,
        min_area=min_area,
        max_area=max_area,
        threshold=threshold,
        pad_ratio=pad_ratio,
    )
    fallback_due = fallback_interval > 0 and frame_index % fallback_interval == 0

    if max_rois > 0 and len(rois) > max_rois:
        return fallback_fn(frame), gray, len(rois), True

    detections = _merge_detections(
        [
            infer_zoomed(
                model,
                frame,
                roi,
                zoom_factor=zoom_factor,
                confidence=confidence,
                device=device,
                imgsz=imgsz,
                class_names=class_names,
            )
            for roi in rois
        ],
        iou_threshold=merge_iou_threshold,
    )

    if len(rois) == 0 and not fallback_due:
        return detections, gray, 0, False

    if fallback_due:
        detections = _merge_detections(
            [detections, fallback_fn(frame)],
            iou_threshold=merge_iou_threshold,
        )
        return detections, gray, len(rois), True

    return detections, gray, len(rois), False
