"""Small, pure, unit-testable helpers used by main.py.

Kept separate from main.py (which does the actual video I/O, model loading,
and OpenCV drawing) so the logic that is easy to get subtly wrong — class
filtering and "is this a genuinely new track" detection — can be tested with
plain synthetic `Detections`, no camera/model/video file required.
"""

from __future__ import annotations

import numpy as np

import supervision as sv


def filter_by_class_names(
    detections: sv.Detections,
    model_names: dict[int, str],
    class_names: list[str],
) -> sv.Detections:
    """Keep only detections whose class name is in `class_names`.

    If `class_names` is empty, `detections` is returned unchanged (no
    filtering) — this is what lets the same pipeline run either a stock
    multi-class COCO model restricted down to "person", or a model that was
    fine-tuned to predict only the classes you care about already.

    Args:
        detections: Detections for one frame or slice.
        model_names: The underlying model's `{class_id: class_name}` mapping
            (e.g. `YOLO(...).names`).
        class_names: Class names to keep, matched case-insensitively. Names
            that don't exist in `model_names` are ignored.

    Returns:
        A new `Detections` containing only the matching rows.
    """
    if not class_names:
        return detections
    if len(detections) == 0 or detections.class_id is None:
        return detections

    wanted = {name.lower() for name in class_names}
    keep_class_ids = {
        class_id
        for class_id, name in model_names.items()
        if name.lower() in wanted
    }
    mask = np.isin(detections.class_id, list(keep_class_ids))
    return detections[mask]


def find_new_confirmed_track_ids(
    tracker_ids: np.ndarray, seen_ids: set[int]
) -> list[int]:
    """Return confirmed track IDs in `tracker_ids` not already in `seen_ids`.

    `ByteTrackTracker` (from the `trackers` package) assigns `tracker_id
    == -1` to a track that hasn't yet been confirmed
    (`minimum_consecutive_frames`, default 2). Those must never be treated as
    "new persons spotted" — they're provisional and may vanish next frame —
    so this function drops them before checking `seen_ids`.

    Does not mutate `seen_ids`; the caller adds the returned IDs once it has
    finished acting on them (e.g. after saving a snapshot), so a failure
    mid-loop can't silently drop a sighting.

    Args:
        tracker_ids: `detections.tracker_id` array for the current frame,
            after `ByteTrackTracker.update(...)`.
        seen_ids: Track IDs already recorded in a previous frame.

    Returns:
        Newly-confirmed track IDs this frame, in the order they appear in
        `tracker_ids`, with no duplicates.
    """
    new_ids: list[int] = []
    for track_id in tracker_ids:
        track_id_int = int(track_id)
        if track_id_int == -1:
            continue
        if track_id_int in seen_ids or track_id_int in new_ids:
            continue
        new_ids.append(track_id_int)
    return new_ids
