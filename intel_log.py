"""Session-level "intelligence" log for the SAR small-person detector.

Every time a *new*, tracker-confirmed person is spotted in the video, one
`IntelEvent` is recorded here. This is deliberately modeled after the
`session_log.py` pattern from the rep-counter project: a small dataclass per
event, a container class that accumulates them, and CSV export plus a
plain-language summary at the end of a run.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class IntelEvent:
    """One newly-confirmed person sighting."""

    track_id: int
    frame_index: int
    timestamp_sec: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    snapshot_path: str = ""


@dataclass
class IntelLog:
    """Accumulates `IntelEvent`s over a video run and reports on them."""

    events: list[IntelEvent] = field(default_factory=list)

    def record(
        self,
        track_id: int,
        frame_index: int,
        timestamp_sec: float,
        xyxy: np.ndarray,
        confidence: float,
        snapshot_path: str = "",
    ) -> IntelEvent:
        """Record one new-person-spotted event and return it."""
        x_min, y_min, x_max, y_max = (float(v) for v in xyxy)
        event = IntelEvent(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
            confidence=confidence,
            snapshot_path=snapshot_path,
        )
        self.events.append(event)
        return event

    def summary(self) -> str:
        """Human-readable one-line summary, same style as the rep counter's."""
        count = len(self.events)
        if count == 0:
            return "No new persons detected this run."
        span = self.events[-1].timestamp_sec - self.events[0].timestamp_sec
        return (
            f"{count} distinct person(s) detected over "
            f"{span:.1f}s of footage (first at {self.events[0].timestamp_sec:.1f}s, "
            f"last at {self.events[-1].timestamp_sec:.1f}s)."
        )

    def save_csv(self, path: str | Path) -> None:
        """Write every recorded event to a CSV file, one row per event."""
        path = Path(path)
        fieldnames = [
            "track_id",
            "frame_index",
            "timestamp_sec",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "confidence",
            "snapshot_path",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                writer.writerow(asdict(event))
