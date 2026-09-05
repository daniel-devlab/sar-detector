"""Session-level "intelligence" log for the SAR small-person detector.

Every time a *new*, tracker-confirmed person is spotted in the video, one
`IntelEvent` is recorded here. This is deliberately modeled after the
`session_log.py` pattern from the rep-counter project: a small dataclass per
event, a container class that accumulates them, and CSV plus map-friendly
exports at the end of a run.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from geo import CameraPose, GroundPin, box_center_pin


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
    lat: float | None = None
    lon: float | None = None
    agl_m: float | None = None
    gsd_cm: float | None = None
    err_radius_m: float | None = None


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
        pose: CameraPose | None = None,
        pin: GroundPin | None = None,
    ) -> IntelEvent:
        """Record one new-person-spotted event and return it."""
        if pin is None and pose is not None:
            pin = box_center_pin(xyxy, pose)
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
            lat=None if pin is None else pin.lat,
            lon=None if pin is None else pin.lon,
            agl_m=None if pin is None else pin.agl_m,
            gsd_cm=None if pin is None else pin.gsd_cm,
            err_radius_m=None if pin is None else pin.err_radius_m,
        )
        self.events.append(event)
        return event

    def summary(self) -> str:
        """Human-readable one-line summary, same style as the rep counter's."""
        count = len(self.events)
        if count == 0:
            return "No new persons detected this run."
        pinned = sum(1 for event in self.events if event.lat is not None)
        span = self.events[-1].timestamp_sec - self.events[0].timestamp_sec
        return (
            f"{count} distinct person(s) detected over "
            f"{span:.1f}s of footage (first at {self.events[0].timestamp_sec:.1f}s, "
            f"last at {self.events[-1].timestamp_sec:.1f}s, {pinned} pinned)."
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
            "lat",
            "lon",
            "agl_m",
            "gsd_cm",
            "err_radius_m",
        ]
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for event in self.events:
                writer.writerow(asdict(event))

    def save_geojson(self, path: str | Path) -> None:
        """Write pinned detections as GeoJSON point features."""
        features = []
        for event in self.events:
            if event.lat is None or event.lon is None:
                continue
            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [event.lon, event.lat],
                    },
                    "properties": {
                        "track_id": event.track_id,
                        "frame_index": event.frame_index,
                        "timestamp_sec": event.timestamp_sec,
                        "confidence": event.confidence,
                        "snapshot_path": event.snapshot_path,
                        "agl_m": event.agl_m,
                        "gsd_cm": event.gsd_cm,
                        "err_radius_m": event.err_radius_m,
                    },
                }
            )
        Path(path).write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
            + "\n"
        )

    def save_gpx(self, path: str | Path) -> None:
        """Write pinned detections as GPX waypoints."""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1" creator="sar-detector">',
        ]
        for event in self.events:
            if event.lat is None or event.lon is None:
                continue
            name = f"person_{event.track_id:04d}"
            lines.append(
                f'  <wpt lat="{event.lat:.7f}" lon="{event.lon:.7f}">'
                f"<name>{name}</name>"
                f"<desc>t={event.timestamp_sec:.1f}s conf={event.confidence:.2f} "
                f"err={event.err_radius_m or 0.0:.1f}m</desc></wpt>"
            )
        lines.append("</gpx>")
        Path(path).write_text("\n".join(lines) + "\n")
