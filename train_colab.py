"""Fine-tune a YOLO model for aerial small-person detection, on Google Colab.

This is NOT meant to be run with `python train_colab.py` locally — it's
written as a sequence of cells to paste into a Google Colab notebook (a free
GPU is exactly what fine-tuning needs; see the "what is Colab" explanation
from earlier in this project). Each `# %% [markdown]` / `# %%` marker below
is one notebook cell — most editors (VS Code, PyCharm, Colab's own paste
detection) will render `# %%` as a cell boundary automatically. If you'd
rather do it by hand: create a new Colab notebook, then copy everything
between two consecutive `# %%` markers into its own cell, in order.

Runtime -> Change runtime type -> GPU (T4 is enough) before running.
"""

# %% [markdown]
# ## 1. Install dependencies

# %%
!pip install -q ultralytics supervision roboflow

# %% [markdown]
# ## 2. Download an aerial person-detection dataset from Roboflow Universe
#
# Search https://universe.roboflow.com for a dataset matching your case —
# e.g. search "aerial person detection", "VisDrone", "SARD" (Search And
# Rescue Dataset), or "drone person detection". Pick one exported in
# "YOLOv8" format, open its "Download this Dataset" panel, and copy the
# `rf.workspace(...).project(...).version(...)` snippet it gives you — it
# will look like the example below (this exact project/version won't
# necessarily exist; replace it with the one you picked).

# %%
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # free account -> Settings -> API Keys
project = rf.workspace("some-workspace").project("aerial-person-detection")
dataset = project.version(1).download("yolov8")
print(dataset.location)

# %% [markdown]
# ## 3. (Optional but recommended) Re-tile the training images with TrainingSlicer
#
# Aerial/drone images are often very large (4K+) with people occupying only
# a handful of pixels. Training YOLO directly on the full-resolution image
# downscaled to its usual 640x640 input shrinks people down to almost
# nothing. Slicing each training image into tiles first — the same
# small-object problem `TrainingSlicer` was built to solve — means each
# person occupies a much larger fraction of what the model actually sees.
#
# This cell assumes your downloaded dataset is in standard YOLO format
# (`images/`, `labels/` with one `.txt` per image, normalized `xywh`). It
# converts each image+labels pair to `sv.Detections`, slices both with
# `TrainingSlicer`, and writes the tiles out as a new YOLO-format dataset.
# Skip this cell if your dataset images are already small drone crops
# rather than full flight-altitude frames.

# %%
import os
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

SOURCE_DIR = Path(dataset.location) / "train"
TILED_DIR = Path(dataset.location) / "train_tiled"
SLICE_WH = 640

slicer = sv.TrainingSlicer(slice_wh=SLICE_WH, overlap_wh=64, min_visibility=0.2)


def read_yolo_labels(label_path: Path, image_wh: tuple[int, int]) -> sv.Detections:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return sv.Detections.empty()
    width, height = image_wh
    rows = np.loadtxt(label_path, ndmin=2)
    class_ids = rows[:, 0].astype(int)
    # Columns are [class, cx, cy, w, h]; cx/w are normalized by width (columns
    # 1 and 3), cy/h by height (columns 2 and 4).
    cx, cy, w, h = (rows[:, i] * (width if i in (1, 3) else height) for i in (1, 2, 3, 4))
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return sv.Detections(xyxy=xyxy, class_id=class_ids)


def write_yolo_labels(label_path: Path, detections: sv.Detections, image_wh: tuple[int, int]) -> None:
    width, height = image_wh
    lines = []
    for (x1, y1, x2, y2), class_id in zip(detections.xyxy, detections.class_id):
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


(TILED_DIR / "images").mkdir(parents=True, exist_ok=True)
(TILED_DIR / "labels").mkdir(parents=True, exist_ok=True)

image_paths = sorted((SOURCE_DIR / "images").glob("*.jpg"))
for image_path in image_paths:
    image = cv2.imread(str(image_path))
    image_wh = (image.shape[1], image.shape[0])
    label_path = SOURCE_DIR / "labels" / f"{image_path.stem}.txt"
    detections = read_yolo_labels(label_path, image_wh)

    for tile_index, (tile_image, tile_detections) in enumerate(slicer(image, detections)):
        tile_stem = f"{image_path.stem}_tile{tile_index}"
        cv2.imwrite(str(TILED_DIR / "images" / f"{tile_stem}.jpg"), tile_image)
        write_yolo_labels(
            TILED_DIR / "labels" / f"{tile_stem}.txt",
            tile_detections,
            (SLICE_WH, SLICE_WH),
        )

print(f"Wrote {len(list((TILED_DIR / 'images').glob('*.jpg')))} tiles to {TILED_DIR}")

# %% [markdown]
# Repeat the same loop for the `valid/` (and `test/`, if present) splits,
# swapping `SOURCE_DIR`/`TILED_DIR` accordingly — then point the `data.yaml`
# used below at the `*_tiled` directories instead of the originals.

# %% [markdown]
# ## 4. Fine-tune YOLO
#
# Starting from a pretrained checkpoint (`yolo11n.pt`) instead of random
# weights is what makes this feasible on a free Colab GPU in well under an
# hour for a dataset of a few thousand images — the model already knows
# general shapes/edges/textures, and fine-tuning only has to specialize it
# to aerial viewpoints and small people.

# %%
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",  # or f"{TILED_DIR}/../data.yaml" if you re-tiled
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    device=0,
    project="sar-person-detector",
    name="run1",
)

# %% [markdown]
# ## 5. Evaluate and export

# %%
metrics = model.val()
print(metrics.box.map, metrics.box.map50)

# %% [markdown]
# The best checkpoint is saved to
# `sar-person-detector/run1/weights/best.pt`. Download it (Colab's file
# browser on the left, or `files.download(...)` below) and pass it to
# `main.py` on your own machine via `--model best.pt`.

# %%
from google.colab import files

files.download("sar-person-detector/run1/weights/best.pt")
