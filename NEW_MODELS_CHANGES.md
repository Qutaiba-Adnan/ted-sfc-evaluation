# Changes After Adding Faster R-CNN and DETR

This file records what changed compared with the original TED-SFC pipeline.

## What stayed original

- The existing saliency model paths are unchanged: `mlnet`, `tasednet`, and `transalnet`.
- The existing optical-flow path is unchanged.
- The grid layout from the original dataset configs is reused.
- The original `cell_values.csv` format is reused.
- The original Morton-code generation in `src/morton.py` is reused.
- The original Morton-code event detector in `src/detector_morton.py` is still the detector used by the pipeline.
- The original evaluation flow is reused when `--annotations-path` is provided.

## What was added

- Added `src/grid_object_detection.py`.
  - Supports `faster-rcnn` and `detr`.
  - Uses pretrained COCO models.
  - Keeps only `person` detections.
  - Maps each frame's strongest person detection to the existing grid.
  - Saves the same `cell_values.csv` format expected by the original SFC steps.
  - Saves an annotated grid video and cell-value plots, following the existing grid modules.

- Updated `src/pipeline.py`.
  - Added `faster-rcnn` and `detr` to the `--method` choices.
  - Routes those methods through the new object-detection grid step.
  - Keeps the downstream Morton-code and evaluation steps unchanged.

- Updated `src/detector_morton.py`.
  - Allows the detector to choose `attention`, `optical_flow`, or `object_detection` cell ranges.
  - Preserves backwards compatibility with the previous boolean call style.
  - Adds a CLI flag for `--object-detection`.

- Updated dataset configs.
  - Added object-detection score thresholds under `grid_config.object_detection`.
  - Added `detector_config.object_detection.cell_ranges` for ZOD and Waymo.
  - Removed the old synthetic dataset config.
  - The object-detection cell ranges match the one-hot cell values produced by the new grid step.

- Updated `README.md`.
  - Added `faster-rcnn` and `detr` to the documented pipeline command.
  - Added a short note explaining that the two new methods use object detection before the original SFC detector.

## New commands

```bash
python src/pipeline.py -d path/to/dataset -o path/to/output -c config/zod/pedestrian_crossing.yml -m faster-rcnn --cpu
python src/pipeline.py -d path/to/dataset -o path/to/output -c config/waymo/pedestrian_crossing.yml -m detr --cpu
```

Remove `--cpu` to use CUDA when available.
