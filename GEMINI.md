# Gemini Project Context: Defect Detection YOLO

This project is a deep learning framework for industrial defect detection, specifically optimized for small target detection and noisy datasets. It is based on the Ultralytics YOLOv8 architecture with significant modifications.

## Project Overview

- **Core Objective:** Detect defects in industrial components (e.g., steel strips, insulators) with a focus on small target accuracy and robustness against annotation noise.
- **Key Technologies:**
  - **Framework:** Python 3.13+, [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
  - **Environment Management:** `uv` (preferred), `pip`.
  - **Key Modules:**
    - **SPDConv (Space-to-Depth Convolution):** Replaces standard strided convolutions to prevent feature loss of small defects.
    - **BiFPN (Bidirectional Feature Pyramid Network):** Optimized feature fusion modules (`BiFPN_Add`, `BiFPN_Concat`) with learnable weights for cross-scale feature aggregation.
    - **WIoU-v3 (Wise-IoU):** A dynamic non-monotonic focusing loss function that mitigates the impact of outlier/noisy samples.
    - **BiFPN:** Enhanced feature pyramid for cross-scale fusion.

## Directory Structure

- `models/`: Contains model architecture (`yolo_improved.yaml`) and custom modules (`modules/spdconv.py`).
- `utils/`: Core algorithmic improvements, notably `loss.py` containing `WIoU_Loss`.
- `datasets/`: Data management scripts and configurations.
- `eval/`: Specialized scripts for generating ablation study results and academic plots.
- `docs/`: Critical documentation for hyperparameter tuning and architecture diagrams.

## Building and Running

### Environment Setup
The project uses `uv` for lightning-fast dependency management.
```bash
uv sync
```

### Data Preparation
Convert VOC annotations to YOLO format:
```bash
uv run datasets/convert_voc_to_yolo.py
```

### Training
Train the improved model using the custom configuration:
```bash
uv run train.py --cfg models/yolo_improved.yaml --data datasets/data.yaml --epochs 100 --batch 16
```

### Validation
Evaluate the trained model on the test set:
```bash
uv run val.py --weights runs/train/exp_improved/weights/best.pt --data datasets/data.yaml
```

### Evaluation & Plotting
Generate metrics and plots for reports/thesis:
```bash
uv run eval/evaluation.py
uv run eval/plotting.py
```

## Development Conventions

### Model Architecture
- All custom layers (e.g., `SPDConv`) should be defined in `models/modules/` and exported via `models/__init__.py`.
- Architecture changes must be reflected in `models/yolo_improved.yaml`.

### Loss Function
- `WIoU_Loss` is implemented in `utils/loss.py`. When tuning for specific datasets, adjust `alpha` and `delta` parameters as described in `docs/hyperparameter_tuning_guide.md`.

### Hyperparameter Tuning (Crucial)
Refer to `docs/hyperparameter_tuning_guide.md` for specific guidance on:
- **Lower Learning Rates:** Use `lr0=0.001` for stability.
- **Longer Warmup:** Essential for `WIoU_Loss` to stabilize its moving average IoU.
- **IoU Thresholds:** Lower `iou_t` (e.g., 0.4) is often necessary for matching very small targets.
- **Data Augmentation:** Mosaic is highly recommended; MixUp should be used sparingly or disabled for textured industrial backgrounds.

### Coding Style
- Follow PEP 8 standards.
- Use type hints for module implementations (see `models/modules/spdconv.py`).
- Documentation should be provided in both code comments and Markdown files in `docs/`.
