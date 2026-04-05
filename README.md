# TibiaFibula-CT-Seg-Unsupervised

Unsupervised tibia/fibula segmentation from ankle CT (DICOM), requiring **no manual annotation**.

## Method

K-means seed extraction → BFS region growing → connected component filtering → morphological closing/opening → enhanced tibia-fibula separation (centroid-based, adaptive to scan direction).

Output: DICOM masks spatially aligned with source CT (background=0, tibia=1, fibula=2).

## Quick Start

```bash
pip install pydicom numpy opencv-python scikit-learn matplotlib natsort pillow
```

**Batch segmentation:**
```bash
python src/pipelines/generate_masks.py \
  --input_dir /path/to/data \
  --output_dir ./outputs \
  --window_center 300 \
  --window_width 800 \
  --growth_threshold 20 \
  --separation_strategy enhanced
```

**Single slice debug:**
```bash
python src/pipelines/single_debug.py \
  --input /path/to/sample.DCM \
  --output ./debug.png
```

**Data cleaning (remove empty/discontinuous masks):**
```bash
python src/tools/data_filter.py \
  --data_root /path/to/data \
  --mode both \
  --gap_threshold 5
```

## Repository Structure

```
src/
├── segmentation/
│   ├── core.py            # preprocessing, K-means, region growing, morphology, separation
│   ├── io_dicom.py        # DICOM read/write with spatial metadata
│   └── visualization.py   # step-by-step 6-panel visualization
├── pipelines/
│   ├── generate_masks.py  # batch CLI
│   └── single_debug.py    # single-slice debug
└── tools/
    └── data_filter.py     # mask quality filter
datavisual.py              # PyTorch Dataset for downstream training
```

## Pipeline

1. HU conversion → bone window (WC=300, WW=800) → Gaussian blur → center crop (360×360)
2. K-means (K=5), top-2 brightness clusters as bone seeds
3. 4-connected BFS region growing (threshold=20)
4. Keep 2 largest connected components
5. Optional 3×3 morphological close + open
6. Enhanced tibia-fibula separation via centroid x-coordinate comparison
7. Pad back to 512×512, write DICOM segmentation instance

## Results

Validated on 19 patients / 1309 ankle CT slices:
- Batch processing success rate: **100%**
- Tibia-fibula separation accuracy: **100%**
- Average processing time: ~3s/slice (CPU only, no GPU required)

## Key Features

- **Zero annotation** — fully unsupervised, no training data needed
- **DICOM-native** — output masks preserve all spatial metadata for 3D alignment
- **Scan-direction adaptive** — enhanced strategy handles both normal and horizontally-flipped scans
- **Downstream-ready** — masks usable directly as pseudo-labels for supervised model training
