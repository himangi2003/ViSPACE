# ViP-SegD — Virchow2-Powered Segmentation & Spatial Feature Pipeline

ViP-SegD is an end-to-end computational pathology pipeline for whole-slide image (WSI) analysis. It combines the Virchow2 Vision Transformer encoder with a pixel-wise decoder to produce five-class tissue segmentation maps, then extracts rich spatial features from the tumour microenvironment (TME).

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Pipeline Stages](#pipeline-stages)
6. [Configuration Reference](#configuration-reference)
7. [Output Directory Layout](#output-directory-layout)
8. [Feature Descriptions](#feature-descriptions)
9. [Running Subsets of Stages](#running-subsets-of-stages)
10. [Dependencies](#dependencies)

---

## Overview


ViP-SegD processes a WSI in eight sequential stages:

| # | Stage | Script | Description |
|---|-------|--------|-------------|
| 1 | Tessellation | `tessellate.py` | Tile the WSI into 224 × 224 px patches using Mussel |
| 2 | Segmentation | `segmenter.py` | Run Virchow2 + pixel-wise decoder on every patch |
| 3 | Stitching | `stitch.py` | Merge per-patch masks into a slide-level GeoJSON |
| 4 | Tumour ROI Overlay | `tumor_roi_overlay.py` | Identify and cluster high-tumour-content ROI boxes |
| 5 | Cluster TSR / sTILs | `cluster_tils_tsr_scoring.py` | Compute TSR and sTILs per tumour cluster |
| 6 | Immune Proximity | `immune_proximity_features.py` | TIL–tumour boundary distance features |
| 7 | Necrosis Proximity | `necrosis_proximity_features.py` | Necrosis–tumour & necrosis–immune distance features |
| 8 | Tumour Morphology | `tumor_morphology_features.py` | Shape, fragmentation, and perimeter features per cluster |

**Segmentation classes:** Tumour · Stroma · Necrosis · Inflammatory (TILs) · Others

---

## Prerequisites

### Virchow2 — Gated HuggingFace Model

ViP-SegD uses [Virchow2](https://huggingface.co/paige-ai/Virchow2) as its encoder backbone. Virchow2 is a **gated model** on Hugging Face Hub: you must request access before the weights can be downloaded.

1. Go to the [Virchow2 model page](https://huggingface.co/paige-ai/Virchow2) and click **Request access**. Access is typically granted within minutes.
2. Install the Hugging Face CLI and authenticate:

```bash
pip install huggingface_hub
huggingface-cli login          # paste your HF access token when prompted
```

3. Once authenticated, Virchow2 weights are fetched automatically at first run. Alternatively, set `cfg.VIRCHOW2_PATH` to a local directory containing a pre-downloaded copy.

> **License note:** Virchow2 is released under the [Paige AI Research License](https://huggingface.co/paige-ai/Virchow2/blob/main/LICENSE). Review the terms before using ViP-SegD in commercial or clinical settings.

### Mussel — WSI Tessellation Backend

Stage 1 (Tessellation) depends on [Mussel](https://github.com/pathology-data-mining/Mussel), an open-source WSI tiling library. It is installed via pip directly from GitHub (see Installation step 3 below) and is already pinned in `VipsegD_requirements.txt`.

---

## Installation

### 1. Create the Conda environment

```bash
conda env create -f ViP-SegD_environment.yml
conda activate vipsegd
```

### 2. Install Python dependencies

```bash
pip install -r VipsegD_requirements.txt
```

### 3. Clone Mussel (tessellation backend)

```bash
git clone https://github.com/pathology-data-mining/Mussel.git ViP-SegD/Mussel
```

### 4. Download model weights

Download the ViP-SegD checkpoint and place it at the path specified by `cfg.CHECKPOINT` (default: `weights/phaseB_best.pt`).  
Virchow2 encoder weights are fetched automatically from Hugging Face Hub unless `cfg.VIRCHOW2_PATH` is set to a local directory.

---

## Quick Start

```python
from config import cfg
from run_vipsegd import run_vipsegd

# Set the three required fields
cfg.OUT_DIR    = "vipsegd_output"
cfg.CHECKPOINT = "TNBC_weights/TNBC_best.pt"
cfg.DATA_PATH  = "histology/"

# Run the full pipeline on one slide
results = run_vipsegd("histology/TCGA-A1-A0SP.svs", cfg)

print(results["success"])          # True if all stages completed
print(results["total_elapsed_s"])  # Wall-clock seconds
```

To run all slides in a folder:

```python
from pathlib import Path
from config import cfg
from run_vipsegd import run_vipsegd

for svs in Path(cfg.DATA_PATH).glob("*.svs"):
    run_vipsegd(str(svs), cfg)
```

---

## Pipeline Stages

### Stage 1 — Tessellation (`tessellate.py`)

Tiles the WSI using Mussel with Otsu tissue masking to skip background patches.

```python
from tessellate import run_tessellation
outdir = run_tessellation("slides/slide.svs", cfg)
```

Key parameters: `cfg.PATCH_SIZE` (default 224), `cfg.WORKERS`, `cfg.SEGMENT_THRESH`.

---

### Stage 2 — Segmentation (`segmenter.py`)

Passes each patch through the Virchow2 ViT-14 encoder (1280-d, 256 tokens → 16×16 spatial grid) and a three-stage upsampling decoder with 1×1 conv classifier. Produces a per-pixel class prediction for all five tissue classes.

Key parameters: `cfg.DEVICE`, `cfg.BATCH_SIZE`, `cfg.WHITE_THRESH`.

---

### Stage 3 — Stitching (`stitch.py`)

Places each patch mask back at its WSI coordinates and vectorises the result into a slide-level `segmentation_all_classes.geojson`. Also writes a colour-coded segmentation PNG.

Key parameters: `cfg.MPP`, `cfg.MAX_PX`, `cfg.ALPHA`.

---

### Stage 4 — Tumour ROI Overlay (`tumor_roi_overlay.py`)

Identifies 200 µm ROI boxes with ≥ 20 % tumour content, groups them into spatial clusters (8-connected, optional gap merging), and filters out necrosis-dominated or isolated tiles.

Key parameters: `cfg.ROI_SIZE_UM`, `cfg.ROI_MIN_TUMOR_FRAC`, `cfg.ROI_MAX_NECROSIS`, `cfg.ROI_MERGE_GAP_UM`.

---

### Stage 5 — Cluster TSR / sTILs Scoring (`cluster_tils_tsr_scoring.py`)

Dissolves each tumour cluster's ROI boxes into a scoring polygon (+ 200 µm buffer) and computes:

- **TSR** = Stroma / (Tumour + Stroma)
- **sTILs (Salgado)** = Inflammatory / Stroma × 100
- **sTILs (stromal)** = Inflammatory / (Stroma + Inflammatory) × 100
- **sTILs (tissue)** = Inflammatory / Viable tissue × 100

Background pixels are excluded from all denominators. A tissue-fraction reliability gate flags clusters with sparse segmentation coverage.

Key parameters: `cfg.TILS_DENOMINATOR`, `cfg.CLUSTER_BUFFER_UM`, `cfg.CLUSTER_MIN_ROI_BOXES`, `cfg.CLUSTER_MIN_TISSUE_FRACTION`.

---

### Stage 6 — Immune Proximity Features (`immune_proximity_features.py`)

For each tumour cluster, measures the spatial relationship between TIL regions and the tumour boundary:

- % TIL area within 20 / 50 / 100 / 200 µm of the tumour boundary
- Contact fraction, median distance, intratumoral TIL fraction
- Phenotype classification: **Desert · Excluded · Margin-localised · Peritumoral · Penetrated**

Key parameters: `cfg.IMMUNE_PROXIMITY_THRESHOLDS_UM`, `cfg.IMMUNE_CONTACT_TOLERANCE_UM`, `cfg.IMMUNE_PENETRATED_MIN_INTRA_FRAC`.

---

### Stage 7 — Necrosis Proximity Features (`necrosis_proximity_features.py`)

Characterises necrosis geometry and its spatial coupling with tumour and immune regions:

- % Necrosis area within 50 / 100 µm of the tumour boundary
- Necrosis–immune coupling fraction (within 100 µm)
- Shape metrics on necrotic components (convexity, elongation) for components > 500 µm²
- Phenotype classification: **Absent · Tumour-central · Peritumoral · Immune-adjacent · Stromal-distant**

Key parameters: `cfg.NECROSIS_PROXIMITY_THRESHOLDS_UM`, `cfg.NECROSIS_MIN_COMPONENT_AREA_UM2`, `cfg.NECROSIS_CENTRAL_MIN_INTRA_FRAC`.

---

### Stage 8 — Tumour Morphology Features (`tumor_morphology_features.py`)

Extracts shape and fragmentation descriptors for each tumour cluster:

- Total tumour area (µm²), perimeter, convex hull area
- Solidity, circularity, elongation (aspect ratio)
- Island count and fragmentation index
- Islands smaller than `cfg.MORPHOLOGY_MIN_ISLAND_AREA_UM2` (default 1 000 µm²) are excluded

Key parameters: `cfg.MORPHOLOGY_MIN_ISLAND_AREA_UM2`, `cfg.MORPHOLOGY_SAVE_ISLAND_QC`, `cfg.MORPHOLOGY_TUMOR_CLASS_NAMES`.

---

## Configuration Reference

All settings live in `config.py`. Import and override with `dataclasses.replace()`:

```python
from config import cfg
from dataclasses import replace

my_cfg = replace(cfg,
    OUT_DIR    = "/scratch/myproject",
    CHECKPOINT = "weights/phaseA_best.pt",
    MPP        = 0.50,   # 20× slides
    BATCH_SIZE = 16,
)
```

### Required fields

| Field | Default | Description |
|-------|---------|-------------|
| `OUT_DIR` | `"vipsegd_output"` | Root output directory |
| `CHECKPOINT` | `"weights/phaseB_best.pt"` | Path to ViP-SegD model weights |
| `DATA_PATH` | `"histology/"` | Folder containing WSI files |

### Key optional fields

| Field | Default | Description |
|-------|---------|-------------|
| `MPP` | `0.25` | Microns-per-pixel (0.25 = 40×, 0.50 = 20×) |
| `PATCH_SIZE` | `224` | Tile edge in pixels |
| `BATCH_SIZE` | `32` | GPU inference batch size |
| `DEVICE` | `"cuda"` | `"cuda"` or `"cpu"` |
| `ROI_SIZE_UM` | `200.0` | ROI box edge in microns |
| `TILS_DENOMINATOR` | `"salgado"` | sTILs denominator variant |
| `CLUSTER_BUFFER_UM` | `200.0` | Buffer around ROI cluster for scoring polygon |
| `MORPHOLOGY_MIN_ISLAND_AREA_UM2` | `1000.0` | Minimum tumour island area filter |

---

## Output Directory Layout

```
cfg.OUT_DIR/
└── <slide_name>/
    ├── tessellation/
    │   ├── patches/                       # 224×224 patch PNGs
    │   ├── <slide>.h5                     # Mussel tile manifest
    │   ├── mask.png
    │   ├── grid_mask.png
    │   └── thumbnail.png
    │
    ├── segmentation/
    │   ├── manifest.csv                   # Tile-level class fraction table
    │   ├── segmentation_all_classes.geojson
    │   └── <slide>_segmentation.png
    │
    └── spatial_feature_results/
        ├── tumor_roi_overlay/
        │   ├── tumor_roi_boxes.csv
        │   ├── tumor_roi_boxes_pseudo_thumbnail.png
        │   └── tumor_roi_boxes_wsi_thumbnail.png
        │
        ├── cluster_tils_tsr_score/
        │   ├── cluster_scoring_polygons.geojson
        │   ├── tils_tsr_by_cluster.csv
        │   ├── tils_tsr_wsi_summary.csv
        │   └── cluster_tils_tsr_overlay.png
        │
        ├── immune_proximity/
        │   ├── immune_proximity_by_cluster.csv
        │   ├── immune_proximity_wsi_summary.csv
        │   └── immune_proximity_plot.png
        │
        ├── necrosis_proximity/
        │   ├── necrosis_proximity_by_cluster.csv
        │   ├── necrosis_proximity_wsi_summary.csv
        │   ├── necrosis_distance_figure.png
        │   └── necrosis_tissue_context_figure.png
        │
        └── tumor_morphology/
            ├── tumor_core_features_by_cluster.csv
            ├── tumor_core_wsi_summary.csv
            └── tumor_island_qc.csv            # optional (cfg.MORPHOLOGY_SAVE_ISLAND_QC)
```

---

## Feature Descriptions

### TSR & sTILs (`tils_tsr_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `cluster_id` | Spatial tumour cluster identifier |
| `TSR` | Tumour-stroma ratio (stroma / tumour+stroma) |
| `sTILs_salgado` | Inflammatory / Stroma × 100 (Salgado 2015) |
| `sTILs_stromal` | Inflammatory / (Stroma + Inflammatory) × 100 |
| `sTILs_tissue` | Inflammatory / Viable tissue × 100 |
| `tissue_fraction` | Fraction of cluster polygon with segmentation coverage |

### Immune Proximity (`immune_proximity_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `pct_til_within_20um` | % TIL area within 20 µm of tumour boundary |
| `pct_til_within_50um` | % TIL area within 50 µm of tumour boundary |
| `pct_til_within_100um` | % TIL area within 100 µm of tumour boundary |
| `contact_fraction` | Fraction of TIL area within 5 µm (contact) |
| `intratumoral_til_frac` | Fraction of TIL area inside tumour |
| `immune_phenotype` | Desert / Excluded / Margin / Peritumoral / Penetrated |

### Tumour Morphology (`tumor_core_features_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `tumor_area_um2` | Total tumour area in µm² |
| `solidity` | Tumour area / convex hull area |
| `circularity` | 4π · area / perimeter² |
| `elongation` | Major axis / minor axis |
| `island_count` | Number of disconnected tumour islands |
| `fragmentation_index` | Island count / total area (normalised) |

---

## Running Subsets of Stages

Completed stages are automatically skipped (sentinel-file check). To run only specific stages:

```python
# Re-run only scoring stages (segmentation must already exist)
results = run_vipsegd(
    "histology/slide.svs", cfg,
    stages={"cluster_tils_tsr_score", "immune_proximity", "necrosis_proximity"},
)

# Force a specific stage to re-run even if output exists
results = run_vipsegd(
    "histology/slide.svs", cfg,
    force_stages={"tumor_roi_overlay"},
)

# Run a single stage via convenience function
from run_vipsegd import run_stage
result = run_stage("tumor_morphology", "histology/slide.svs", cfg, force=True)
```

Valid stage names: `tessellation` · `segmentation` · `stitching` · `tumor_roi_overlay` · `cluster_tils_tsr_score` · `immune_proximity` · `necrosis_proximity` · `tumor_morphology`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.5.1+cu121 | Model inference |
| timm | 1.0.17 | Virchow2 ViT backbone |
| Mussel | git | WSI tessellation |
| tiffslide | 2.5.1 | SVS/TIF reading |
| geopandas / shapely | 1.1.1 / 2.1.1 | GeoJSON spatial operations |
| opencv-python-headless | 4.12.0 | Image processing |
| h5py | 3.14.0 | Tile storage |
| pandas | 2.3.1 | Feature tables |
| scikit-image | 0.25.2 | Morphology helpers |

See `ViP-SegD_environment.yml` for the complete pinned environment and `VipsegD_requirements.txt` for the full dependency tree.

---

## Citation

If you use ViP-SegD in your research, please cite the associated publication (forthcoming). In the meantime, you can cite this repository directly:

```
@software{vipsegd,
  author  = {Srivastava, Himangi},
  title   = {{ViP-SegD}: Virchow2-Powered Segmentation \& Spatial Feature Pipeline},
  url     = {https://github.com/himangi2003/ViPsegD},
  year    = {2024},
}
```

Please also cite the underlying models and tools this pipeline depends on:

- **Virchow2:** Vorontsov et al., *A foundation model for clinical-grade computational pathology and biomarker discovery in oncology*, Nature Medicine 2024.
- **Mussel:** [pathology-data-mining/Mussel](https://github.com/pathology-data-mining/Mussel)

---

## License

ViP-SegD source code is released under the [MIT License](LICENSE).

Note that use of this pipeline is additionally subject to the terms of its dependencies:
- **Virchow2** model weights are governed by the [Paige AI Research License](https://huggingface.co/paige-ai/Virchow2/blob/main/LICENSE) — review before commercial or clinical use.
- **Mussel** is released under its own open-source license; see the [Mussel repository](https://github.com/pathology-data-mining/Mussel) for details.