# ViSPACE — Virchow2 powered Spatial Characterization and feature Extraction Pipeline

ViSPACE is an end-to-end computational pathology pipeline for analysing
whole-slide images (WSIs). The pipeline combines the Virchow2 foundation
model with a pixel-wise segmentation decoder to generate five-class tissue
segmentations and extract quantitative spatial biomarkers from the tumour
microenvironment (TME).

## Features

- End-to-end WSI analysis
- Virchow2-powered semantic segmentation
- Five tissue classes
  - Tumour
  - Stroma
  - Inflammatory (TILs)
  - Necrosis
  - Others
- Tumour ROI generation
- TSR and sTIL scoring
- Immune proximity analysis
- Necrosis proximity analysis
- Tumour morphology analysis
- Automatic HTML report generation


<p align="center">
  <img src="ViSpace_workflow.png" alt="ViSPACE pipeline architecture" width="800"><br>
  <em>Detailed pipeline architecture — tessellation through spatial feature extraction</em>
</p>

<p align="center">
  <img src="workflow2.png" alt="ViSPACE high-level workflow" width="800"><br>
  <em>High-level workflow overview</em>
</p>

---

---

# Documentation

Complete documentation is available in the **ViSPACE User Manual**.

---
# Recommended System Requirements

| Component | Recommended |
|-----------|-------------|
| Python | 3.11 |
| GPU | NVIDIA RTX 3080 / RTX 4090 / A5000 |
| CUDA | 12.1 |
| RAM | 32–64 GB |
| VRAM | ≥16 GB |
| Storage | SSD/NVMe |

ViSPACE was developed and tested using

- Python 3.11
- PyTorch 2.5.1
- CUDA 12.1
- NVIDIA RTX 4090 
---

## Table of Contents

0. [Prerequisites](#prerequisites)
1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [CLI Reference](#cli-reference)
4. [Running in a Jupyter Notebook](#running-in-a-jupyter-notebook)
5. [Pipeline Stages](#pipeline-stages)
6. [Configuration Reference](#configuration-reference)
7. [Output Directory Layout](#output-directory-layout)
8. [Feature Descriptions](#feature-descriptions)
9. [Running Subsets of Stages](#running-subsets-of-stages)
10. [Dependencies](#dependencies)
11. [Citation](#citation)
12. [License](#license)

---

## Prerequisites

# Installation


```bash
conda env create -f ViSPACE_environment.yml
conda activate vispace
```

## Google Colab

```bash
pip install -q -r ViSpace_requirements-colab.txt
```

See **Chapter 1** of the User Manual for complete installation instructions.

---

# Hugging Face Authentication

Virchow2 is a **gated model** — you need a HuggingFace account and access token before the weights can be downloaded (skip this entirely if you set `VIRCHOW2_PATH` to local weights instead).

Authenticate once using

*Method A — CLI (recommended for servers & scripts):*

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted — input is hidden, this is expected
huggingface-cli whoami   # verify
```

To persist across sessions:

```bash
echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc   # or ~/.zshrc
source ~/.bashrc
```

*Method B — Jupyter notebook cell* (see `Vispace_tutorial.ipynb` for the full walkthrough):

```python
import os
from huggingface_hub import whoami

# ✏ Paste your HF access token here (Read role is sufficient)
HF_ACCESS_TOKEN = "hf_your_token_here"

user = whoami(token=HF_ACCESS_TOKEN)
print(f"Authenticated as: {user['name']}")

os.environ["HF_TOKEN"] = HF_ACCESS_TOKEN
print("HF_TOKEN environment variable set.")
```

Alternatively, local Virchow2 weights can be specified using
`VIRCHOW2_PATH`.

Complete instructions are provided in **Chapter 1** of the User Manual.

---

# Environment Check

Before processing any slide, verify the installation:

```bash
python environment_check.py
```

This script validates

- Python environment
- CUDA availability
- PyTorch installation
- Virchow2 dependencies
- Mussel installation
- Model checkpoints
- Output directories

See **Chapter 2** for further details.


## Overview

Vispace processes a WSI in eight sequential stages, orchestrated end-to-end by `run_vispace.py`:

| # | Stage | Script | Description |
|---|-------|--------|-------------|
| 1 | Tessellation | `tessellate.py` | Tile the WSI into 224×224 px patches using Mussel |
| 2 | Segmentation | `segmenter.py` | Run Virchow2 + pixel-wise decoder on every patch |
| 3 | Stitching | `stitch.py` | Merge per-patch masks into a slide-level GeoJSON |
| 4 | Tumour ROI Overlay | `tumor_roi_overlay.py` | Identify and cluster high-tumour-content ROI boxes |
| 5 | Cluster TSR / sTILs | `cluster_tils_tsr_score.py` | Compute TSR and sTILs per tumour cluster |
| 6 | Immune Proximity | `immune_proximity_features.py` | TIL–tumour boundary distance features |
| 7 | Necrosis Proximity | `necrosis_proximity_features.py` | Necrosis–tumour & necrosis–immune distance features |
| 8 | Tumour Morphology | `tumor_morphology_features.py` | Shape, fragmentation, and perimeter features per cluster |

**Segmentation classes:** Tumour · Stroma · Necrosis · Inflammatory (TILs) · Others

**Orchestration & utilities** (not pipeline stages themselves):

| Script | Purpose |
|---|---|
| `config.py` | Build and save a run configuration |
| `environment_check.py` | Pre-flight dependency / GPU / path validation |
| `run_vispace.py` | Runs all 8 stages, resumable, subset-capable |
| `report/generate_report.py` | Renders an HTML report (Quarto) from a completed run |

---

## Quick Start

```python
from config import PipelineConfig
from run_vispace import run_vispace

cfg = PipelineConfig(
    OUT_DIR    = "vispace_output",
    CHECKPOINT = "TNBC_weights/TNBC_best.pt",
    WSI_PATH   = "slides/your_slide.svs",
    MUSSEL_DIR = "Mussel/",
)

results = run_vispace(cfg.WSI_PATH, cfg)

print(results["success"])          # True if all stages completed
print(results["total_elapsed_s"])  # Wall-clock seconds
```

To run every slide in a folder, loop over the files yourself and swap in each path — `WSI_PATH` is per-slide, not a folder setting:

```python
from pathlib import Path
from dataclasses import replace
from config import PipelineConfig
from run_vispace import run_vispace

base_cfg = PipelineConfig(OUT_DIR="vispace_output", CHECKPOINT="TNBC_weights/TNBC_best.pt", MUSSEL_DIR="Mussel/")

for svs in Path("slides/").glob("*.svs"):
    cfg = replace(base_cfg, WSI_PATH=str(svs))
    run_vispace(str(svs), cfg)
```

---

## CLI Reference

All scripts share the same conventions: every field on `PipelineConfig` is available as a `--flag`, and every script accepts `--from-json run_config.json` to reuse a config saved once via `config.py --print-config`. Run any script with `--help` to see its full flag list.

### 0. Build the config once

```bash
python config.py \
    --wsi-path slides/TCGA-A1-A0SP.svs \
    --checkpoint TNBC_weights/TNBC_best.pt \
    --mussel-dir Mussel/ \
    --print-config > run_config.json
```

### 1. Pre-flight environment check *(recommended before a long run)*

```bash
python environment_check.py --from-json run_config.json
```

### 2. Authenticate with HuggingFace *(skip if using local Virchow2 weights)*

```bash
huggingface-cli login
```

### 3. Run the full pipeline in one command

```bash
python run_vispace.py --from-json run_config.json
```

```bash
# run only a subset of stages (prerequisites must already exist)
python run_vispace.py --from-json run_config.json \
    --stages cluster_tils_tsr_score,immune_proximity,necrosis_proximity

# force specific stages to re-run even if output exists
python run_vispace.py --from-json run_config.json \
    --force-stages tumor_roi_overlay
```

**— or —** run each stage individually:

```bash
# 3a. tessellate
python tessellate.py --from-json run_config.json

# 3b. segment
python segmenter.py --from-json run_config.json
python segmenter.py --from-json run_config.json --batch-size 96   # override example

# 3c. stitch + GeoJSON
python stitch.py --from-json run_config.json
python stitch.py --from-json run_config.json --min-area-px 200    # override example

# 3d. tumor ROI clustering + overlay
python tumor_roi_overlay.py --from-json run_config.json
python tumor_roi_overlay.py --from-json run_config.json --roi-size-um 150

# 3e. cluster TSR / sTILs scoring
python cluster_tils_tsr_score.py --from-json run_config.json
python cluster_tils_tsr_score.py --from-json run_config.json --tils-denominator tissue

# 3f. immune / TIL proximity features
python immune_proximity_features.py --from-json run_config.json
python immune_proximity_features.py --from-json run_config.json --immune-contact-tolerance-um 10

# 3g. necrosis proximity features
python necrosis_proximity_features.py --from-json run_config.json
python necrosis_proximity_features.py --from-json run_config.json --necrosis-immune-coupling-threshold-um 150

# 3h. tumor morphology features
python tumor_morphology_features.py --from-json run_config.json
python tumor_morphology_features.py --from-json run_config.json --morphology-min-island-area-um2 500
```



### Full command reference table

| # | Script | Purpose | Minimal command |
|---|---|---|---|
| 0 | `config.py` | Build + save the run config | `python config.py --wsi-path ... --checkpoint ... --print-config > run_config.json` |
| — | `environment_check.py` | Validate dependencies, GPU, paths before running | `python environment_check.py --from-json run_config.json` |
| 1 | `tessellate.py` | Tile the WSI (Mussel) | `python tessellate.py --from-json run_config.json` |
| 2 | `segmenter.py` | Run Virchow2 segmentation inference | `python segmenter.py --from-json run_config.json` |
| 3 | `stitch.py` | Stitch tiles → WSI canvas + GeoJSON | `python stitch.py --from-json run_config.json` |
| 4 | `tumor_roi_overlay.py` | Cluster tumor tiles, build ROI boxes | `python tumor_roi_overlay.py --from-json run_config.json` |
| 5 | `cluster_tils_tsr_score.py` | TSR + sTILs scoring per cluster | `python cluster_tils_tsr_score.py --from-json run_config.json` |
| 6 | `immune_proximity_features.py` | TIL proximity to tumor boundary | `python immune_proximity_features.py --from-json run_config.json` |
| 7 | `necrosis_proximity_features.py` | Necrosis proximity + phenotyping | `python necrosis_proximity_features.py --from-json run_config.json` |
| 8 | `tumor_morphology_features.py` | Tumor shape / fragmentation features | `python tumor_morphology_features.py --from-json run_config.json` |
| — | `run_vispace.py` | Orchestrates stages 1–8, resumable | `python run_vispace.py --from-json run_config.json` |
| — | `generate_qmd_file.py` | Render the HTML report (Quarto) qmd file for QUARTO rendering later| `python generate_report.py` |

Every stage script also works as a Python import — see the next section.

---

## Running in a Jupyter Notebook

A full walkthrough notebook is provided at `ViSpace_tutorial.ipynb`. The short version: every stage exposes a plain Python function (`run_*(wsi_path, cfg)`), so notebook cells can call them directly instead of shelling out with `!`. `cfg` stays in memory across cells — no need to save/reload `run_config.json` within a single session.

## Running Subsets of Stages

Completed stages are automatically skipped (sentinel-file check). To run only specific stages:

```python
# Re-run only scoring stages (segmentation must already exist)
results = run_vispace(
    "histology/slide.svs", cfg,
    stages={"cluster_tils_tsr_score", "immune_proximity", "necrosis_proximity"},
)

# Force a specific stage to re-run even if output exists
results = run_vispace(
    "histology/slide.svs", cfg,
    force_stages={"tumor_roi_overlay"},
)

# Run a single stage via convenience function
from run_vispace import run_stage
result = run_stage("tumor_morphology", "histology/slide.svs", cfg, force=True)
```

Valid stage names: `tessellation` · `segmentation` · `stitching` · `tumor_roi_overlay` · `cluster_tils_tsr_score` · `immune_proximity` · `necrosis_proximity` · `tumor_morphology`

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
        │   └── tumor_roi_boxes_wsi_thumbnail.png   # only if the WSI file exists on disk
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
            └── tumor_island_qc.csv            # only if MORPHOLOGY_SAVE_ISLAND_QC = True
```

The rendered HTML report is saved separately, alongside `report/report.qmd`:

```
Vispace_report_slide.html
```

---

## Feature Descriptions

Column names below match the actual CSV headers written by each script. Each table shows the most commonly used columns — see the CSV itself for the complete set.

### TSR & sTILs (`tils_tsr_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `cluster_id` | Spatial tumour cluster identifier |
| `TSR_display` | Human-readable `"tumour%/stroma%"` string |
| `TSR_stroma_fraction` | Raw TSR value: stroma / (tumour + stroma) |
| `TSR_category` | `stroma-high` / `stroma-low` / `unreliable` |
| `sTILs_pct_salgado` | Inflammatory / Stroma × 100 (Salgado 2015) |
| `sTILs_pct_stromal` | Inflammatory / (Stroma + Inflammatory) × 100 |
| `sTILs_pct_tissue` | Inflammatory / viable tissue × 100 |
| `sTILs_level` | `very low` / `low` / `intermediate` / `high` / `unreliable` |
| `tissue_fraction` | Fraction of the cluster polygon with segmentation coverage |

### Immune Proximity (`immune_proximity_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `til_pct_within_20um` / `_50um` / `_100um` / `_200um` | % TIL area within each distance of the tumour boundary |
| `til_contact_fraction` | Fraction of TIL area within the contact tolerance (default 5 µm) |
| `til_fraction_intratumoral` | Fraction of TIL area located inside the tumour |
| `til_extratumoral_distance_aw_median_um` | Area-weighted median distance for extratumoral TILs |
| `immune_phenotype` | `immune-desert` / `immune-excluded` / `margin-localized` / `peritumoral` / `immune-penetrated` |

### Necrosis Proximity (`necrosis_proximity_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `necrosis_pct_within_50um` / `_100um` | % necrosis area within each distance of the tumour boundary |
| `necrosis_immune_coupling_index` | Fraction of necrosis area within the immune-coupling threshold |
| `necrosis_fraction_intratumoral` | Fraction of necrosis located inside the tumour |
| `necrosis_phenotype` | `necrosis-absent` / `tumour-central` / `peritumoural` / `immune-adjacent` / `stral-distant` |

### Tumour Morphology (`tumor_core_features_by_cluster.csv`)

| Column | Description |
|--------|-------------|
| `tumor_area_um2` | Total tumour area in µm² |
| `tumor_solidity_mean` | Area-weighted mean of (island area / island convex-hull area) |
| `tumor_compactness_mean` | Area-weighted mean of 4π·area / perimeter² per island |
| `tumor_elongation_mean` | Area-weighted mean major/minor axis ratio |
| `tumor_n_islands` | Number of disconnected tumour islands (above the min-area filter) |
| `tumor_fragmentation_index` | `1 − largest_patch_index`; higher = more fragmented |

---

## Citation

If you use Vispace in your research, please cite the associated publication (forthcoming).

---

## License

See `LICENSE` for terms of use.