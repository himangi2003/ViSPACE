"""
ViSPACE — Virchow2 powered Spatial Characterization And feature Extraction.

An end-to-end computational pathology pipeline for whole-slide images (WSIs):
Virchow2-powered five-class tissue segmentation plus quantitative spatial
biomarker extraction from the tumour microenvironment.

Typical use
-----------
    import vispace

    result = vispace.run_vispace(
        wsi_path="/path/to/slide.svs",
        cfg=vispace.cfg,
    )

Or configure explicitly:

    from vispace import PipelineConfig, run_vispace

    cfg = PipelineConfig(CHECKPOINT="weights/TNBC_best.pt", MPP=0.50)
    result = run_vispace("/path/to/slide.svs", cfg=cfg)
"""

from __future__ import annotations

__version__ = "0.1.0"

# --- Configuration -----------------------------------------------------------
from .config import (
    PipelineConfig,
    cfg,
    config_from_json,
    build_arg_parser,
)

# --- Environment validation --------------------------------------------------
from .environment_check import run_environment_check

# --- Top-level orchestrator --------------------------------------------------
from .run_vispace import run_vispace, run_stage

# --- Individual pipeline stages (advanced / partial runs) --------------------
from .tessellate import run_tessellation
from .segmenter import run_segmentation
from .stitch import run_stitching
from .tumor_roi_overlay import run_tumor_roi_overlay
from .cluster_tils_tsr_scoring import run_cluster_tils_tsr_score
from .immune_proximity_features import run_immune_proximity_features
from .necrosis_features import run_necrosis_features
from .tumor_morphology_features import run_tumor_morphology_features

# --- Reporting ---------------------------------------------------------------
from .generate_qmd_file import generate_report

__all__ = [
    "__version__",
    # config
    "PipelineConfig",
    "cfg",
    "config_from_json",
    "build_arg_parser",
    # environment
    "run_environment_check",
    # orchestrator
    "run_vispace",
    "run_stage",
    # stages
    "run_tessellation",
    "run_segmentation",
    "run_stitching",
    "run_tumor_roi_overlay",
    "run_cluster_tils_tsr_score",
    "run_immune_proximity_features",
    "run_necrosis_features",
    "run_tumor_morphology_features",
    # reporting
    "generate_report",
]
