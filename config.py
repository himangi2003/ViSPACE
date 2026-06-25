"""
config.py
=========
Central configuration for the ViP-SegD pipeline.
All paths, hyperparameters, and flags live here.
Import this in every pipeline script instead of hardcoding values.

Usage
-----
    from config import cfg, PipelineConfig

    # Use the default singleton
    from run_vipsegd import run_vipsegd
    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg)

    # Override specific fields for a one-off run
    from dataclasses import replace
    cfg2 = replace(cfg, CHECKPOINT="weights/phaseA_best.pt", MPP=0.50)
    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """
    Single configuration object consumed by every pipeline stage.

    All pipeline scripts read attributes directly (e.g. cfg.CHECKPOINT) or
    via getattr with a safe fallback.  Every attribute used anywhere in the
    pipeline is defined here with its default value so that the singleton
    ``cfg`` works out-of-the-box and the user only needs to set the three
    ✏  REQUIRED fields.
    """

    # ── ✏  REQUIRED — set these before running ─────────────────────────────
    OUT_DIR:    str = "vipsegd_output"          # root directory for all outputs
    CHECKPOINT: str = "TNBC_weights/TNBC_best.pt"  # path to ViP-SegD model checkpoint
    WSI_PATH:  str = "your data path"              #  .svs / .tif slides

    # ── Mussel (tessellation) ───────────────────────────────────────────────
    MUSSEL_DIR:     str   = "Mussel/"  # cloned Mussel repo root
    PATCH_SIZE:     int   = 224                 # tile edge in pixels
    WORKERS:        int   = 4                   # parallel tiling workers
    SEGMENT_THRESH: int   = 20                  # Otsu tissue-mask threshold
    THUMBNAIL_SIZE: tuple = (1024, 1024)        # slide thumbnail dimensions

    # ── Model / inference ───────────────────────────────────────────────────
    VIRCHOW2_PATH:  str   = None                # local Virchow2 weights; None = HF Hub
    DEVICE:         str   = "cuda"              # "cuda" or "cpu"
    BATCH_SIZE:     int   = 32
    WHITE_THRESH:   int   = 220                 # background pixel threshold

    # ── Slide geometry ──────────────────────────────────────────────────────
    MPP:            float = 0.25                # microns-per-pixel  (40x TCGA)
    STEP_X:         int   = 444                 # WSI tile spacing x (px) — informational
    STEP_Y:         int   = 444                 # WSI tile spacing y (px) — informational
    MAX_PX:         int   = 4096                # max output PNG long edge

    # ── Stitch / visualisation ──────────────────────────────────────────────
    ALPHA:          float = 0.6                 # overlay blend weight
    THUMB_MAX:      int   = 2048                # thumbnail max size

    # ── Stage 4: Tumor ROI Overlay ──────────────────────────────────────────
    ROI_SIZE_UM:           float = 200.0   # ROI box edge in microns
    ROI_MIN_TUMOR_FRAC:    float = 0.20    # min frac_Tumour to keep a tile
    ROI_MAX_NECROSIS:      float = 0.50    # max mean frac_Necrosis inside a box
    ROI_MIN_CLUSTER_TILES: int   = 3       # min tiles to keep a tumour clump
    ROI_CONNECTIVITY:      int   = 8       # grid connectivity: 4 or 8
    ROI_MERGE_GAP_UM:      float = 60.0    # clumps within this gap (µm) are merged
    ROI_THUMB_WIDTH:       int   = 1800    # WSI thumbnail width for overlay PNG
    ROI_COLOR_BY_CLUSTER:  bool  = True    # colour ROI boxes by cluster ID

    # ── Stage 5: Cluster TSR / sTILs scoring ───────────────────────────────
    TILS_DENOMINATOR:            str   = "salgado"
    #   "salgado"                   → Inflammatory / Stroma             (recommended)
    #   "stroma_plus_inflammatory"  → Inflammatory / (Stroma + Inflam)
    #   "tissue"                    → Inflammatory / viable tissue
    CLUSTER_BUFFER_UM:           float = 200.0  # buffer around ROI boxes → scoring polygon
    CLUSTER_MIN_ROI_BOXES:       int   = 5      # min ROI boxes to keep a cluster
    CLUSTER_MIN_POLYGON_AREA_PX2: float = 1.0   # minimum GeoJSON polygon area filter
    CLUSTER_MAX_AREA_QUANTILE:   float = 1.0    # upper area quantile filter (1.0 = off)
    CLUSTER_MAX_AREA_MAD_Z:      float = 0.0    # MAD-z outlier rejection (0.0 = off)
    CLUSTER_MIN_TISSUE_FRACTION: float = 0.10   # reliability gate: min segmented fraction
    CLUSTER_MIN_TSR_DENOM_PX2:   float = 5000.0 # min TSR denominator area (px²)
    CLUSTER_MIN_TILS_DENOM_PX2:  float = 5000.0 # min sTILs denominator area (px²)
    CLUSTER_NO_OVERLAY:          bool  = False   # skip rendering the overlay PNG

    # ── Stage 6: Immune proximity features ─────────────────────────────────
    IMMUNE_PROXIMITY_THRESHOLDS_UM:   list  = field(default_factory=lambda: [20, 50, 100, 200])
    #   % of TIL area within each threshold distance from the tumour boundary
    IMMUNE_CONTACT_TOLERANCE_UM:      float = 5.0    # TIL within this = "contact"
    IMMUNE_DESERT_MAX_AREA_UM2:       float = 1_000.0  # < this → immune-desert phenotype
    IMMUNE_PENETRATED_MIN_INTRA_FRAC: float = 0.30   # ≥ this intratumoral → penetrated
    IMMUNE_MARGIN_MIN_PCT_50UM:       float = 50.0   # ≥ this % within 50µm → margin-localized
    IMMUNE_EXCLUDED_MIN_MEDIAN_UM:    float = 100.0  # median ≥ this → immune-excluded
    IMMUNE_PERITUMORAL_MAX_MEDIAN_UM: float = 100.0  # median < this → peritumoral

    # ── Stage 7: Necrosis proximity features ───────────────────────────────
    NECROSIS_PROXIMITY_THRESHOLDS_UM:      list  = field(default_factory=lambda: [50.0, 100.0])
    NECROSIS_CONTACT_TOLERANCE_UM:         float = 5.0    # necrosis within this = "contact"
    NECROSIS_IMMUNE_COUPLING_THRESHOLD_UM: float = 100.0  # necrosis within this of immune = coupled
    NECROSIS_MIN_COMPONENT_AREA_UM2:       float = 500.0  # noise filter for shape analysis
    NECROSIS_ABSENT_MAX_AREA_UM2:          float = 500.0  # < this → necrosis-absent phenotype
    NECROSIS_CENTRAL_MIN_INTRA_FRAC:       float = 0.50   # ≥ this intratumoral → tumour-central
    NECROSIS_PERITUMOURAL_MIN_PCT_100UM:   float = 60.0   # ≥ this % within 100µm → peritumoural
    NECROSIS_IMMUNE_ADJACENT_MIN_COUPLING: float = 0.20   # ≥ this coupling → immune-adjacent
    NECROSIS_DISTANT_MIN_MEDIAN_UM:        float = 150.0  # median ≥ this → stromal-distant

    # ── Stage 8: Tumour morphology features ────────────────────────────────
    MORPHOLOGY_MIN_ISLAND_AREA_UM2: float = 1_000.0
    #   Islands smaller than this are excluded from shape and fragmentation
    #   metrics.  Default 1000 µm² ≈ 32µm side ≈ 5 cells.  Set to 0 to keep
    #   all islands (not recommended — single-cell fragments distort metrics).
    #   Always report this value alongside any shape metric you publish.
    MORPHOLOGY_SAVE_ISLAND_QC:      bool  = False
    #   If True, write tumor_island_qc.csv with per-island geometry stats.
    #   Useful for choosing an appropriate MORPHOLOGY_MIN_ISLAND_AREA_UM2.
    MORPHOLOGY_TUMOR_CLASS_NAMES:   set   = field(
        default_factory=lambda: {"Tumour", "Tumor", "tumour", "tumor"}
    )
    #   Set of class name strings that count as tumour in the segmentation
    #   GeoJSON.  Extend if your pipeline uses a different naming convention.

    # ── Convenience properties (read-only) ─────────────────────────────────

    @property
    def out_dir(self) -> Path:
        """cfg.OUT_DIR as a Path object."""
        return Path(self.OUT_DIR)

    @property
    def data_path(self) -> Path:
        """cfg.DATA_PATH as a Path object."""
        return Path(self.WSI_PATH)

    @property
    def cohort_features_csv(self) -> str:
        """Path to the merged cohort-level feature CSV."""
        return str(self.out_dir / "cohort_features.csv")

    # ── Per-slide path helpers ──────────────────────────────────────────────

    def slide_outdir(self, slide_name: str) -> Path:
        """Root output directory for one slide: OUT_DIR/<slide_name>/"""
        return self.out_dir / slide_name

    def slide_segmentation_dir(self, slide_name: str) -> Path:
        """Segmentation directory: OUT_DIR/<slide_name>/segmentation/"""
        return self.slide_outdir(slide_name) / "segmentation"

    def slide_tessellation_dir(self, slide_name: str) -> Path:
        """Tessellation directory: OUT_DIR/<slide_name>/tessellation/"""
        return self.slide_outdir(slide_name) / "tessellation"

    def slide_spatial_dir(self, slide_name: str) -> Path:
        """Spatial features root: OUT_DIR/<slide_name>/spatial_feature_results/"""
        return self.slide_outdir(slide_name) / "spatial_feature_results"

    def slide_geojson(self, slide_name: str) -> Path:
        """Path to the stitched segmentation GeoJSON for one slide."""
        return self.slide_segmentation_dir(slide_name) / "segmentation_all_classes.geojson"

    def slide_manifest(self, slide_name: str) -> Path:
        """Path to the tile manifest CSV for one slide."""
        return self.slide_segmentation_dir(slide_name) / "manifest.csv"

    def slide_cluster_geojson(self, slide_name: str) -> Path:
        """Path to the cluster scoring polygon GeoJSON for one slide."""
        return (
            self.slide_spatial_dir(slide_name)
            / "cluster_tils_tsr_score"
            / "cluster_scoring_polygons.geojson"
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
# Import this everywhere:
#
#     from config import cfg
#
# Override individual fields with dataclasses.replace():
#
#     from dataclasses import replace
#     my_cfg = replace(cfg, OUT_DIR="/scratch/myrun", MPP=0.50)
#
cfg = PipelineConfig()