"""
config.py
=========
Central configuration for the Vispace pipeline.

All paths, hyperparameters, and flags live here.
Import this in every pipeline script instead of hardcoding values.

Usage (as a library)
---------------------
    from vispace import cfg, PipelineConfig
    # Use the default singleton
    from vispace import run_vispace
    results = run_vispace("slides/TCGA-A1-A0SP.svs", cfg)

    # Override specific fields for a one-off run
    from dataclasses import replace
    cfg2 = replace(cfg, CHECKPOINT="weights/phaseA_best.pt", MPP=0.50)
    results = run_vispace("slides/TCGA-A1-A0SP.svs", cfg2)

Usage (from the command line)
------------------------------
Every field on PipelineConfig is auto-exposed as a `--flag`, so you never
have to hand-maintain a second copy of the arguments list.

    # See every available flag with its type and default
    python config.py --help

    # Build a config, print it as JSON, and exit
    python config.py --wsi-path slides/TCGA-A1-A0SP.svs \\
                      --checkpoint weights/phaseA_best.pt \\
                      --mpp 0.50 --print-config

    # Save that JSON to a file so you can reuse it later
    python config.py --wsi-path slides/TCGA-A1-A0SP.svs \\
                      --checkpoint weights/phaseA_best.pt \\
                      --mpp 0.50 --print-config > run_config.json

    # Build a config and actually run the pipeline
    # (requires run_vispace.py to be importable on PYTHONPATH)
    python config.py --wsi-path slides/TCGA-A1-A0SP.svs \\
                      --checkpoint weights/phaseA_best.pt \\
                      --out-dir vispace_output --run

    # Later: reload a previously saved config and run from it directly
    python config.py --from-json run_config.json --run

    # Reload a saved config but override one or two fields at run time
    python config.py --from-json run_config.json --device cpu --run

    # List-, tuple- and set-typed fields take comma-separated values
    python config.py --immune-proximity-thresholds-um 10,25,50,100 \\
                      --thumbnail-size 512,512 \\
                      --morphology-tumor-class-names Tumour,Tumor --print-config

You can also import `config_from_args()` / `build_arg_parser()` /
`config_from_json()` directly in another script if you want CLI parsing
without going through `main()`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields, replace
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import Any, Optional


def _default_checkpoint() -> str:
    """Path to the model checkpoint bundled inside the installed package."""
    return str(_pkg_files("vispace").joinpath("assets", "weights", "TNBC_best.pt"))


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
    OUT_DIR:    str = "vispace_output"          # root directory for all outputs
    CHECKPOINT: str = field(default_factory=_default_checkpoint)  # bundled model checkpoint; override for a custom .pt
    WSI_PATH:  str = "your data path"              #  .svs / .tif slides


    # ── Mussel (tessellation) ───────────────────────────────────────────────
    MPP:            float = 0.25                # microns-per-pixel  (40x TCGA) not to changed 
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
    STEP_X:         int   = None                 # WSI tile spacing x (px) — informational
    STEP_Y:         int   = None                 # WSI tile spacing y (px) — informational
    MAX_PX:         int   = 4096                # max output PNG long edge

    # ── Stitch / visualisation ──────────────────────────────────────────────
    ALPHA:          float = 0.6                 # overlay blend weight
    THUMB_MAX:      int   = 2048                # thumbnail max size

    # ── Stage 4: Tumor ROI Overlay ──────────────────────────────────────────
    # ============================================================
    # TUMOR ROI / TUMOR FOCUS PARAMETERS
    # ============================================================
    

    # --- Tumour focus detection + representative tumour ROI sampling ------------
    ROI_SIZE_UM: float = 200.0
    ROI_MIN_TUMOR_FRAC: float = 0.20
    ROI_FOCUS_REPAIR_GAP_UM: float = 25.0
    ROI_MIN_FOCUS_AREA_UM2: float = 40000.0  # good starting point for suppressing tiny foci
    ROI_MIN_ROI_TUMOR_FRAC: float = 0.30
    ROI_MAX_NECROSIS: float = 0.50
    
    # Candidate-centre ranking score:
    #   tumor_weight * frac_Tumour
    # + neighbor_weight * local_neighbor_tumor_mean
    # - necrosis_penalty * frac_Necrosis
    ROI_TUMOR_SIGNAL_WEIGHT: float = 1.0
    ROI_NEIGHBOR_TUMOR_WEIGHT: float = 0.5
    ROI_NECROSIS_PENALTY: float = 0.5
    
    # Representative sampling rather than full-clump tiling.
    # 0 restores unlimited sampling; a finite value is recommended.
    ROI_MAX_ROIS_PER_FOCUS: int = 3
    
    # --- Inter-tumour graph / tissue-aware path ----------------------------------
    INTER_TUMOR_MIN_GAP_UM: float = 50.0
    INTER_TUMOR_MAX_GAP_UM: float = 1000.0
    INTER_TUMOR_KNN_K: int = 3
    INTER_TUMOR_CORRIDOR_WIDTH_UM: float = 100.0
    INTER_TUMOR_SEARCH_MARGIN_UM: float = 500.0
    
    # A* tissue costs are already config-driven in cluster_tils_tsr_scoring.py.
    INTER_PATH_TUMOR_WEIGHT: float = 5.0
    INTER_PATH_NECROSIS_WEIGHT: float = 6.0
    INTER_PATH_OTHER_WEIGHT: float = 1.0
    
    # --- Inter-tumour ROI + Master ROI -------------------------------------------
    INTER_TUMOR_ROI_SIZE_UM: float = 200.0
    INTER_TUMOR_SAMPLE_FRACTIONS = (0.5,)
    INTER_TUMOR_MAX_TUMOR_FRAC: float = 0.20
    INTER_TUMOR_MAX_NECROSIS: float = 0.50
    MASTER_STROMA_MARGIN_UM: float = 50.0
    
    # --- Scoring QC ---------------------------------------------------------------
    CLUSTER_MIN_TISSUE_FRACTION: float = 0.10
    CLUSTER_MIN_TSR_DENOM_PX2: float = 5000.0
    CLUSTER_MIN_TILS_DENOM_PX2: float = 5000.0
    CLUSTER_MIN_POLYGON_AREA_PX2: float = 1.0

        # ── Stage 5: Cluster TSR / sTILs scoring ───────────────────────────────
    # ============================================================
    # CLUSTER / TIL / TSR / INTER-TUMOR PARAMETERS
    # ============================================================
    SPATIAL_SCORING_MODE = "fast"
    
    # Dense-lattice memory safety guard.
    SPATIAL_GRID_MAX_CELLS = 25_000_000
    
    # ============================================================
    # FOCUS RELATIONSHIPS
    # ============================================================
    
    # gap < MIN:
    #   direct Master-ROI merge; NO inter-tumour corridor/TIL is created.
    # MIN <= gap <= MAX:
    #   eligible for Dijkstra routing + corridor QC.
    INTER_TUMOR_MIN_GAP_UM = 50.0
    INTER_TUMOR_MAX_GAP_UM = 1000.0
    INTER_TUMOR_KNN_K = 3
    
    # ============================================================
    # TISSUE-AWARE DIJKSTRA ROUTING
    # ============================================================
    INTER_TUMOR_CORRIDOR_WIDTH_UM = 100.0
    INTER_TUMOR_SEARCH_MARGIN_UM = 300.0
    
    INTER_PATH_TUMOR_WEIGHT = 5.0
    INTER_PATH_NECROSIS_WEIGHT = 6.0
    INTER_PATH_OTHER_WEIGHT = 1.0
    
    # ============================================================
    # CORRIDOR QC
    # ============================================================
    INTER_CORRIDOR_MAX_TUMOR_FRAC = 0.20
    INTER_CORRIDOR_MAX_NECROSIS_FRAC = 0.30
    INTER_CORRIDOR_MIN_STROMAL_LIKE_FRAC = 0.30
    INTER_CORRIDOR_MIN_TISSUE_FRACTION = 0.50
    INTER_CORRIDOR_MIN_PATH_EFFICIENCY = 0.50
    
    # ============================================================
    # REPRESENTATIVE INTER-TUMOUR ROI SAMPLING
    # ============================================================
    INTER_TUMOR_ROI_SIZE_UM = 200.0
    INTER_TUMOR_SAMPLE_FRACTIONS = (0.5,)
    INTER_TUMOR_MAX_TUMOR_FRAC = 0.20
    INTER_TUMOR_MAX_NECROSIS = 0.50
    
    # ============================================================
    # MASTER ROI / SCORING QC
    # ============================================================
    MASTER_STROMA_MARGIN_UM = 100.0
    CLUSTER_MIN_TISSUE_FRACTION = 0.10
    CLUSTER_MIN_TSR_DENOM_PX2 = 5000.0
    CLUSTER_MIN_TILS_DENOM_PX2 = 5000.0
    
    # Exact-vector refinement only.
    CLUSTER_MIN_POLYGON_AREA_PX2 = 1.0
    
    # % inflammatory area within each distance from the tumor boundary
    IMMUNE_PROXIMITY_THRESHOLDS_UM: list = field(
        default_factory=lambda: [50, 100, 200]
    )
    
    # Inflammatory tissue within 5 µm of tumor boundary = contact
    IMMUNE_CONTACT_TOLERANCE_UM: float = 5.0
    
    # Immune phenotype thresholds
    IMMUNE_DESERT_MAX_AREA_UM2: float = 1_000.0
    
    # ≥30% of inflammatory area intratumoral = immune-penetrated
    IMMUNE_PENETRATED_MIN_INTRA_FRAC: float = 0.30
    
    # ≥50% of inflammatory area within 50 µm = margin-localized
    IMMUNE_MARGIN_MIN_PCT_50UM: float = 50.0
    
    # Median extratumoral distance ≥100 µm = immune-excluded
    IMMUNE_EXCLUDED_MIN_MEDIAN_UM: float = 100.0
    
    # Median extratumoral distance <100 µm = peritumoral
    IMMUNE_PERITUMORAL_MAX_MEDIAN_UM: float = 100.0
    # ── Stage 7: Necrosis proximity features ─────────────────────────────────
    NECROSIS_MIN_COMPONENT_AREA_UM2: float = 500.0
    # Noise filter: necrosis polygon components smaller than this (µm²) are
    # excluded from area and perimeter computation. Removes segmentation
    # artefacts at tile boundaries without affecting genuine necrotic foci.
    
    NECROSIS_FOCAL_THRESHOLD: float = 0.05
    # Fraction threshold separating focal from present phenotype.
    # Clusters with necrosis_frac < NECROSIS_FOCAL_THRESHOLD are classified
    # as focal (punctate foci); those >= threshold are classified as present
    # (geographic / confluent necrosis). Default 0.05 (5%) reflects standard
    # histopathological practice for distinguishing minor from significant
    # necrosis in invasive breast carcinoma (WHO 2019).

    # ── Stage 8: Tumour morphology features ────────────────────────────────

    # Stage 8: Tumour morphology features

    MORPHOLOGY_MIN_ISLAND_AREA_UM2: float = 1000.0
    
    # Vector box-counting tumour-boundary fractal dimension.
    # Scales are relative to each tumour geometry's own maximum dimension:
    #   finest box  = max_dimension / 50
    #   coarsest box = max_dimension / 4
    MORPHOLOGY_FD_NUM_SCALES: int = 10
    MORPHOLOGY_FD_MIN_VALID_SCALES: int = 5
    MORPHOLOGY_FD_FINE_DIVISOR: float = 50.0
    MORPHOLOGY_FD_COARSE_DIVISOR: float = 4.0
    
    MORPHOLOGY_SAVE_ISLAND_QC: bool = False
    
    MORPHOLOGY_TUMOR_CLASS_NAMES: set = field(
        default_factory=lambda: {"Tumour", "Tumor", "tumour", "tumor"}
    )
    

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


# ═════════════════════════════════════════════════════════════════════════
# CLI support
# ═════════════════════════════════════════════════════════════════════════
# The functions below auto-derive a command-line interface from the
# dataclass fields above, so the CLI can never drift out of sync with
# PipelineConfig — add/remove/rename a field and the flag follows.

def _flag_name(field_name: str) -> str:
    """PATCH_SIZE -> --patch-size"""
    return "--" + field_name.lower().replace("_", "-")


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.strip().lower()
    if v in ("true", "t", "1", "yes", "y", "on"):
        return True
    if v in ("false", "f", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def _parse_list_of_number(value: str) -> list:
    """'20,50,100,200' -> [20.0, 50.0, 100.0, 200.0]"""
    return [float(x) for x in value.split(",") if x.strip() != ""]


def _parse_tuple_of_int(value: str) -> tuple:
    """'1024,1024' -> (1024, 1024)"""
    return tuple(int(x) for x in value.split(",") if x.strip() != "")


def _parse_set_of_str(value: str) -> set:
    """'Tumour,Tumor' -> {'Tumour', 'Tumor'}"""
    return {x.strip() for x in value.split(",") if x.strip() != ""}


def add_config_fields_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add one `--flag` per PipelineConfig field to an existing parser.

    Factored out of build_arg_parser() so other pipeline-stage scripts
    (tessellate.py, segment.py, ...) can build their own ArgumentParser with
    a stage-specific description/epilog and still get every config field as
    a flag for free, without also inheriting config.py's own --run /
    --print-config flags (which are specific to the full-pipeline entry
    point in this file's main()).

        parser = argparse.ArgumentParser(prog="tessellate.py", ...)
        add_config_fields_to_parser(parser)
        args = parser.parse_args()
    """
    defaults = PipelineConfig()
    for f in fields(PipelineConfig):
        name = f.name
        flag = _flag_name(name)
        current = getattr(defaults, name)
        help_text = f.metadata.get("help", "") if f.metadata else ""

        # dest is pinned to the exact dataclass field name (argparse would
        # otherwise lowercase "--out-dir" to "out_dir", not "OUT_DIR").
        if isinstance(current, bool):
            parser.add_argument(
                flag, dest=name, type=_parse_bool, default=current, metavar="{true,false}",
                help=f"[bool] {help_text}",
            )
        elif isinstance(current, tuple):
            parser.add_argument(
                flag, dest=name, type=_parse_tuple_of_int, default=current, metavar="INT,INT,...",
                help=f"[tuple[int], comma-separated] {help_text}",
            )
        elif isinstance(current, list):
            parser.add_argument(
                flag, dest=name, type=_parse_list_of_number, default=current, metavar="NUM,NUM,...",
                help=f"[list[float], comma-separated] {help_text}",
            )
        elif isinstance(current, set):
            parser.add_argument(
                flag, dest=name, type=_parse_set_of_str, default=current, metavar="STR,STR,...",
                help=f"[set[str], comma-separated] {help_text}",
            )
        elif isinstance(current, int):  # after bool check, since bool is an int subclass
            parser.add_argument(flag, dest=name, type=int, default=current, help=f"[int] {help_text}")
        elif isinstance(current, float):
            parser.add_argument(flag, dest=name, type=float, default=current, help=f"[float] {help_text}")
        elif current is None or isinstance(current, str):
            parser.add_argument(
                flag, dest=name, type=str, default=current,
                help=f"[str{'|None' if current is None else ''}] {help_text}",
            )
        else:
            # Fallback for any future field type: pass through as a string.
            parser.add_argument(
                flag, dest=name, type=str, default=current,
                help=f"[{type(current).__name__}] {help_text}",
            )
    return parser


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Auto-generate the full-pipeline CLI parser: every PipelineConfig field
    as a flag, plus config.py's own --print-config / --run / --from-json
    flags.
    """
    parser = argparse.ArgumentParser(
        prog="vispace",
        description="Vispace pipeline — build a PipelineConfig (and optionally run the "
                     "pipeline) from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_fields_to_parser(parser)
    parser.add_argument(
        "--print-config", action="store_true",
        help="Print the resolved configuration as JSON and exit (no pipeline run).",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="After building the config, import run_vispace and execute "
             "run_vispace(WSI_PATH, cfg).",
    )
    parser.add_argument(
        "--from-json", type=str, default=None,
        help="Load a PipelineConfig previously saved via --print-config "
             "(e.g. `python config.py --print-config > run_config.json`). "
             "Any other --flag passed alongside this one overrides the "
             "corresponding field from the JSON file. Combine with --run "
             "to execute immediately, e.g. "
             "`python config.py --from-json run_config.json --run`.",
    )
    return parser


def config_from_json(path: str) -> PipelineConfig:
    """
    Load a PipelineConfig from a JSON file previously written by
    `python config.py --print-config > run_config.json`.
    """
    with open(path, "r") as fh:
        data = json.load(fh)

    valid = {f.name for f in fields(PipelineConfig)}
    unknown = set(data) - valid
    if unknown:
        raise SystemExit(f"Unknown field(s) in {path}: {sorted(unknown)}")

    coerced = {}
    defaults = PipelineConfig()
    for f in fields(PipelineConfig):
        if f.name not in data:
            continue
        default = getattr(defaults, f.name)
        val = data[f.name]
        # list/tuple/set-typed fields round-trip through JSON as plain
        # lists; coerce them back to the annotated type.
        if isinstance(default, tuple) and isinstance(val, list):
            val = tuple(val)
        elif isinstance(default, set) and isinstance(val, list):
            val = set(val)
        coerced[f.name] = val

    return replace(PipelineConfig(), **coerced)


def config_from_args(argv: Optional[list] = None) -> tuple[PipelineConfig, argparse.Namespace]:
    """
    Parse CLI args and return (PipelineConfig, argparse.Namespace).

    Any flag the user didn't pass falls back to the PipelineConfig default,
    so this is safe to call with a partial command line.

    If --from-json is given, that file becomes the base config, and only
    the flags the user *explicitly* passed on top of it are applied as
    overrides (flags left at their CLI default do not clobber the JSON
    values).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.from_json:
        base_cfg = config_from_json(args.from_json)
        defaults = PipelineConfig()
        overrides = {
            f.name: getattr(args, f.name)
            for f in fields(PipelineConfig)
            if getattr(args, f.name) != getattr(defaults, f.name)
        }
        new_cfg = replace(base_cfg, **overrides)
    else:
        overrides = {f.name: getattr(args, f.name) for f in fields(PipelineConfig)}
        new_cfg = replace(PipelineConfig(), **overrides)

    return new_cfg, args


def _json_default(o: Any):
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def main(argv: Optional[list] = None) -> None:
    new_cfg, args = config_from_args(argv)

    if args.print_config:
        print(json.dumps(new_cfg.__dict__, indent=2, default=_json_default))
        return

    if args.run:
        try:
            from .run_vispace import run_vispace
        except ImportError as e:
            raise SystemExit(
                "Could not import run_vispace — make sure run_vispace.py is on "
                f"the Python path (same directory or PYTHONPATH). Original error: {e}"
            )
        results = run_vispace(new_cfg.WSI_PATH, new_cfg)
        print(results)
        return

    # Default: no action flag given — just confirm the config built cleanly.
    print("Config built successfully. Pass --print-config to view it in full, "
          "or --run to execute the pipeline.")
    print(f"  WSI_PATH   = {new_cfg.WSI_PATH}")
    print(f"  CHECKPOINT = {new_cfg.CHECKPOINT}")
    print(f"  OUT_DIR    = {new_cfg.OUT_DIR}")


if __name__ == "__main__":
    main()