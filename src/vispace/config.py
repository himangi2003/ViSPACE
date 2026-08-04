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
from pathlib import Path
from typing import Any, Optional


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
    CHECKPOINT: str = "TNBC_weights/TNBC_best.pt"  # path to Vispace model checkpoint
    WSI_PATH:  str = "your data path"              #  .svs / .tif slides


    # ── Mussel (tessellation) ───────────────────────────────────────────────
    MPP:            float = 0.25                # microns-per-pixel  (40x TCGA)
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
    CLUSTER_BUFFER_UM:           float = 300.0  # buffer around ROI boxes → scoring polygon
    CLUSTER_MIN_ROI_BOXES:       int   = 1     # min ROI boxes to keep a cluster
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
    NECROSIS_IMMUNE_COUPLING_THRESHOLD_UM: float = 20.0  # necrosis within this of immune = coupled
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