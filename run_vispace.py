#!/usr/bin/env python3
"""
run_vispace.py
==============
Full ViSPACE pipeline orchestrator for a single WSI.

Runs all eight stages end-to-end in dependency order and returns a
structured result dict.  Each stage is guarded: if its primary output
already exists on disk it is skipped, so partial runs can be resumed.

Pipeline stages
---------------
1.  run_tessellation          tessellate.py
2.  run_segmentation          segmenter.py
3.  run_stitching             stitch.py
4.  run_tumor_roi_overlay     tumor_roi_overlay.py
5.  run_cluster_tils_tsr_score cluster_tils_tsr_score.py
6.  run_immune_proximity_features    immune_proximity_features.py
7.  run_necrosis_proximity_features  necrosis_proximity_features.py
8.  run_tumor_morphology_features    tumor_morphology_features.py

Directory layout produced
-------------------------
cfg.OUT_DIR/<slide_name>/
    tessellation/
        patches/
        <slide>.h5
        mask.png  grid_mask.png  thumbnail.png

    segmentation/
        manifest.csv
        segmentation_all_classes.geojson
        <slide>_segmentation.png

    spatial_feature_results/
        tumor_roi_overlay/
            tumor_roi_boxes.csv
            tumor_roi_boxes_pseudo_thumbnail.png
            tumor_roi_boxes_wsi_thumbnail.png   (when WSI is on disk)

        cluster_tils_tsr_score/
            cluster_scoring_polygons.geojson
            tils_tsr_by_cluster.csv
            tils_tsr_wsi_summary.csv
            cluster_tils_tsr_overlay.png

        immune_proximity/
            immune_proximity_by_cluster.csv
            immune_proximity_wsi_summary.csv
            immune_proximity_plot.png

        necrosis_proximity/
            necrosis_proximity_by_cluster.csv
            necrosis_proximity_wsi_summary.csv
            necrosis_distance_figure.png
            necrosis_tissue_context_figure.png

        tumor_morphology/
            tumor_core_features_by_cluster.csv
            tumor_core_wsi_summary.csv
            tumor_island_qc.csv   (when cfg.MORPHOLOGY_SAVE_ISLAND_QC = True)

Usage (as a library)
---------------------
    from run_vispace import run_vispace
    from config import cfg

    results = run_vispace("slides/TCGA-A1-A0SP.svs", cfg)

    # Re-run only failed / missing stages (completed stages are skipped):
    results = run_vispace("slides/TCGA-A1-A0SP.svs", cfg)

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

Usage (from the command line)
------------------------------
Same shared flags as the rest of the pipeline — every PipelineConfig field
is available here too. Also supports --from-json to pick up a config saved
earlier via `config.py --print-config`, plus --stages / --force-stages to
mirror the stages= / force_stages= parameters above.

    # run the full pipeline
    python run_vispace.py --from-json run_config.json

    # run only a subset of stages
    python run_vispace.py --from-json run_config.json \\
        --stages cluster_tils_tsr_score,immune_proximity,necrosis_proximity

    # force specific stages to re-run even if output exists
    python run_vispace.py --from-json run_config.json \\
        --force-stages tumor_roi_overlay

    # see every available flag
    python run_vispace.py --help
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Set

from config import cfg as default_cfg, PipelineConfig

# ── Stage imports ────────────────────────────────────────────────────────────
from tessellate                  import run_tessellation
from segmenter                   import run_segmentation
from stitch                      import run_stitching
from tumor_roi_overlay           import run_tumor_roi_overlay
from cluster_tils_tsr_scoring    import run_cluster_tils_tsr_score
from immune_proximity_features   import run_immune_proximity_features
from necrosis_proximity_features import run_necrosis_proximity_features
from tumor_morphology_features   import run_tumor_morphology_features


# ---------------------------------------------------------------------------
# Sentinel-file helpers
# Stages are skipped when their primary output already exists on disk.
# Each entry: stage_name → callable that returns the Path to check.
# ---------------------------------------------------------------------------

def _sentinel(wsi_path: str, cfg: PipelineConfig) -> dict:
    """Return {stage_name: Path} for the primary output of every stage."""
    slide_name = Path(wsi_path).stem
    root       = Path(cfg.OUT_DIR) / slide_name
    sf         = root / "spatial_feature_results"
    return {
        "tessellation":   root / "tessellation" / f"{slide_name}.h5",
        "segmentation":   root / "segmentation" / "manifest.csv",
        "stitching":      root / "segmentation" / "segmentation_all_classes.geojson",
        "tumor_roi_overlay":         sf / "tumor_roi_overlay"    / "tumor_roi_boxes.csv",
        "cluster_tils_tsr_score":    sf / "cluster_tils_tsr_score" / "tils_tsr_by_cluster.csv",
        "immune_proximity":          sf / "immune_proximity"     / "immune_proximity_by_cluster.csv",
        "necrosis_proximity":        sf / "necrosis_proximity"   / "necrosis_proximity_by_cluster.csv",
        "tumor_morphology":          sf / "tumor_morphology"     / "tumor_core_features_by_cluster.csv",
    }


# ---------------------------------------------------------------------------
# Ordered stage registry
# ---------------------------------------------------------------------------

_STAGES = [
    ("tessellation",           run_tessellation),
    ("segmentation",           run_segmentation),
    ("stitching",              run_stitching),
    ("tumor_roi_overlay",      run_tumor_roi_overlay),
    ("cluster_tils_tsr_score", run_cluster_tils_tsr_score),
    ("immune_proximity",       run_immune_proximity_features),
    ("necrosis_proximity",     run_necrosis_proximity_features),
    ("tumor_morphology",       run_tumor_morphology_features),
]

ALL_STAGE_NAMES = [name for name, _ in _STAGES]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_vispace(
    wsi_path:     str,
    cfg:          PipelineConfig   = None,
    stages:       Optional[Set[str]] = None,
    force_stages: Optional[Set[str]] = None,
) -> dict:
    """
    Run the full ViSPACE pipeline for one WSI.

    Parameters
    ----------
    wsi_path : str
        Path to the .svs / .tif WSI file.  Used to derive slide_name and
        (when the file exists on disk) for WSI-thumbnail overlays.

    cfg : PipelineConfig
        Pipeline configuration.  All stage-specific knobs are read from
        cfg with safe getattr fallbacks — see each stage module for details.
        Defaults to the config.cfg singleton.

    stages : set of str, optional
        If provided, only run these named stages (in pipeline order).
        Any stages whose prerequisites are not already on disk will raise
        a FileNotFoundError.  Useful for running a subset after a partial
        pipeline has already been executed.
        Valid names: tessellation, segmentation, stitching,
                     tumor_roi_overlay, cluster_tils_tsr_score,
                     immune_proximity, necrosis_proximity, tumor_morphology

    force_stages : set of str, optional
        Stage names that must re-run even if their output already exists.
        All other stages still use the normal skip-if-done logic.

    Returns
    -------
    dict
        {
          "slide_name":   str,
          "wsi_path":     str,
          "stages":       {stage_name: {"status": "done"|"skipped"|"failed",
                                        "elapsed_s": float,
                                        "result": dict | None,
                                        "error": str | None}},
          "success":      bool,   # True iff every requested stage succeeded
          "total_elapsed_s": float,
        }
    """
    if cfg is None:
        cfg = default_cfg

    if not wsi_path:
        raise ValueError("wsi_path cannot be empty.")

    wsi = Path(wsi_path).expanduser()
    slide_name = wsi.stem

    force_stages = set(force_stages or ())
    run_set = set(stages) if stages is not None else set(ALL_STAGE_NAMES)

    if not run_set:
        raise ValueError("At least one pipeline stage must be requested.")

    # Validate stage names.
    valid_stage_names = set(ALL_STAGE_NAMES)
    unknown = (run_set | force_stages) - valid_stage_names
    if unknown:
        raise ValueError(
            f"Unknown stage name(s): {sorted(unknown)}\n"
            f"Valid names: {ALL_STAGE_NAMES}"
        )

    # A forced stage must also be in the requested run set. Adding it here
    # prevents a silent no-op when force_stages contains a stage omitted from
    # stages.
    run_set |= force_stages

    # The WSI must exist for tessellation. Later-stage-only runs may use the
    # path merely to derive the slide name, provided their prerequisite files
    # already exist.
    if "tessellation" in run_set and not wsi.is_file():
        raise FileNotFoundError(f"Whole-slide image not found: {wsi}")

    if "segmentation" in run_set:
        checkpoint = Path(cfg.CHECKPOINT).expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"ViSPACE checkpoint not found: {checkpoint}"
            )

    sentinels = _sentinel(str(wsi), cfg)

    pipeline_start = time.perf_counter()

    print(f"\n{'#'*60}")
    print(f"  ViSPACE pipeline")
    print(f"  Slide : {slide_name}")
    print(f"  Stages: {', '.join(n for n, _ in _STAGES if n in run_set)}")
    print(f"{'#'*60}")

    stage_results: dict = {}
    any_failed = False

    for stage_name, stage_fn in _STAGES:
        if stage_name not in run_set:
            continue  # not in requested subset

        sentinel = sentinels[stage_name]
        already_done = sentinel.exists() and stage_name not in force_stages

        if already_done:
            print(f"\n  ↷  [{stage_name}] output exists — skipping")
            print(f"      ({sentinel})")
            stage_results[stage_name] = {
                "status":    "skipped",
                "elapsed_s": 0.0,
                "result":    None,
                "error":     None,
            }
            continue

        t0 = time.perf_counter()
        print(f"\n  ▶  [{stage_name}] starting …")

        try:
            result = stage_fn(str(wsi), cfg)
            elapsed = time.perf_counter() - t0
            print(f"  ✓  [{stage_name}] done in {elapsed:.1f}s")
            stage_results[stage_name] = {
                "status":    "done",
                "elapsed_s": elapsed,
                "result":    result,
                "error":     None,
            }
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            any_failed = True
            tb = traceback.format_exc()
            print(f"  ✗  [{stage_name}] FAILED after {elapsed:.1f}s")
            print(f"     {exc}")
            stage_results[stage_name] = {
                "status":    "failed",
                "elapsed_s": elapsed,
                "result":    None,
                "error":     tb,
            }
            # Abort — downstream stages depend on this one
            print(f"\n  Pipeline halted: [{stage_name}] failed. "
                  f"Fix the error and re-run; completed stages will be skipped.")
            break

    total_elapsed = time.perf_counter() - pipeline_start
    success = (
        not any_failed
        and all(
            stage_results.get(name, {}).get("status") in {"done", "skipped"}
            for name in run_set
        )
    )

    print(f"\n{'#'*60}")
    if success:
        print(f"  ✓  Pipeline complete for {slide_name}  ({total_elapsed:.1f}s total)")
    else:
        print(f"  ✗  Pipeline finished with errors for {slide_name}  ({total_elapsed:.1f}s)")
    print(f"{'#'*60}\n")

    # Print per-stage timing table
    _print_stage_summary(stage_results, run_set)

    return {
        "slide_name":       slide_name,
        "wsi_path":         str(wsi),
        "stages":           stage_results,
        "success":          success,
        "total_elapsed_s":  total_elapsed,
    }


def _print_stage_summary(stage_results: dict, run_set: set) -> None:
    """Print a compact timing table after the pipeline finishes."""
    print(f"  {'Stage':<32} {'Status':<10} {'Time':>8}")
    print(f"  {'-'*32} {'-'*10} {'-'*8}")
    for stage_name in ALL_STAGE_NAMES:
        if stage_name not in run_set:
            continue
        info   = stage_results.get(stage_name, {})
        status = info.get("status", "not run")
        t      = info.get("elapsed_s", 0.0)
        symbol = {"done": "✓", "skipped": "↷", "failed": "✗"}.get(status, "?")
        t_str  = f"{t:.1f}s" if t > 0 else "—"
        print(f"  {symbol} {stage_name:<31} {status:<10} {t_str:>8}")
    print()


# ---------------------------------------------------------------------------
# Convenience: run individual stages by name
# ---------------------------------------------------------------------------

def run_stage(
    stage_name: str,
    wsi_path:   str,
    cfg:        PipelineConfig = None,
    force:      bool           = False,
) -> dict:
    """
    Run a single named pipeline stage.

    Parameters
    ----------
    stage_name : str
        One of the valid stage names (see ALL_STAGE_NAMES).
    wsi_path   : str
    cfg        : PipelineConfig
    force      : bool
        If True, re-run even if the output already exists on disk.

    Returns
    -------
    dict
        The stage result dict (same format as run_vispace's per-stage entry).
    """
    return run_vispace(
        wsi_path,
        cfg          = cfg,
        stages       = {stage_name},
        force_stages = {stage_name} if force else None,
    )["stages"][stage_name]


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's field flags + --from-json, plus run_vispace.py-specific
# --stages / --force-stages flags mirroring the stages= / force_stages=
# parameters of run_vispace() above.

def _explicit_destinations(parser, argv: list[str]) -> set[str]:
    """
    Return argparse destination names explicitly supplied by the user.

    This allows values from --from-json to be overridden even when the
    explicit CLI value happens to equal PipelineConfig's default.
    """
    supplied = set()
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }

    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            supplied.add(dest)

    return supplied


def _parse_stage_list(value: Optional[str]) -> Optional[Set[str]]:
    """Parse a comma-separated stage list, removing blanks."""
    if value is None:
        return None

    parsed = {item.strip() for item in value.split(",") if item.strip()}
    return parsed or None


def main(argv=None) -> None:
    import argparse
    from dataclasses import fields, replace

    from config import add_config_fields_to_parser, config_from_json

    cli_argv = list(sys.argv[1:] if argv is None else argv)

    parser = argparse.ArgumentParser(
        prog="run_vispace.py",
        description=(
            "Run the full ViSPACE pipeline, or a selected subset of stages, "
            "for one whole-slide image."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_fields_to_parser(parser)

    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help=(
            "Load a PipelineConfig saved with `config.py --print-config`. "
            "Explicit CLI configuration flags override values from the JSON."
        ),
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated stage names to run. The default is all stages. "
            f"Valid names: {', '.join(ALL_STAGE_NAMES)}"
        ),
    )
    parser.add_argument(
        "--force-stages",
        type=str,
        default=None,
        metavar="NAME,NAME,...",
        help=(
            "Comma-separated stage names to rerun even when their sentinel "
            "outputs already exist."
        ),
    )

    args = parser.parse_args(cli_argv)
    explicit = _explicit_destinations(parser, cli_argv)

    if args.from_json:
        cfg = config_from_json(args.from_json)
        overrides = {
            field_info.name: getattr(args, field_info.name)
            for field_info in fields(PipelineConfig)
            if field_info.name in explicit
        }
        if overrides:
            cfg = replace(cfg, **overrides)
    else:
        cfg_values = {
            field_info.name: getattr(args, field_info.name)
            for field_info in fields(PipelineConfig)
        }
        cfg = replace(PipelineConfig(), **cfg_values)

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        parser.error(
            "--wsi-path is required, either directly or through --from-json."
        )

    stages = _parse_stage_list(args.stages)
    force_stages = _parse_stage_list(args.force_stages)

    valid = set(ALL_STAGE_NAMES)
    if stages:
        unknown = stages - valid
        if unknown:
            parser.error(
                f"Unknown --stages value(s): {sorted(unknown)}. "
                f"Valid names: {ALL_STAGE_NAMES}"
            )

    if force_stages:
        unknown = force_stages - valid
        if unknown:
            parser.error(
                f"Unknown --force-stages value(s): {sorted(unknown)}. "
                f"Valid names: {ALL_STAGE_NAMES}"
            )

    result = run_vispace(
        cfg.WSI_PATH,
        cfg,
        stages=stages,
        force_stages=force_stages,
    )

    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()