#!/usr/bin/env python3
"""
run_vipsegd.py
==============
Full ViP-SegD pipeline orchestrator for a single WSI.

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

Usage
-----
    from run_vipsegd import run_vipsegd
    from config import cfg

    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg)

    # Re-run only failed / missing stages (completed stages are skipped):
    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg)

    # Force a specific stage to re-run even if output exists:
    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg,
                          force_stages={"segmentation", "stitching"})

    # Run only a subset of stages (their prerequisites must already exist):
    results = run_vipsegd("slides/TCGA-A1-A0SP.svs", cfg,
                          stages={"tumor_roi_overlay", "cluster_tils_tsr_score"})
"""

from __future__ import annotations

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
from cluster_tils_tsr_score      import run_cluster_tils_tsr_score
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

def run_vipsegd(
    wsi_path:     str,
    cfg:          PipelineConfig   = None,
    stages:       Optional[Set[str]] = None,
    force_stages: Optional[Set[str]] = None,
) -> dict:
    """
    Run the full ViP-SegD pipeline for one WSI.

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

    slide_name   = Path(wsi_path).stem
    sentinels    = _sentinel(wsi_path, cfg)
    force_stages = set(force_stages) if force_stages else set()
    run_set      = set(stages) if stages else set(ALL_STAGE_NAMES)

    # Validate stage names
    unknown = (run_set | force_stages) - set(ALL_STAGE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown stage name(s): {sorted(unknown)}\n"
            f"Valid names: {ALL_STAGE_NAMES}"
        )

    pipeline_start = time.perf_counter()

    print(f"\n{'#'*60}")
    print(f"  ViP-SegD pipeline")
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
            result = stage_fn(wsi_path, cfg)
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
    success       = not any_failed and all(
        stage_results.get(n, {}).get("status") in ("done", "skipped")
        for n in run_set
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
        "wsi_path":         str(wsi_path),
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
        The stage result dict (same format as run_vipsegd's per-stage entry).
    """
    return run_vipsegd(
        wsi_path,
        cfg          = cfg,
        stages       = {stage_name},
        force_stages = {stage_name} if force else None,
    )["stages"][stage_name]