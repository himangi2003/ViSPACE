#!/usr/bin/env python3
"""
tests/run_test_vispace.py
Standalone smoke test (no pytest). Runs the full ViSPACE pipeline on every
slide in tests/data/*.tif and reports per-stage status + output checks.

    python tests/run_test_vispace.py                 # DEVICE from cfg default
    VISPACE_DEVICE=cpu python tests/run_test_vispace.py
Exit code 0 = all slides/stages passed, 1 = something failed.
"""
from __future__ import annotations
import os, sys, glob, tempfile
from pathlib import Path
from dataclasses import replace

from vispace import run_vispace, cfg as base_cfg

DATA_DIR = Path(__file__).parent / "data"

# Expected primary output of each stage, relative to OUT_DIR/<slide>/
STAGE_OUTPUTS = {
    "tessellation":           "tessellation/{slide}.h5",
    "segmentation":           "segmentation/manifest.csv",
    "stitching":              "segmentation/segmentation_all_classes.geojson",
    "tumor_roi_overlay":      "spatial_feature_results/tumor_roi_overlay/tumor_roi_boxes.csv",
    "cluster_tils_tsr_score": "spatial_feature_results/cluster_tils_tsr_score/tils_tsr_by_cluster.csv",
    "immune_proximity":       "spatial_feature_results/immune_proximity/immune_proximity_by_cluster.csv",
    "necrosis_features":      "spatial_feature_results/necrosis_feature/necrosis_feature_by_cluster.csv",
    "tumor_morphology":       "spatial_feature_results/tumor_morphology/tumor_core_features_by_cluster.csv",
}


def run_one(slide: Path, out_dir: Path) -> bool:
    print(f"\n=== {slide.name} ===")
    cfg = replace(base_cfg, OUT_DIR=str(out_dir),
                  DEVICE=os.getenv("VISPACE_DEVICE", base_cfg.DEVICE))
    try:
        res = run_vispace(str(slide), cfg=cfg)
    except Exception as e:                       # pipeline blew up
        print(f"  ✗ pipeline raised: {type(e).__name__}: {e}")
        return False

    ok = True
    root = out_dir / slide.stem
    for stage, info in res["stages"].items():
        status = info["status"]                  # done | skipped | failed
        rel = STAGE_OUTPUTS.get(stage, "")
        exists = (root / rel.format(slide=slide.stem)).exists() if rel else True
        good = status in ("done", "skipped") and exists
        ok &= good
        mark = "✓" if good else "✗"
        extra = "" if exists else "  [output file missing]"
        err = f"  {info['error']}" if info.get("error") else ""
        print(f"  {mark} {stage:<24} {status:<8} {info['elapsed_s']:.1f}s{extra}{err}")

    print(f"  -> {'PASS' if ok and res['success'] else 'FAIL'} "
          f"(total {res['total_elapsed_s']:.1f}s)")
    return ok and res["success"]


def main() -> int:
    slides = sorted(glob.glob(str(DATA_DIR / "*.tif"))) \
           + sorted(glob.glob(str(DATA_DIR / "*.tiff")))
    if not slides:
        print(f"No .tif slides in {DATA_DIR} — nothing to test.")
        return 0                                 # not a failure: just skip

    all_ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for s in slides:
            all_ok &= run_one(Path(s), Path(tmp))

    print(f"\n{'='*40}\n{'ALL PASSED' if all_ok else 'FAILURES DETECTED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())