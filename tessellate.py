"""
tessellate.py
=============
Step 1 of the ViP-SegD pipeline.

Runs Mussel tiling on one WSI using settings from config.py.

Output directory
----------------
    cfg.OUT_DIR/<slide_name>/tessellation/
        patches/
        mask.png
        grid_mask.png
        thumbnail.png
        <slide>.h5

Usage (as a library)
---------------------
    from tessellate import run_tessellation
    from config import cfg
    outdir = run_tessellation(
        wsi_path = "slides/TCGA-A1-A0SP.svs",
        cfg      = cfg,
    )

Usage (from the command line)
------------------------------
tessellate.py shares its CLI with config.py — every PipelineConfig field
(including MUSSEL_DIR, PATCH_SIZE, WORKERS, SEGMENT_THRESH, THUMBNAIL_SIZE,
OUT_DIR, WSI_PATH, ...) is available as a flag, so you don't need to learn a
second set of arguments for this step. It also supports --from-json, so you
can pick up a config saved earlier via `config.py --print-config`.

    # minimal
    python tessellate.py --wsi-path slides/TCGA-A1-A0SP.svs

    # pointing at a non-default Mussel checkout, custom patch size / workers
    python tessellate.py \\
        --wsi-path slides/TCGA-A1-A0SP.svs \\
        --mussel-dir /opt/Mussel \\
        --patch-size 224 \\
        --workers 8 \\
        --segment-thresh 20 \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python tessellate.py --from-json run_config.json

    # continue from a saved config but override one field
    python tessellate.py --from-json run_config.json --patch-size 256

    # see every available flag
    python tessellate.py --help
"""
import sys
from pathlib import Path

from omegaconf import OmegaConf

from config import cfg as default_cfg, PipelineConfig


def run_tessellation(
    wsi_path: str,
    cfg: PipelineConfig = None,
) -> str:
    """
    Run Mussel tiling on a single WSI.

    Parameters
    ----------
    wsi_path : path to .svs / .tif
    cfg      : PipelineConfig (defaults to config.cfg singleton)

    Returns
    -------
    str : path to the slide tessellation directory
          cfg.OUT_DIR/<slide_name>/tessellation/
          contains: patches/, mask.png, grid_mask.png, thumbnail.png, <slide>.h5
    """
    if cfg is None:
        cfg = default_cfg

    # FIX 1: Validate that MUSSEL_DIR exists before inserting into sys.path.
    # Previously, sys.path.insert() silently succeeded even if the directory
    # was missing, leading to a cryptic ImportError later.
    mussel_dir = Path(cfg.MUSSEL_DIR)
    if not mussel_dir.exists():
        raise FileNotFoundError(
            f"Mussel not found at '{cfg.MUSSEL_DIR}'. "
            f"Clone the Mussel repository and set MUSSEL_DIR in config.py (or pass "
            f"--mussel-dir on the command line) to its root path.\n"
            f"  git clone https://github.com/pathology-data-mining/Mussel {cfg.MUSSEL_DIR}"
        )

    # Add Mussel to path at call time — not at import time
    # so the module works even if Mussel is not installed globally
    if str(mussel_dir) not in sys.path:
        sys.path.insert(0, str(mussel_dir))

    import mussel.cli.tessellate
    from mussel.cli.tessellate import TessellateConfig, SegConfig

    wsi        = Path(wsi_path)
    slide_name = wsi.stem
    outdir     = Path(cfg.OUT_DIR) / slide_name / "tessellation"
    outdir.mkdir(parents=True, exist_ok=True)

    output_h5_path = outdir / f"{slide_name}.h5"

    print(f"\n{'='*55}")
    print(f"  Tessellation")
    print(f"  Slide      : {wsi.name}")
    print(f"  Mussel dir : {cfg.MUSSEL_DIR}")
    print(f"  Patch size : {cfg.PATCH_SIZE}")
    print(f"  Workers    : {cfg.WORKERS}")
    print(f"  Output     : {outdir}")
    print(f"{'='*55}")

    seg_config = SegConfig(
        patch_size        = cfg.PATCH_SIZE,
        use_otsu          = True,
        segment_threshold = cfg.SEGMENT_THRESH,
    )

    tess_config = TessellateConfig(
        slide_path            = str(wsi),
        output_h5_path        = str(output_h5_path),
        output_png_dir        = str(outdir / "patches"),
        output_mask_path      = str(outdir / "mask.png"),
        output_grid_mask_path = str(outdir / "grid_mask.png"),
        output_thumbnail_path = str(outdir / "thumbnail.png"),
        thumbnail_size        = cfg.THUMBNAIL_SIZE,
        seg_config            = seg_config,
        num_workers           = cfg.WORKERS,
    )

    # FIX 2: Removed the misleading tqdm(total=1) wrapper that showed "1/1"
    # instantly and gave no real progress signal.
    # Mussel does not currently expose a per-tile progress callback, so we
    # print a clear start/end message instead. If Mussel adds a callback in
    # the future, wire it up here.
    print(f"\n  Running Mussel tessellation (this may take a while)...")
    mussel.cli.tessellate.main(OmegaConf.create(tess_config))

    if output_h5_path.exists():
        n_patches = len(list((outdir / "patches").glob("*.png")))
        print(f"\n  Done. {n_patches:,} patches saved → {outdir}")
        return str(outdir)
    else:
        raise RuntimeError(
            f"Tessellation failed for {wsi_path}\n"
            f"Expected H5 at: {output_h5_path}"
        )


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) instead of hand-rolling a
# second parser here. This means tessellate.py automatically gets every
# PipelineConfig field flag AND --from-json support for free, so it can
# pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python tessellate.py --from-json run_config.json

def main(argv=None) -> None:
    from config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    run_tessellation(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()