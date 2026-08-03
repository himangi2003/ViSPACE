"""
stitch.py
=========
Step 3 of the ViSpace pipeline — runs after segmenter.py.

Stitches per-tile .npy segmentation masks into a single gap-free canvas,
then produces:

    cfg.OUT_DIR/<slide_name>/segmentation/
        manifest.csv                      — written by segmenter.py (input here)
        stitched_seg.npy                  — intermediate combined int32 class map
        stitched_seg_metadata.json        — canvas origin, shape, step
        <slide>_segmentation.png          — colour segmentation + legend strip
        segmentation_all_classes.geojson  — all classes combined (QuPath-ready)

After both final outputs exist, ALL .npy files in the segmentation folder
(including stitched_seg.npy and every per-tile *_seg.npy) are deleted
automatically.  The folder then contains only:

    manifest.csv
    <slide>_segmentation.png
    segmentation_all_classes.geojson

Pipeline position
-----------------
    tessellate.py  →  segmenter.py  →  stitch.py  →  (TME features)

Usage (as a library)
---------------------
    from stitch import run_stitching
    from config import cfg

    results = run_stitching(
        wsi_path = "slides/TCGA-A1.svs",
        cfg      = cfg,
    )

Usage (from the command line)
------------------------------
Same shared flags as config.py / tessellate.py / segmenter.py — every
PipelineConfig field is available here too (--out-dir, --wsi-path, ...),
plus this script also supports --from-json to pick up a config saved
earlier via `config.py --print-config`.

    # minimal — requires segmenter.py to have already run for this slide
    python stitch.py --wsi-path slides/TCGA-A1-A0SP.svs --out-dir vipsegd_output

    # continue from a config saved earlier
    python stitch.py --from-json run_config.json

    # continue from a saved config but override one field
    python stitch.py --from-json run_config.json --out-dir other_output

    # override the tile step used when stitching (rare — auto-inference from
    # manifest.csv is recommended and used by default) and tighten the
    # minimum polygon area kept in the output GeoJSON
    python stitch.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output \\
        --stitch-step-x 444 --stitch-step-y 444 \\
        --min-area-px 200

    # combine --from-json with stitch.py-specific flags
    python stitch.py --from-json run_config.json --min-area-px 200

    # see every available flag
    python stitch.py --help

Note: --stitch-step-x / --stitch-step-y are distinct from --step-x /
--step-y (which only set cfg.STEP_X / cfg.STEP_Y — these are informational
fields and are not currently read by any script in this pipeline, including
this one). Leave --stitch-step-x/-y unset unless you need to force a
specific tile spacing instead of auto-inferring it from manifest.csv.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import cfg as default_cfg, PipelineConfig, add_config_fields_to_parser, config_from_json
from segmenter import (
    CLASS_NAMES,
    COLORS_BGR,
    N_CLASSES,
    PATCH_SIZE,
    SEG_IGNORE,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

QUPATH_COLORS = {
    "Tumour":       [255,   0,   0],
    "Stroma":       [  0, 200,   0],
    "Inflammatory": [  0, 100, 255],
    "Necrosis":     [255, 165,   0],
    "Others":       [220,   0, 220],
}


# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _infer_step(coords: np.ndarray, fallback: int = PATCH_SIZE) -> int:
    """
    Infer the most common nonzero step between sorted unique coordinates.
    Avoids hardcoding tile spacing — works for any Mussel step size.
    """
    coords = np.asarray(coords, dtype=np.int64)
    if coords.size < 2:
        return fallback
    uniq = np.unique(coords)
    if uniq.size < 2:
        return fallback
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return fallback
    vals, counts = np.unique(diffs, return_counts=True)
    return int(vals[np.argmax(counts)])


def _seg_to_colour_bgr(seg: np.ndarray) -> np.ndarray:
    """(H, W) int32 → (H, W, 3) uint8 BGR. SEG_IGNORE → black."""
    bgr = np.zeros((*seg.shape, 3), dtype=np.uint8)
    for c, (b, g, r) in enumerate(COLORS_BGR):
        bgr[seg == c] = (b, g, r)
    return bgr


def _rgb_to_int(r, g, b) -> int:
    return (255 << 24) | (int(r) << 16) | (int(g) << 8) | int(b)


def _cleanup_npys(seg_dir: Path) -> int:
    """
    Remove ALL .npy files from seg_dir (tile *_seg.npy + stitched_seg.npy).
    Called automatically after both final outputs exist.
    Returns number of files removed.
    """
    npy_files = list(seg_dir.glob("*.npy"))
    removed   = 0
    for p in tqdm(npy_files, desc="Cleaning temporary files", unit="file"):
        try:
            p.unlink()
            removed += 1
        except Exception as e:
            print(f"  [cleanup] could not remove {p.name}: {e}")
    print(f"  Cleaned up : {removed:,} .npy files from {seg_dir.name}/")
    return removed


def _draw_legend_top_right(
    bgr:          np.ndarray,
    seg:          np.ndarray,
    title:        Optional[str] = None,
    present_only: bool          = True,
) -> np.ndarray:
    """
    Draw a class legend on a BLACK strip added OUTSIDE (above) the image.

    Legend swatches are anchored to the top-right of the strip; the title
    (if given) sits at the top-left. Font and swatches are large enough
    to read clearly at typical WSI thumbnail resolutions.

    Parameters
    ----------
    bgr          : (H, W, 3) uint8 BGR image — NOT modified in place.
                   A new TALLER canvas is returned with the image shifted
                   down below the legend strip.
    seg          : (H, W) int32 segmentation — used to filter present classes
    title        : optional title text drawn at the top-left of the strip
    present_only : if True only show classes that appear in seg

    Returns
    -------
    New (H + strip_height, W, 3) uint8 BGR array:
        rows [0 : strip_height]      → black legend strip (top)
        rows [strip_height : end]    → original image (below)
    """
    H, W = bgr.shape[:2]

    # ── Scale — large enough to read clearly ────────────────────────────────
    scale  = max(2.5, min(W, H) / 800.0)
    font   = cv2.FONT_HERSHEY_SIMPLEX
    fscale = 0.55 * scale
    thick  = max(1, int(round(1.4 * scale)))
    swatch = max(28, int(round(32 * scale)))
    pad    = max(14, int(round(18 * scale)))
    cgap   = max(22, int(round(30 * scale)))

    # ── Which classes to show ───────────────────────────────────────────────
    rows = [
        (c, name, tuple(int(v) for v in COLORS_BGR[c]))
        for c, name in enumerate(CLASS_NAMES)
        if not present_only or (seg == c).any()
    ]
    if not rows and not title:
        return bgr

    # ── Measure text widths ─────────────────────────────────────────────────
    entry_widths = []
    for _, name, _ in rows:
        (tw, _), _ = cv2.getTextSize(name, font, fscale, thick)
        entry_widths.append(swatch + pad // 2 + tw)
    entries_total_w = (sum(entry_widths)
                       + cgap * max(0, len(entry_widths) - 1))

    title_h = 0
    if title:
        (_, title_h_raw), _ = cv2.getTextSize(
            title, font, fscale * 1.1, thick)
        title_h = title_h_raw + pad // 2

    # ── Strip height ────────────────────────────────────────────────────────
    strip_h = title_h + swatch + 3 * pad

    # ── New canvas: black strip ON TOP, original image BELOW ────────────────
    canvas = np.zeros((H + strip_h, W, 3), dtype=np.uint8)
    canvas[strip_h:, :, :] = bgr

    # ── Title — top-left of the strip ───────────────────────────────────────
    if title:
        ty = pad + title_h
        cv2.putText(canvas, title,
                    (pad, ty),
                    font, fscale * 1.1, (255, 255, 255), thick, cv2.LINE_AA)

    # ── Swatches anchored to TOP-RIGHT of the strip ─────────────────────────
    cx = W - pad - entries_total_w
    cy = (strip_h - swatch) // 2   # vertically centred within strip
    for (_, name, color), ew in zip(rows, entry_widths):
        # Coloured swatch
        cv2.rectangle(canvas,
                      (cx,          cy),
                      (cx + swatch, cy + swatch),
                      color, thickness=-1)
        # White border around swatch
        cv2.rectangle(canvas,
                      (cx,          cy),
                      (cx + swatch, cy + swatch),
                      (255, 255, 255), thickness=1)
        # Class name in white
        (_, th), _ = cv2.getTextSize(name, font, fscale, thick)
        text_x = cx + swatch + pad // 2
        text_y = cy + (swatch + th) // 2
        cv2.putText(canvas, name,
                    (text_x, text_y),
                    font, fscale, (255, 255, 255), thick, cv2.LINE_AA)
        cx += ew + cgap

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — STITCH .npy masks into combined canvas
# ─────────────────────────────────────────────────────────────────────────────

def _stitch_segmentation(
    manifest_csv: str,
    seg_dir:      Path,
    slide_name:   str,
    step_x:       Optional[int] = None,
    step_y:       Optional[int] = None,
) -> dict:
    """
    Stitch per-tile .npy masks into a single gap-free canvas.

    step_x / step_y
    ---------------
    None (recommended) → auto-inferred from manifest (wx, wy) coordinates
    as the mode of successive differences. This removes the checkerboard
    artefact that occurs when the step does not match actual tile spacing.
    Pass integers only to override.

    Saves
    -----
    seg_dir/
        stitched_seg.npy               int32 (H, W) class map
        stitched_seg_metadata.json     origin, shape, step info
        <slide>_segmentation.png       colour PNG with black legend strip

    Returns
    -------
    meta dict — passed directly to _extract_geojson()
    """
    df = pd.read_csv(manifest_csv)
    required = {"wx", "wy", "npy_path"}
    missing  = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {missing}")
    if len(df) == 0:
        raise ValueError(f"Manifest is empty: {manifest_csv}")

    wx_vals = df["wx"].to_numpy(dtype=np.int64)
    wy_vals = df["wy"].to_numpy(dtype=np.int64)

    # Auto-infer step from manifest coordinates
    step_was_inferred = False
    if step_x is None:
        step_x = _infer_step(wx_vals, fallback=PATCH_SIZE)
        step_was_inferred = True
    if step_y is None:
        step_y = _infer_step(wy_vals, fallback=PATCH_SIZE)
        step_was_inferred = True

    first_seg      = np.load(df["npy_path"].iloc[0])
    tile_h, tile_w = first_seg.shape[:2]

    x_min = int(wx_vals.min()); y_min = int(wy_vals.min())
    x_max = int(wx_vals.max()); y_max = int(wy_vals.max())

    out_w = (x_max - x_min) + step_x
    out_h = (y_max - y_min) + step_y

    print(f"\n{'='*55}")
    print(f"  Stitching")
    print(f"  Slide      : {slide_name}")
    print(f"  Tiles      : {len(df):,}")
    print(f"  Canvas     : {out_w}x{out_h} px  origin=({x_min},{y_min})")
    print(f"  Step       : {step_x}x{step_y}"
          f"{'  (auto-inferred)' if step_was_inferred else ''}")
    print(f"{'='*55}")

    combined    = np.full((out_h, out_w), SEG_IGNORE, dtype=np.int32)
    missing_npy = 0

    for row in tqdm(df.itertuples(index=False),
                    total=len(df), desc="Stitching segmentation tiles", unit="tile"):
        npy_path = Path(row.npy_path)
        if not npy_path.exists():
            missing_npy += 1
            continue

        seg = np.load(str(npy_path)).astype(np.int32)
        seg[seg == -1] = SEG_IGNORE

        lx = int(row.wx) - x_min
        ly = int(row.wy) - y_min

        # Resize tile to step size — fills gaps when step > tile_size
        seg_resized = cv2.resize(
            seg.astype(np.uint8), (step_x, step_y),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)

        combined[ly:ly + step_y, lx:lx + step_x] = seg_resized

    if missing_npy:
        print(f"  WARN: {missing_npy:,} .npy files not found")

    # Save stitched integer canvas
    seg_npy_path = seg_dir / "stitched_seg.npy"
    np.save(str(seg_npy_path), combined)
    print(f"  Saved npy  : {seg_npy_path.name}")

    # Colour image + black legend strip above the image
    colour_bgr = _seg_to_colour_bgr(combined)
    colour_bgr = _draw_legend_top_right(
        colour_bgr, combined,
        title=slide_name,
        present_only=True,
    )
    png_path = seg_dir / f"{slide_name}_segmentation.png"
    cv2.imwrite(str(png_path), colour_bgr)
    print(f"  Saved png  : {png_path.name}")

    # Class distribution
    valid = combined[combined != SEG_IGNORE]
    total = max(len(valid), 1)
    print(f"\n  Class distribution:")
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((combined == c).sum())
        if cnt:
            bar = "█" * int(40 * cnt / total)
            print(f"    {name:<20} {cnt:,} px  ({100*cnt/total:.1f}%)  {bar}")

    meta = {
        "manifest_csv":        str(manifest_csv),
        "stitched_seg_npy":    str(seg_npy_path),
        "stitched_seg_png":    str(png_path),
        "slide_name":          slide_name,
        "origin":              [int(x_min), int(y_min)],
        "canvas_shape":        [int(out_h), int(out_w)],
        "original_tile_shape": [int(tile_h), int(tile_w)],
        "step":                [int(step_x), int(step_y)],
        "step_inferred":       bool(step_was_inferred),
        "missing_npy":         int(missing_npy),
    }
    meta_path = seg_dir / "stitched_seg_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — EXTRACT PER-CLASS GeoJSON from stitched canvas
# ─────────────────────────────────────────────────────────────────────────────

def _extract_geojson(
    meta:        dict,
    seg_dir:     Path,
    min_area_px: int = 100,
) -> Dict[str, str]:
    """
    Extract GeoJSON polygons from stitched_seg.npy.

    Coordinates are in WSI level-0 pixel space:
        wsi_x = origin_x + canvas_x
        wsi_y = origin_y + canvas_y

    Saves
    -----
    seg_dir/
        segmentation_all_classes.geojson   — all classes combined (QuPath-ready)

    Returns
    -------
    dict: {"combined": path}
    """
    npy_path = Path(meta["stitched_seg_npy"])
    combined = np.load(str(npy_path)).astype(np.int32)
    combined[combined == -1] = SEG_IGNORE

    x_min, y_min = int(meta["origin"][0]), int(meta["origin"][1])
    H, W         = combined.shape

    print(f"\n{'='*55}")
    print(f"  GeoJSON extraction")
    print(f"  Canvas   : {W}x{H} px  origin=({x_min},{y_min})")

    valid = combined[combined != SEG_IGNORE]
    total = max(len(valid), 1)
    for c, name in enumerate(CLASS_NAMES):
        cnt = int((combined == c).sum())
        if cnt:
            print(f"    {name:<20} {cnt:,} px  ({100*cnt/total:.1f}%)")

    kernel       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    all_features = []

    for c in tqdm(range(N_CLASSES), desc="Writing GeoJSON", unit="class"):
        name = CLASS_NAMES[c]
        col  = QUPATH_COLORS.get(name, [128, 128, 128])

        binary = (combined == c).astype(np.uint8)
        if not binary.any():
            continue

        # Close 1-px gaps between adjacent tiles of the same class
        binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        features = []
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            # Canvas → WSI level-0 pixel coordinates
            ring = [
                [float(x_min + pt[0][0]), float(y_min + pt[0][1])]
                for pt in cnt
            ]
            if len(ring) < 3:
                continue
            ring.append(ring[0])

            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "objectType":     "annotation",
                    "classification": {
                        "name":     name,
                        "colorRGB": _rgb_to_int(*col),
                    },
                    "class_index": c,
                    "area_px2":    float(area),
                },
            })

        all_features.extend(features)
        if features:
            sz_approx = sum(len(str(f)) for f in features) / 1e6
            print(f"  {name:<20} {len(features):>5,} polygons  ~{sz_approx:.2f} MB")

    # Combined GeoJSON — all classes
    combined_gj = seg_dir / "segmentation_all_classes.geojson"
    with open(str(combined_gj), "w") as fh:
        json.dump({"type": "FeatureCollection", "features": all_features},
                  fh, separators=(",", ":"))

    sz_all = combined_gj.stat().st_size / 1e6
    print(f"\n  Combined : {len(all_features):,} polygons  "
          f"{sz_all:.2f} MB  → {combined_gj.name}")

    return {"combined": str(combined_gj)}


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL — run stitching + GeoJSON for one slide
# ─────────────────────────────────────────────────────────────────────────────

def run_stitching(
    wsi_path:    str,
    cfg:         PipelineConfig = None,
    step_x:      Optional[int]  = None,
    step_y:      Optional[int]  = None,
    min_area_px: int             = 100,
) -> dict:
    """
    Run stitching + GeoJSON extraction for one WSI.

    Continues the pipeline after run_segmentation().

    All paths are derived from wsi_path and cfg:
        slide_name   = Path(wsi_path).stem
        seg_dir      = cfg.OUT_DIR / slide_name / "segmentation"
        manifest_csv = seg_dir / "manifest.csv"

    Final outputs written to seg_dir:
        <slide>_segmentation.png
        segmentation_all_classes.geojson

    After both exist, ALL .npy files in seg_dir are deleted automatically.

    Parameters
    ----------
    wsi_path    : path to .svs / .tif — used to derive slide_name
    cfg         : PipelineConfig (defaults to config.cfg singleton)
    step_x      : tile step in X pixels (None = auto-infer from manifest)
    step_y      : tile step in Y pixels (None = auto-infer from manifest)
    min_area_px : minimum polygon area to keep in GeoJSON

    Returns
    -------
    dict:
        slide_name      — str
        manifest_csv    — path used as input
        stitched_png    — path to colour PNG with legend
        geojson         — dict {"combined": path}
        meta            — full stitch metadata dict
        npys_removed    — int, number of .npy files deleted
    """
    if cfg is None:
        cfg = default_cfg

    slide_name   = Path(wsi_path).stem
    seg_dir      = Path(cfg.OUT_DIR) / slide_name / "segmentation"
    manifest_csv = seg_dir / "manifest.csv"

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}\n"
            f"Run run_segmentation(wsi_path, cfg) first."
        )

    print(f"\n{'='*55}")
    print(f"  Stitching + GeoJSON")
    print(f"  Slide      : {slide_name}")
    print(f"  Manifest   : {manifest_csv}")
    print(f"  Output     : {seg_dir}")
    print(f"{'='*55}")

    # Step 1 — stitch .npy tiles → combined canvas + colour PNG
    meta = _stitch_segmentation(
        manifest_csv = str(manifest_csv),
        seg_dir      = seg_dir,
        slide_name   = slide_name,
        step_x       = step_x,
        step_y       = step_y,
    )

    # Step 2 — extract GeoJSON polygons
    gj_paths = _extract_geojson(
        meta        = meta,
        seg_dir     = seg_dir,
        min_area_px = min_area_px,
    )

    # Step 3 — automatic cleanup: delete ALL .npy files only once both
    # final outputs exist.
    png_path = Path(meta["stitched_seg_png"])
    gj_path  = Path(gj_paths["combined"])

    n_removed = 0
    if png_path.exists() and gj_path.exists():
        n_removed = _cleanup_npys(seg_dir)
    else:
        print(f"  WARN: final outputs incomplete — skipping .npy cleanup")

    print(f"\n  Done.")
    print(f"  Colour PNG   : {png_path.name}")
    print(f"  Combined GJ  : {gj_path.name}")
    print(f"  .npy removed : {n_removed:,}")

    return {
        "slide_name":   slide_name,
        "manifest_csv": str(manifest_csv),
        "stitched_png": str(png_path),
        "geojson":      gj_paths,
        "meta":         meta,
        "npys_removed": n_removed,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's add_config_fields_to_parser() for every PipelineConfig
# field (--out-dir, --wsi-path, ...), plus a small set of stitch.py-specific
# flags for arguments that aren't part of PipelineConfig (step_x/step_y
# overrides, min_area_px), plus --from-json to pick up a config saved
# earlier via `config.py --print-config`.
#
# stitch.py can't simply delegate to config.config_from_args() the way
# tessellate.py / segmenter.py do, because it has its own extra flags
# (--stitch-step-x/-y, --min-area-px) that aren't PipelineConfig fields.
# Instead --from-json is added directly here, alongside those flags, using
# the same "only override fields the user explicitly passed" logic as
# config_from_args().

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="stitch.py",
        description="Step 3 of the ViSpace pipeline — stitch per-tile "
                     "segmentation into a WSI-level canvas + GeoJSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_config_fields_to_parser(parser)  # every PipelineConfig field

    parser.add_argument(
        "--from-json", type=str, default=None,
        help="Load a PipelineConfig previously saved via "
             "`config.py --print-config`. Any other --flag passed alongside "
             "this one (including --stitch-step-x/-y, --min-area-px) "
             "overrides the corresponding value from the JSON file.",
    )

    stitch_group = parser.add_argument_group("stitch.py-specific overrides")
    stitch_group.add_argument(
        "--stitch-step-x", type=int, default=None, metavar="PX",
        help="Override the tile step in X (px) used when stitching. "
             "Default: auto-infer from manifest.csv (recommended). Distinct "
             "from --step-x, which only sets cfg.STEP_X (informational).",
    )
    stitch_group.add_argument(
        "--stitch-step-y", type=int, default=None, metavar="PX",
        help="Override the tile step in Y (px) used when stitching. "
             "Default: auto-infer from manifest.csv (recommended). Distinct "
             "from --step-y, which only sets cfg.STEP_Y (informational).",
    )
    stitch_group.add_argument(
        "--min-area-px", type=int, default=100, metavar="PX2",
        help="Minimum polygon area (px^2) to keep when extracting GeoJSON.",
    )

    args = parser.parse_args(argv)

    # Build cfg — from JSON (with explicit-flag overrides layered on top)
    # or from CLI flags alone, same semantics as config.config_from_args().
    if args.from_json:
        base_cfg = config_from_json(args.from_json)
        defaults = PipelineConfig()
        overrides = {
            f.name: getattr(args, f.name)
            for f in fields(PipelineConfig)
            if getattr(args, f.name) != getattr(defaults, f.name)
        }
        cfg = replace(base_cfg, **overrides)
    else:
        overrides = {f.name: getattr(args, f.name) for f in fields(PipelineConfig)}
        cfg = replace(PipelineConfig(), **overrides)

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        parser.error(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    manifest_csv = Path(cfg.OUT_DIR) / Path(cfg.WSI_PATH).stem / "segmentation" / "manifest.csv"
    if not manifest_csv.exists():
        parser.error(
            f"manifest.csv not found: {manifest_csv}. "
            f"Run segmenter.py for this slide (with the same --out-dir) first."
        )

    # --step-x / --step-y only set cfg.STEP_X / cfg.STEP_Y, which are
    # informational fields that run_stitching() never reads — the actual
    # stitching step is controlled by --stitch-step-x / --stitch-step-y
    # (defaulting to None = auto-infer from manifest.csv). Someone passing
    # --step-x expecting it to affect stitching would otherwise get no error
    # and no effect, so warn explicitly instead of failing silently.
    _defaults = PipelineConfig()
    if cfg.STEP_X != _defaults.STEP_X and args.stitch_step_x is None:
        print(
            f"  WARNING: --step-x={cfg.STEP_X} was set, but stitch.py does not use "
            f"cfg.STEP_X (it's informational only). Stitching will still "
            f"auto-infer its step from manifest.csv. Use --stitch-step-x "
            f"{cfg.STEP_X} instead if you meant to override the stitching step.",
        )
    if cfg.STEP_Y != _defaults.STEP_Y and args.stitch_step_y is None:
        print(
            f"  WARNING: --step-y={cfg.STEP_Y} was set, but stitch.py does not use "
            f"cfg.STEP_Y (it's informational only). Stitching will still "
            f"auto-infer its step from manifest.csv. Use --stitch-step-y "
            f"{cfg.STEP_Y} instead if you meant to override the stitching step.",
        )

    run_stitching(
        wsi_path    = cfg.WSI_PATH,
        cfg         = cfg,
        step_x      = args.stitch_step_x,
        step_y      = args.stitch_step_y,
        min_area_px = args.min_area_px,
    )


if __name__ == "__main__":
    main()