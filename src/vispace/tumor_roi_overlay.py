#!/usr/bin/env python3
"""
tumor_roi_overlay.py
====================
Step 5 of the ViSpace pipeline (Step 1 of the spatial-analysis stage).

Clusters tumor tiles into spatially-connected clumps, builds non-overlapping
ROI boxes of a chosen physical size, and renders overlay visualisations.

Pipeline
--------
1. FILTER  — keep tiles where frac_Tumour >= cfg.ROI_MIN_TUMOR_FRAC
2. CLUSTER — BFS on the wx/wy tile lattice (8-connected by default)
2b. MERGE  — optionally merge clumps within cfg.ROI_MERGE_GAP_UM of each other
3. BOX     — tile each clump with non-overlapping square ROI boxes
4. DEDUPE  — reject any box that overlaps a previously accepted box
5. OVERLAY — write pseudo-thumbnail PNG (+ WSI thumbnail if WSI file exists)

Inputs (all inferred from wsi_path + cfg)
-----------------------------------------
  manifest : cfg.OUT_DIR/<slide>/segmentation/manifest.csv
             Must contain: wx, wy, frac_Tumour, frac_Stroma,
             frac_Inflammatory, frac_Necrosis, frac_Others
             Written by run_segmentation() (step 2 / segmenter.py).

Outputs
-------
  cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_roi_overlay/
      tumor_roi_boxes.csv
      tumor_roi_boxes_pseudo_thumbnail.png
      tumor_roi_boxes_wsi_thumbnail.png   (only when wsi_path exists on disk;
                                            requires openslide-python)

Pipeline position
-----------------
    tessellate.py  →  segmenter.py  →  stitch.py  →  tumor_roi_overlay.py  →  cluster_tils_tsr_score.py

Usage (as a library)
---------------------
    from vispace import run_tumor_roi_overlay
    from vispace import cfg

    run_tumor_roi_overlay("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
------------------------------
Same shared flags as config.py / tessellate.py / segmenter.py / stitch.py —
every PipelineConfig field is available here too, including the ROI_* knobs
(--roi-size-um, --roi-min-tumor-frac, --roi-max-necrosis, etc.). Also
supports --from-json to pick up a config saved earlier via
`config.py --print-config`.

    # minimal — requires stitch.py to have already run for this slide
    python tumor_roi_overlay.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python tumor_roi_overlay.py --from-json run_config.json

    # continue from a saved config but override one ROI knob
    python tumor_roi_overlay.py --from-json run_config.json --roi-size-um 150

    # see every available flag
    python tumor_roi_overlay.py --help
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

from .config import cfg as default_cfg, PipelineConfig


# ---------------------------------------------------------------------------
# Module-level constants  (unchanged from original)
# ---------------------------------------------------------------------------

FRACTION_COLS = [
    "frac_Tumour",
    "frac_Stroma",
    "frac_Inflammatory",
    "frac_Necrosis",
    "frac_Others",
]

CLASS_ORDER = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]

COLORS_BGR = [
    (0,   0, 255),    # Tumour        — red
    (0, 200,   0),    # Stroma        — green
    (255, 100, 0),    # Inflammatory  — blue
    (0, 165, 255),    # Necrosis      — orange
    (220,   0, 220),  # Others        — magenta
]

CLASS_COLORS = {
    cls: np.array(bgr[::-1]) / 255.0
    for cls, bgr in zip(CLASS_ORDER, COLORS_BGR)
}

ROI_BOX_COLOR = "#000000"
CLUSTER_OUTLINE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5",
]


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def infer_tile_step(df: pd.DataFrame) -> Tuple[int, int]:
    """Infer tile spacing from wx/wy coordinate grid (median nearest-neighbor gap)."""
    def step(vals):
        u = np.sort(pd.Series(vals).dropna().unique())
        d = np.diff(u)
        d = d[d > 0]
        return int(round(np.median(d))) if len(d) else 256

    return step(df["wx"]), step(df["wy"])


# ---------------------------------------------------------------------------
# Step 1: filter tumor tiles
# ---------------------------------------------------------------------------

def filter_tumor_tiles(df: pd.DataFrame, min_tumor_frac: float) -> pd.DataFrame:
    return df[df["frac_Tumour"] >= min_tumor_frac].copy()


# ---------------------------------------------------------------------------
# Step 2: cluster tumor tiles into spatial clumps
# ---------------------------------------------------------------------------

def cluster_tiles(
    tum: pd.DataFrame,
    step_x: int,
    step_y: int,
    x0: int,
    y0: int,
    connectivity: int = 8,
    min_cluster_tiles: int = 3,
) -> pd.DataFrame:
    """Assign a cluster_id to each tumor tile via grid-connectivity BFS.

    connectivity: 4 (von Neumann) or 8 (Moore, default — tolerates diagonal
    adjacency, which matters because real tissue/tumor edges are irregular).
    min_cluster_tiles: clumps smaller than this are dropped as noise/speckle.
    """
    tum = tum.copy()
    gx = ((tum["wx"] - x0) / step_x).round().astype(int)
    gy = ((tum["wy"] - y0) / step_y).round().astype(int)
    tum["_gx"] = gx
    tum["_gy"] = gy

    coord_to_idx: Dict[Tuple[int, int], int] = {}
    for idx, (cx, cy) in zip(tum.index, zip(gx, gy)):
        coord_to_idx[(cx, cy)] = idx

    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbors = [
            (dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
            if not (dx == 0 and dy == 0)
        ]

    visited = set()
    cluster_id = 0
    cluster_assignment = {}

    for coord, idx in coord_to_idx.items():
        if coord in visited:
            continue
        q = deque([coord])
        visited.add(coord)
        members = [idx]
        while q:
            cx, cy = q.popleft()
            for dx, dy in neighbors:
                n = (cx + dx, cy + dy)
                if n in coord_to_idx and n not in visited:
                    visited.add(n)
                    members.append(coord_to_idx[n])
                    q.append(n)
        if len(members) >= min_cluster_tiles:
            for m in members:
                cluster_assignment[m] = cluster_id
            cluster_id += 1

    tum["cluster_id"] = tum.index.map(lambda i: cluster_assignment.get(i, -1))
    tum = tum[tum["cluster_id"] >= 0].copy()
    return tum


def _bbox_gap(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> float:
    """Edge-to-edge gap between two axis-aligned bboxes (0 if overlapping/touching)."""
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    dx = max(bx1 - ax2, ax1 - bx2, 0)
    dy = max(by1 - ay2, ay1 - by2, 0)
    return math.hypot(dx, dy)


def merge_nearby_clusters(
    tum: pd.DataFrame,
    merge_gap_px: float,
) -> pd.DataFrame:
    """Merge clumps whose bounding boxes are within merge_gap_px of each other.

    Only merges clumps that are actually close in space — this is what lets a
    1-2 tile fragment sitting right next to (or just across a small gap from)
    a bigger clump get absorbed into it, WITHOUT merging small clumps that
    happen to be far apart elsewhere on the slide. Distant small clumps stay
    separate, as they should — they're real, independent foci.

    Uses union-find over pairwise bbox gaps, so a chain of close clumps
    (A near B, B near C) merges transitively into one group.
    """
    if tum.empty or merge_gap_px <= 0:
        return tum

    tum = tum.copy()
    bbox = tum.groupby("cluster_id").agg(
        x_min=("wx", "min"), x_max=("wx", "max"),
        y_min=("wy", "min"), y_max=("wy", "max"),
    )
    ids = bbox.index.tolist()
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(ids)):
        bi = tuple(bbox.loc[ids[i], ["x_min", "y_min", "x_max", "y_max"]])
        for j in range(i + 1, len(ids)):
            bj = tuple(bbox.loc[ids[j], ["x_min", "y_min", "x_max", "y_max"]])
            if _bbox_gap(bi, bj) <= merge_gap_px:
                union(ids[i], ids[j])

    root_to_new_id = {}
    next_id = 0
    remap = {}
    for cid in ids:
        root = find(cid)
        if root not in root_to_new_id:
            root_to_new_id[root] = next_id
            next_id += 1
        remap[cid] = root_to_new_id[root]

    tum["cluster_id"] = tum["cluster_id"].map(remap)
    return tum


# ---------------------------------------------------------------------------
# Steps 3 + 4: build non-overlapping ROI boxes
# ---------------------------------------------------------------------------

def boxes_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def window_composition(inside: pd.DataFrame) -> Dict[str, float]:
    """Mean class composition inside an ROI window (description only, no scoring)."""
    return {
        "tumor":         float(inside["frac_Tumour"].mean()),
        "stroma":        float(inside["frac_Stroma"].mean()),
        "inflammatory":  float(inside["frac_Inflammatory"].mean()),
        "necrosis":      float(inside["frac_Necrosis"].mean()),
        "others":        float(inside["frac_Others"].mean()),
    }


def build_rois_for_cluster(
    cluster_tiles_df: pd.DataFrame,
    full_df: pd.DataFrame,
    cluster_id: int,
    roi_size_px: float,
    accepted_boxes: List[Tuple[float, float, float, float]],
    max_necrosis: float,
) -> List[dict]:
    """Tile a single tumor clump with one or more non-overlapping ROI boxes.

    Strategy: snap a regular grid (spacing = roi_size_px) anchored at the
    clump's bounding box, keep grid cells that actually contain >=1 tumor
    tile from this cluster, and reject any cell that overlaps an
    already-accepted box (from this cluster or any other).
    """
    x_min = cluster_tiles_df["wx"].min() - roi_size_px / 2
    x_max = cluster_tiles_df["wx"].max() + roi_size_px / 2
    y_min = cluster_tiles_df["wy"].min() - roi_size_px / 2
    y_max = cluster_tiles_df["wy"].max() + roi_size_px / 2

    n_cols = max(1, int(math.ceil((x_max - x_min) / roi_size_px)))
    n_rows = max(1, int(math.ceil((y_max - y_min) / roi_size_px)))

    # rank candidate grid cells by tumor signal — best ROI placed first
    candidates = []
    for r in range(n_rows):
        for c in range(n_cols):
            bx1 = x_min + c * roi_size_px
            by1 = y_min + r * roi_size_px
            bx2 = bx1 + roi_size_px
            by2 = by1 + roi_size_px
            in_cell = cluster_tiles_df[
                (cluster_tiles_df["wx"] >= bx1) & (cluster_tiles_df["wx"] < bx2)
                & (cluster_tiles_df["wy"] >= by1) & (cluster_tiles_df["wy"] < by2)
            ]
            if in_cell.empty:
                continue
            candidates.append((in_cell["frac_Tumour"].sum(), bx1, by1, bx2, by2))

    candidates.sort(key=lambda t: t[0], reverse=True)

    results = []
    for _, bx1, by1, bx2, by2 in candidates:
        box = (bx1, by1, bx2, by2)
        if any(boxes_overlap(box, b) for b in accepted_boxes):
            continue

        inside = full_df[
            (full_df["wx"] >= bx1) & (full_df["wx"] < bx2)
            & (full_df["wy"] >= by1) & (full_df["wy"] < by2)
        ]
        if inside.empty:
            continue
        comp = window_composition(inside)
        if comp["necrosis"] > max_necrosis:
            continue

        results.append({
            "cluster_id": cluster_id,
            "x_min": float(bx1), "y_min": float(by1),
            "x_max": float(bx2), "y_max": float(by2),
            "roi_tiles": int(len(inside)),
            **comp,
        })
        accepted_boxes.append(box)

    return results


def generate_rois(
    df: pd.DataFrame,
    min_tumor_frac: float,
    roi_size_um: float,
    mpp: float,
    max_necrosis: float,
    min_cluster_tiles: int,
    connectivity: int,
    merge_gap_um: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: filter → cluster → merge nearby clumps → box → dedupe.

    Returns (rois_df, tumor_tiles_with_cluster_id_df).
    """
    roi_size_px = roi_size_um / mpp

    step_x, step_y = infer_tile_step(df)
    x0, y0 = int(df["wx"].min()), int(df["wy"].min())

    tum = filter_tumor_tiles(df, min_tumor_frac)
    if tum.empty:
        return pd.DataFrame(), tum

    tum = cluster_tiles(
        tum, step_x, step_y, x0, y0,
        connectivity=connectivity,
        min_cluster_tiles=min_cluster_tiles,
    )
    if tum.empty:
        return pd.DataFrame(), tum

    if merge_gap_um > 0:
        merge_gap_px = merge_gap_um / mpp
        tum = merge_nearby_clusters(tum, merge_gap_px)

    # process clusters largest-first so the biggest clumps get first pick of
    # space when boxes from different clusters compete near a boundary
    cluster_order = (
        tum.groupby("cluster_id").size()
        .sort_values(ascending=False).index.tolist()
    )

    accepted_boxes: List[Tuple[float, float, float, float]] = []
    all_rois: List[dict] = []

    for cid in tqdm(cluster_order, desc="Building ROI boxes", unit="cluster"):
        sub = tum[tum["cluster_id"] == cid]
        rois = build_rois_for_cluster(
            sub, df, cid, roi_size_px, accepted_boxes, max_necrosis
        )
        all_rois.extend(rois)

    rois_df = pd.DataFrame(all_rois)
    if not rois_df.empty:
        rois_df = rois_df.sort_values(
            ["cluster_id", "tumor"], ascending=[True, False]
        ).reset_index(drop=True)
        rois_df.insert(0, "roi_id", range(1, len(rois_df) + 1))
        rois_df["roi_size_px"] = round(roi_size_px, 1)
        rois_df["roi_size_um"] = roi_size_um
        rois_df["mpp"]         = mpp

    return rois_df, tum


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def make_pseudo_thumbnail(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Create a low-res class-color map from tile coordinates (one pixel per tile)."""
    step_x, step_y = infer_tile_step(df)
    x0, y0 = int(df["wx"].min()), int(df["wy"].min())
    x1 = int(df["wx"].max() + step_x)
    y1 = int(df["wy"].max() + step_y)

    cols = int(math.ceil((x1 - x0) / step_x))
    rows = int(math.ceil((y1 - y0) / step_y))
    img  = np.ones((rows, cols, 3), dtype=float)

    class_map = {
        "frac_Tumour":       "Tumour",
        "frac_Stroma":       "Stroma",
        "frac_Inflammatory": "Inflammatory",
        "frac_Necrosis":     "Necrosis",
        "frac_Others":       "Others",
    }

    vals    = df[FRACTION_COLS].to_numpy(float)
    dom_idx = vals.argmax(axis=1)

    for i, (_, r) in enumerate(
        tqdm(df.iterrows(), total=len(df),
             desc="Rendering pseudo-thumbnail", unit="tile", leave=False)
    ):
        c      = int(round((r["wx"] - x0) / step_x))
        rr     = int(round((r["wy"] - y0) / step_y))
        cls_col = FRACTION_COLS[dom_idx[i]]
        cls    = class_map[cls_col]
        alpha  = min(1.0, max(0.25, float(r[cls_col])))
        img[rr, c, :] = alpha * CLASS_COLORS[cls] + (1 - alpha) * np.ones(3)

    return img, (x0, y0, x1, y1)


def draw_legend(ax) -> None:
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=name)
        for name, color in CLASS_COLORS.items()
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="Tile class")


def overlay_on_pseudo_thumbnail(
    df: pd.DataFrame,
    rois: pd.DataFrame,
    out_png: Path,
    roi_size_um: float,
    color_by_cluster: bool,
    label_boxes: bool = "auto",
) -> None:
    img, extent = make_pseudo_thumbnail(df)
    x0, y0, x1, y1 = extent

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(img, extent=[x0, x1, y1, y0], interpolation="nearest")

    n_boxes = len(rois)
    if label_boxes == "auto":
        label_boxes = n_boxes <= 40

    if not rois.empty:
        for cid, grp in rois.groupby("cluster_id"):
            edge_color = (
                CLUSTER_OUTLINE_COLORS[int(cid) % len(CLUSTER_OUTLINE_COLORS)]
                if color_by_cluster else ROI_BOX_COLOR
            )
            cx1, cy1 = grp["x_min"].min(), grp["y_min"].min()
            cx2, cy2 = grp["x_max"].max(), grp["y_max"].max()
            outline = Rectangle(
                (cx1, cy1), cx2 - cx1, cy2 - cy1,
                fill=False, linewidth=1.2, linestyle="--",
                edgecolor=edge_color, alpha=0.6,
            )
            ax.add_patch(outline)
            ax.text(
                cx1, cy1 - 25, f"clump {int(cid)} (n={len(grp)})",
                fontsize=8, weight="bold", color=edge_color,
                bbox=dict(facecolor="white", alpha=0.7, pad=0.5, edgecolor="none"),
            )

        for _, r in rois.iterrows():
            edge_color = (
                CLUSTER_OUTLINE_COLORS[int(r["cluster_id"]) % len(CLUSTER_OUTLINE_COLORS)]
                if color_by_cluster else ROI_BOX_COLOR
            )
            rect = Rectangle(
                (r["x_min"], r["y_min"]),
                r["x_max"] - r["x_min"],
                r["y_max"] - r["y_min"],
                fill=False, linewidth=1.4, edgecolor=edge_color,
            )
            ax.add_patch(rect)
            if label_boxes:
                cx = (r["x_min"] + r["x_max"]) / 2
                cy = (r["y_min"] + r["y_max"]) / 2
                ax.text(
                    cx, cy, f"{int(r['roi_id'])}",
                    fontsize=7, weight="bold", color=edge_color,
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.55, pad=0.4, edgecolor="none"),
                )

    draw_legend(ax)
    n_clusters = rois["cluster_id"].nunique() if not rois.empty else 0
    ax.set_title(
        f"Tumor ROI boxes on segmentation-derived slide map\n"
        f"{n_boxes} ROI box(es) across {n_clusters} tumor clump(s) "
        f"— {roi_size_um:.0f} µm boxes"
    )
    ax.set_xlabel("WSI x coordinate (px)")
    ax.set_ylabel("WSI y coordinate (px)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def overlay_on_wsi_thumbnail(
    wsi_path: Path,
    rois: pd.DataFrame,
    out_png: Path,
    thumb_width: int,
    roi_size_um: float,
    color_by_cluster: bool,
    label_boxes: bool = "auto",
) -> Optional[float]:
    try:
        import openslide
    except ImportError as e:
        raise RuntimeError(
            "openslide-python is required for WSI overlay. "
            "Install openslide-python and OpenSlide."
        ) from e

    slide = openslide.OpenSlide(str(wsi_path))
    w, h  = slide.dimensions
    thumb_height = int(round(thumb_width * h / w))
    thumb = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")

    sx = thumb_width  / w
    sy = thumb_height / h

    n_boxes = len(rois)
    if label_boxes == "auto":
        label_boxes = n_boxes <= 40

    fig, ax = plt.subplots(figsize=(13, 13 * thumb_height / thumb_width))
    ax.imshow(thumb)

    if not rois.empty:
        for cid, grp in rois.groupby("cluster_id"):
            edge_color = (
                CLUSTER_OUTLINE_COLORS[int(cid) % len(CLUSTER_OUTLINE_COLORS)]
                if color_by_cluster else ROI_BOX_COLOR
            )
            cx1, cy1 = grp["x_min"].min() * sx, grp["y_min"].min() * sy
            cx2, cy2 = grp["x_max"].max() * sx, grp["y_max"].max() * sy
            outline = Rectangle(
                (cx1, cy1), cx2 - cx1, cy2 - cy1,
                fill=False, linewidth=1.0, linestyle="--",
                edgecolor=edge_color, alpha=0.6,
            )
            ax.add_patch(outline)

        for _, r in rois.iterrows():
            edge_color = (
                CLUSTER_OUTLINE_COLORS[int(r["cluster_id"]) % len(CLUSTER_OUTLINE_COLORS)]
                if color_by_cluster else ROI_BOX_COLOR
            )
            x  = r["x_min"] * sx
            y  = r["y_min"] * sy
            ww = (r["x_max"] - r["x_min"]) * sx
            hh = (r["y_max"] - r["y_min"]) * sy
            rect = Rectangle((x, y), ww, hh, fill=False,
                              linewidth=1.4, edgecolor=edge_color)
            ax.add_patch(rect)
            if label_boxes:
                ax.text(
                    x + ww / 2, y + hh / 2, f"{int(r['roi_id'])}",
                    fontsize=7, weight="bold", color=edge_color,
                    ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.55, pad=0.4, edgecolor="none"),
                )

    ax.axis("off")
    n_clusters = rois["cluster_id"].nunique() if not rois.empty else 0
    ax.set_title(
        f"Tumor ROI boxes on WSI thumbnail — "
        f"{n_boxes} box(es) across {n_clusters} clump(s), {roi_size_um:.0f} µm each"
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    try:
        return float(slide.properties.get("openslide.mpp-x"))
    except (TypeError, ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------

def run_tumor_roi_overlay(
    wsi_path: str,
    cfg: PipelineConfig = None,
) -> dict:
    """
    Run tumor ROI box generation + overlay for one WSI.

    All inputs and outputs are derived from wsi_path and cfg — the caller
    never needs to pass manifest_path, out_dir, slide_name, or any other
    path manually.

    Reads
    -----
    cfg.OUT_DIR/<slide>/segmentation/manifest.csv
        Written by run_segmentation().  Must contain wx, wy and five
        frac_* columns.

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_roi_overlay/
        tumor_roi_boxes.csv
        tumor_roi_boxes_pseudo_thumbnail.png
        tumor_roi_boxes_wsi_thumbnail.png   (only when wsi_path is a real file)

    Parameters
    ----------
    wsi_path : str
        Path to .svs / .tif.  Used to derive slide_name and, if the file
        exists on disk, to render the WSI thumbnail overlay.
    cfg : PipelineConfig
        Pipeline config.  All tuning knobs are read from cfg:
            cfg.MPP                  microns-per-pixel (default 0.25)
            cfg.ROI_SIZE_UM          ROI box edge in microns (default 200)
            cfg.ROI_MIN_TUMOR_FRAC   min frac_Tumour to include tile (default 0.20)
            cfg.ROI_MAX_NECROSIS     max mean frac_Necrosis allowed in a box (0.50)
            cfg.ROI_MIN_CLUSTER_TILES min tiles to keep a clump (default 3)
            cfg.ROI_CONNECTIVITY     grid connectivity, 4 or 8 (default 8)
            cfg.ROI_MERGE_GAP_UM     merge gap in µm, 0 = disabled (default 60)
            cfg.ROI_THUMB_WIDTH      thumbnail width in pixels (default 1800)
            cfg.ROI_COLOR_BY_CLUSTER colour boxes by cluster id (default True)

    Returns
    -------
    dict
        slide_name        str
        manifest_csv      str   path of input manifest
        roi_csv           str   path of output tumor_roi_boxes.csv
        pseudo_png        str   path of pseudo-thumbnail PNG
        wsi_png           str | None
        n_boxes           int
        n_clusters        int
    """
    if cfg is None:
        cfg = default_cfg

    slide_name   = Path(wsi_path).stem
    manifest_csv = Path(cfg.OUT_DIR) / slide_name / "segmentation" / "manifest.csv"
    out_dir      = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "tumor_roi_overlay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}\n"
            f"Run run_segmentation(wsi_path, cfg) first."
        )

    # ── Config knobs with safe fallbacks ────────────────────────────────────
    mpp               = getattr(cfg, "MPP",                   0.25)
    roi_size_um       = getattr(cfg, "ROI_SIZE_UM",           200.0)
    min_tumor_frac    = getattr(cfg, "ROI_MIN_TUMOR_FRAC",    0.20)
    max_necrosis      = getattr(cfg, "ROI_MAX_NECROSIS",      0.50)
    min_cluster_tiles = getattr(cfg, "ROI_MIN_CLUSTER_TILES", 3)
    connectivity      = getattr(cfg, "ROI_CONNECTIVITY",      8)
    merge_gap_um      = getattr(cfg, "ROI_MERGE_GAP_UM",      60.0)
    thumb_width       = getattr(cfg, "ROI_THUMB_WIDTH",       1800)
    color_by_cluster  = getattr(cfg, "ROI_COLOR_BY_CLUSTER",  True)

    print(f"\n{'='*55}")
    print(f"  Tumor ROI overlay")
    print(f"  Slide      : {slide_name}")
    print(f"  Manifest   : {manifest_csv}")
    print(f"  ROI size   : {roi_size_um:.0f} µm  →  {roi_size_um/mpp:.1f} px  (mpp={mpp})")
    print(f"  Output     : {out_dir}")
    print(f"{'='*55}")

    # ── Load manifest ────────────────────────────────────────────────────────
    df = pd.read_csv(manifest_csv)
    missing = [c for c in ["wx", "wy"] + FRACTION_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # ── Generate ROIs ────────────────────────────────────────────────────────
    rois, _tum = generate_rois(
        df=df,
        min_tumor_frac=min_tumor_frac,
        roi_size_um=roi_size_um,
        mpp=mpp,
        max_necrosis=max_necrosis,
        min_cluster_tiles=min_cluster_tiles,
        connectivity=connectivity,
        merge_gap_um=merge_gap_um,
    )

    # ── Save ROI CSV ─────────────────────────────────────────────────────────
    roi_csv = out_dir / "tumor_roi_boxes.csv"
    rois.to_csv(roi_csv, index=False)
    print(f"  Wrote: {roi_csv.name}  ({len(rois)} boxes)")

    if rois.empty:
        print(
            "  WARNING: no ROI boxes generated. "
            "Try lowering cfg.ROI_MIN_TUMOR_FRAC or cfg.ROI_MIN_CLUSTER_TILES."
        )
    else:
        n_clusters = rois["cluster_id"].nunique()
        print(
            f"  Generated {len(rois)} ROI box(es) across "
            f"{n_clusters} tumor clump(s)."
        )

    # ── Pseudo-thumbnail overlay ─────────────────────────────────────────────
    pseudo_png = out_dir / "tumor_roi_boxes_pseudo_thumbnail.png"
    with tqdm(total=1, desc="Saving pseudo-thumbnail overlay", unit="image"):
        overlay_on_pseudo_thumbnail(
            df, rois, pseudo_png, roi_size_um, color_by_cluster, label_boxes="auto"
        )
    print(f"  Wrote: {pseudo_png.name}")

    # ── WSI thumbnail overlay (only when file exists on disk) ────────────────
    wsi_png    = None
    wsi_exists = Path(wsi_path).exists()
    if wsi_exists:
        wsi_png_path = out_dir / "tumor_roi_boxes_wsi_thumbnail.png"
        with tqdm(total=1, desc="Saving WSI thumbnail overlay", unit="image"):
            slide_mpp = overlay_on_wsi_thumbnail(
                Path(wsi_path), rois, wsi_png_path,
                thumb_width, roi_size_um, color_by_cluster, label_boxes="auto",
            )
        wsi_png = str(wsi_png_path)
        print(f"  Wrote: {wsi_png_path.name}")
        if slide_mpp:
            print(f"  (WSI metadata mpp-x = {slide_mpp:.4f} µm/px, for reference)")
    else:
        print(f"  WSI file not found on disk — skipping WSI thumbnail overlay.")

    print(f"\n  Done.")

    return {
        "slide_name":  slide_name,
        "manifest_csv": str(manifest_csv),
        "roi_csv":     str(roi_csv),
        "pseudo_png":  str(pseudo_png),
        "wsi_png":     wsi_png,
        "n_boxes":     len(rois),
        "n_clusters":  int(rois["cluster_id"].nunique()) if not rois.empty else 0,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) — every PipelineConfig
# field (including the ROI_* knobs) is available as a flag, plus
# --from-json to pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python tumor_roi_overlay.py --from-json run_config.json

def main(argv=None) -> None:
    from .config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    manifest_csv = Path(cfg.OUT_DIR) / Path(cfg.WSI_PATH).stem / "segmentation" / "manifest.csv"
    if not manifest_csv.exists():
        raise SystemExit(
            f"manifest.csv not found: {manifest_csv}. "
            f"Run segmenter.py for this slide (with the same --out-dir) first."
        )

    run_tumor_roi_overlay(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()