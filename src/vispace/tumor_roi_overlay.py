#!/usr/bin/env python3
"""
tumor_roi_overlay.py
====================
Tumour-focus detection and representative tumour ROI generation for ViSpace.

This version keeps the public entry point ``run_tumor_roi_overlay`` but changes
its spatial model:

1. FILTER tumour-positive segmentation tiles.
2. BUILD physical tumour-focus geometries from tile footprints.
3. REPAIR only small segmentation gaps using a physical closing distance.
4. ASSIGN a stable ``focus_id`` to each tumour tile.
5. RANK candidate ROI centres using local tumour continuity.
6. PLACE non-overlapping square tumour ROIs for downstream image extraction.
7. EXPORT tumour foci, focus-tile membership, ROI boxes, and QC overlays.

The final image ROIs remain square.  The tumour focus is the biological/spatial
object; the ROI is a representative image sample from that focus rather than a
full-clump tiling product.  ``ROI_MAX_ROIS_PER_FOCUS`` controls sampling depth.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, box as shapely_box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from .config import cfg as default_cfg, PipelineConfig


FRACTION_COLS = [
    "frac_Tumour",
    "frac_Stroma",
    "frac_Inflammatory",
    "frac_Necrosis",
    "frac_Others",
]
CLASS_ORDER = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]
CLASS_COLORS = {
    "Tumour": np.array([220, 30, 30], dtype=float) / 255.0,
    "Stroma": np.array([30, 180, 30], dtype=float) / 255.0,
    "Inflammatory": np.array([30, 100, 255], dtype=float) / 255.0,
    "Necrosis": np.array([255, 165, 0], dtype=float) / 255.0,
    "Others": np.array([200, 0, 200], dtype=float) / 255.0,
}
FOCUS_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#1b9e77",
]

# Visualization is intentionally not part of PipelineConfig.
# These are presentation-only implementation defaults.
_THUMBNAIL_WIDTH_PX = 1800
_COLOR_BY_FOCUS = True



def make_valid(geom: BaseGeometry) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def iter_polygon_parts(geom: BaseGeometry) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for g in geom.geoms:
            yield from iter_polygon_parts(g)


def infer_tile_step(df: pd.DataFrame) -> Tuple[int, int]:
    """Infer median x/y tile-centre spacing from the manifest."""
    def _step(vals: pd.Series) -> int:
        u = np.sort(pd.Series(vals).dropna().unique())
        d = np.diff(u)
        d = d[d > 0]
        return int(round(np.median(d))) if len(d) else 256
    return _step(df["wx"]), _step(df["wy"])


def validate_manifest(df: pd.DataFrame) -> None:
    required = ["wx", "wy"] + FRACTION_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")


def tile_footprint(wx: float, wy: float, step_x: float, step_y: float) -> Polygon:
    return shapely_box(
        wx - step_x / 2.0,
        wy - step_y / 2.0,
        wx + step_x / 2.0,
        wy + step_y / 2.0,
    )


def filter_tumor_tiles(df: pd.DataFrame, min_tumor_frac: float) -> pd.DataFrame:
    return df.loc[df["frac_Tumour"] >= min_tumor_frac].copy()


def _physical_closing(geom: BaseGeometry, repair_gap_px: float) -> BaseGeometry:
    """Close only narrow gaps; this is segmentation repair, not focus merging."""
    if geom is None or geom.is_empty or repair_gap_px <= 0:
        return geom
    r = repair_gap_px / 2.0
    return make_valid(geom.buffer(r, join_style=1).buffer(-r, join_style=1))


def build_tumor_foci(
    df: pd.DataFrame,
    min_tumor_frac: float,
    mpp: float,
    repair_gap_um: float,
    min_focus_area_um2: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create tumour-focus polygons and assign every retained tumour tile a focus."""
    step_x, step_y = infer_tile_step(df)
    tum = filter_tumor_tiles(df, min_tumor_frac)
    if tum.empty:
        return pd.DataFrame(), tum

    footprints = [
        tile_footprint(float(r.wx), float(r.wy), step_x, step_y)
        for r in tum.itertuples()
    ]
    dissolved = make_valid(unary_union(footprints))
    dissolved = _physical_closing(dissolved, repair_gap_um / mpp)

    min_area_px2 = max(0.0, min_focus_area_um2 / (mpp * mpp))
    parts = [p for p in iter_polygon_parts(dissolved) if p.area >= min_area_px2]
    if not parts:
        return pd.DataFrame(), tum.iloc[0:0].copy()

    # Stable slide-order IDs rather than area-rank IDs.
    parts.sort(key=lambda g: (g.representative_point().y, g.representative_point().x))
    focus_rows = []
    for focus_id, geom in enumerate(parts, start=1):
        rp = geom.representative_point()
        minx, miny, maxx, maxy = geom.bounds
        focus_rows.append({
            "focus_id": focus_id,
            "cluster_id": focus_id,  # backward-compatible alias
            "geometry": geom,
            "focus_area_px2": float(geom.area),
            "focus_area_um2": float(geom.area * mpp * mpp),
            "focus_perimeter_px": float(geom.length),
            "focus_perimeter_um": float(geom.length * mpp),
            "centroid_x": float(rp.x),
            "centroid_y": float(rp.y),
            "x_min": float(minx), "y_min": float(miny),
            "x_max": float(maxx), "y_max": float(maxy),
        })
    foci = pd.DataFrame(focus_rows)

    geoms = foci["geometry"].tolist()
    tree = STRtree(geoms)
    assignments: List[Optional[int]] = []
    for r in tum.itertuples():
        p = Point(float(r.wx), float(r.wy))
        hits = tree.query(p)
        chosen = None
        for h in hits:
            idx = int(h) if isinstance(h, (int, np.integer)) else geoms.index(h)
            if geoms[idx].covers(p):
                chosen = int(foci.iloc[idx]["focus_id"])
                break
        assignments.append(chosen)

    tum = tum.copy()
    tum["focus_id"] = assignments
    tum = tum.dropna(subset=["focus_id"]).copy()
    tum["focus_id"] = tum["focus_id"].astype(int)
    tum["cluster_id"] = tum["focus_id"]

    counts = tum.groupby("focus_id").size().rename("n_tumor_tiles")
    mean_t = tum.groupby("focus_id")["frac_Tumour"].mean().rename("mean_tumor_fraction")
    foci = foci.merge(counts, left_on="focus_id", right_index=True, how="left")
    foci = foci.merge(mean_t, left_on="focus_id", right_index=True, how="left")
    foci["n_tumor_tiles"] = foci["n_tumor_tiles"].fillna(0).astype(int)
    return foci, tum


def add_local_tumor_score(
    tum: pd.DataFrame,
    full_df: pd.DataFrame,
    tumor_signal_weight: float = 1.0,
    neighbor_weight: float = 0.5,
    necrosis_penalty: float = 0.5,
) -> pd.DataFrame:
    """Rank tumour tiles using configurable tumour/continuity/necrosis weights."""
    if tum.empty:
        return tum.copy()
    step_x, step_y = infer_tile_step(full_df)
    x0, y0 = float(full_df["wx"].min()), float(full_df["wy"].min())

    work = full_df[["wx", "wy", "frac_Tumour", "frac_Necrosis"]].copy()
    work["gx"] = np.rint((work["wx"] - x0) / step_x).astype(int)
    work["gy"] = np.rint((work["wy"] - y0) / step_y).astype(int)
    tmap = {(int(r.gx), int(r.gy)): float(r.frac_Tumour) for r in work.itertuples()}

    out = tum.copy()
    out["gx"] = np.rint((out["wx"] - x0) / step_x).astype(int)
    out["gy"] = np.rint((out["wy"] - y0) / step_y).astype(int)
    scores = []
    neighbor_means = []
    for r in out.itertuples():
        vals = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = tmap.get((int(r.gx) + dx, int(r.gy) + dy))
                if v is not None:
                    vals.append(v)
        nmean = float(np.mean(vals)) if vals else 0.0
        score = (
            tumor_signal_weight * float(r.frac_Tumour)
            + neighbor_weight * nmean
            - necrosis_penalty * float(r.frac_Necrosis)
        )
        neighbor_means.append(nmean)
        scores.append(score)
    out["neighbor_tumor_mean"] = neighbor_means
    out["roi_candidate_score"] = scores
    return out


def boxes_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def window_composition(inside: pd.DataFrame) -> Dict[str, float]:
    if inside.empty:
        return {"tumor": np.nan, "stroma": np.nan, "inflammatory": np.nan,
                "necrosis": np.nan, "others": np.nan}
    return {
        "tumor": float(inside["frac_Tumour"].mean()),
        "stroma": float(inside["frac_Stroma"].mean()),
        "inflammatory": float(inside["frac_Inflammatory"].mean()),
        "necrosis": float(inside["frac_Necrosis"].mean()),
        "others": float(inside["frac_Others"].mean()),
    }


def build_rois_for_focus(
    focus_tiles: pd.DataFrame,
    full_df: pd.DataFrame,
    focus_id: int,
    roi_size_px: float,
    accepted_boxes: List[Tuple[float, float, float, float]],
    min_roi_tumor_frac: float,
    max_necrosis: float,
    max_rois_per_focus: int = 0,
) -> List[dict]:
    """Greedily place high-quality square ROIs centred on tumour tiles."""
    if focus_tiles.empty:
        return []

    ranked = focus_tiles.sort_values(
        ["roi_candidate_score", "frac_Tumour"], ascending=False
    )
    half = roi_size_px / 2.0
    results: List[dict] = []

    for r in ranked.itertuples():
        box = (float(r.wx - half), float(r.wy - half), float(r.wx + half), float(r.wy + half))
        if any(boxes_overlap(box, old) for old in accepted_boxes):
            continue

        bx1, by1, bx2, by2 = box
        inside = full_df[
            (full_df["wx"] >= bx1) & (full_df["wx"] < bx2) &
            (full_df["wy"] >= by1) & (full_df["wy"] < by2)
        ]
        if inside.empty:
            continue
        comp = window_composition(inside)
        if not np.isfinite(comp["tumor"]) or comp["tumor"] < min_roi_tumor_frac:
            continue
        if np.isfinite(comp["necrosis"]) and comp["necrosis"] > max_necrosis:
            continue

        results.append({
            "focus_id": int(focus_id),
            "cluster_id": int(focus_id),  # compatibility
            "x_min": bx1, "y_min": by1, "x_max": bx2, "y_max": by2,
            "center_x": float(r.wx), "center_y": float(r.wy),
            "roi_tiles": int(len(inside)),
            "neighbor_tumor_mean": float(r.neighbor_tumor_mean),
            "roi_candidate_score": float(r.roi_candidate_score),
            **comp,
        })
        accepted_boxes.append(box)
        if max_rois_per_focus > 0 and len(results) >= max_rois_per_focus:
            break
    return results


def generate_rois(
    df: pd.DataFrame,
    min_tumor_frac: float,
    roi_size_um: float,
    mpp: float,
    max_necrosis: float,
    min_cluster_tiles: int = 3,
    connectivity: int = 8,
    merge_gap_um: float = 0.0,
    *,
    focus_repair_gap_um: Optional[float] = None,
    min_focus_area_um2: Optional[float] = None,
    min_roi_tumor_frac: Optional[float] = None,
    tumor_signal_weight: float = 1.0,
    neighbor_weight: float = 0.5,
    necrosis_penalty: float = 0.5,
    max_rois_per_focus: int = 3,
    return_foci: bool = False,
):
    """Generate tumour foci and representative square tumour ROIs.

    By default this retains the historical two-value return contract
    ``(rois, tumour_tiles)``.  ``run_tumor_roi_overlay`` calls this function with
    ``return_foci=True`` so that all generation logic lives in one place.

    ``connectivity`` is retained only for call compatibility with the old BFS
    implementation and is not used by the geometry-based focus detector.
    ``merge_gap_um`` is interpreted only as a legacy fallback for
    ``focus_repair_gap_um``; it is not used to merge biologically distinct foci.

    ROI selection is representative sampling, not full-clump tiling.  A finite
    ``max_rois_per_focus`` (default 3) caps sampling from large foci.
    """
    validate_manifest(df)
    step_x, step_y = infer_tile_step(df)
    if focus_repair_gap_um is None:
        focus_repair_gap_um = min(float(merge_gap_um), 50.0) if merge_gap_um > 0 else 25.0
    if min_focus_area_um2 is None:
        tile_area_um2 = step_x * step_y * mpp * mpp
        min_focus_area_um2 = max(1.0, min_cluster_tiles * tile_area_um2)
    if min_roi_tumor_frac is None:
        min_roi_tumor_frac = min_tumor_frac

    foci, tum = build_tumor_foci(
        df=df,
        min_tumor_frac=min_tumor_frac,
        mpp=mpp,
        repair_gap_um=focus_repair_gap_um,
        min_focus_area_um2=min_focus_area_um2,
    )
    if tum.empty or foci.empty:
        empty_rois = pd.DataFrame()
        return (empty_rois, tum, foci) if return_foci else (empty_rois, tum)

    tum = add_local_tumor_score(
        tum, df,
        tumor_signal_weight=tumor_signal_weight,
        neighbor_weight=neighbor_weight,
        necrosis_penalty=necrosis_penalty,
    )
    roi_size_px = roi_size_um / mpp
    accepted: List[Tuple[float, float, float, float]] = []
    rows: List[dict] = []

    # Large foci get first access to space, but IDs are retained unchanged.
    order = foci.sort_values("focus_area_px2", ascending=False)["focus_id"].tolist()
    for focus_id in tqdm(order, desc="Building tumour ROIs", unit="focus"):
        sub = tum.loc[tum["focus_id"] == focus_id]
        rows.extend(build_rois_for_focus(
            sub, df, int(focus_id), roi_size_px, accepted,
            min_roi_tumor_frac=min_roi_tumor_frac,
            max_necrosis=max_necrosis,
            max_rois_per_focus=max_rois_per_focus,
        ))

    rois = pd.DataFrame(rows)
    if not rois.empty:
        rois = rois.sort_values(["focus_id", "roi_candidate_score"], ascending=[True, False]).reset_index(drop=True)
        rois.insert(0, "roi_id", np.arange(1, len(rois) + 1))
        rois["roi_size_px"] = float(roi_size_px)
        rois["roi_size_um"] = float(roi_size_um)
        rois["mpp"] = float(mpp)
    return (rois, tum, foci) if return_foci else (rois, tum)


def export_foci_geojson(foci: pd.DataFrame, out_path: Path) -> None:
    features = []
    for r in foci.itertuples():
        props = {
            "focus_id": int(r.focus_id),
            "cluster_id": int(r.focus_id),
            "focus_area_px2": float(r.focus_area_px2),
            "focus_area_um2": float(r.focus_area_um2),
            "focus_perimeter_um": float(r.focus_perimeter_um),
            "centroid_x": float(r.centroid_x),
            "centroid_y": float(r.centroid_y),
            "n_tumor_tiles": int(r.n_tumor_tiles),
            "mean_tumor_fraction": float(r.mean_tumor_fraction),
        }
        features.append({"type": "Feature", "properties": props, "geometry": mapping(r.geometry)})
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def make_pseudo_thumbnail(df: pd.DataFrame) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    step_x, step_y = infer_tile_step(df)
    x0, y0 = float(df["wx"].min()), float(df["wy"].min())
    x1, y1 = float(df["wx"].max() + step_x), float(df["wy"].max() + step_y)
    cols = int(math.ceil((x1 - x0) / step_x))
    rows = int(math.ceil((y1 - y0) / step_y))
    img = np.ones((rows, cols, 3), dtype=float)
    vals = df[FRACTION_COLS].to_numpy(float)
    dom = vals.argmax(axis=1)
    for i, r in enumerate(df.itertuples()):
        c = int(round((float(r.wx) - x0) / step_x))
        rr = int(round((float(r.wy) - y0) / step_y))
        cls = CLASS_ORDER[int(dom[i])]
        frac = float(vals[i, dom[i]])
        alpha = min(1.0, max(0.25, frac))
        if 0 <= rr < rows and 0 <= c < cols:
            img[rr, c] = alpha * CLASS_COLORS[cls] + (1.0 - alpha)
    return img, (x0, y0, x1, y1)


def _draw_focus_outline(ax, geom: BaseGeometry, color: str, lw: float = 1.6) -> None:
    for poly in iter_polygon_parts(geom):
        coords = np.asarray(poly.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, fill=False, edgecolor=color, linewidth=lw))


def draw_pseudo_legend(ax) -> None:
    """Restore semantic-class legend and explain focus/ROI outlines."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=CLASS_COLORS[name], edgecolor="none", label=name)
        for name in CLASS_ORDER
    ]
    handles.extend([
        Line2D([0], [0], color="#111111", lw=1.6, label="Tumour focus outline"),
        Line2D([0], [0], color="#111111", lw=1.4, label="Representative tumour ROI"),
    ])
    ax.legend(handles=handles, loc="upper right", fontsize=8,
              framealpha=0.9, title="Segmentation / ROI")


def draw_wsi_legend(ax) -> None:
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#111111", lw=1.6, label="Tumour focus outline"),
        Line2D([0], [0], color="#111111", lw=1.4, label="Representative tumour ROI"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


def overlay_on_pseudo_thumbnail(
    df: pd.DataFrame,
    rois: pd.DataFrame,
    foci: pd.DataFrame,
    out_png: Path,
    roi_size_um: float,
    color_by_cluster: bool = True,
    label_boxes: bool | str = "auto",
) -> None:
    img, (x0, y0, x1, y1) = make_pseudo_thumbnail(df)
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(img, extent=[x0, x1, y1, y0], interpolation="nearest")

    for r in foci.itertuples():
        color = FOCUS_COLORS[(int(r.focus_id) - 1) % len(FOCUS_COLORS)] if color_by_cluster else "#111111"
        _draw_focus_outline(ax, r.geometry, color)
        rp = r.geometry.representative_point()
        ax.text(rp.x, rp.y, f"F{int(r.focus_id)}", fontsize=8, weight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

    show_labels = len(rois) <= 40 if label_boxes == "auto" else bool(label_boxes)
    for r in rois.itertuples():
        color = FOCUS_COLORS[(int(r.focus_id) - 1) % len(FOCUS_COLORS)] if color_by_cluster else "#000000"
        ax.add_patch(Rectangle(
            (r.x_min, r.y_min), r.x_max - r.x_min, r.y_max - r.y_min,
            fill=False, edgecolor=color, linewidth=1.4,
        ))
        if show_labels:
            ax.text(r.center_x, r.center_y, f"T{int(r.roi_id)}", fontsize=7,
                    ha="center", va="center", weight="bold",
                    bbox=dict(facecolor="white", alpha=0.55, edgecolor="none", pad=0.3))

    draw_pseudo_legend(ax)
    ax.set_title(f"Tumour foci and representative tumour ROIs — {len(rois)} ROI(s), {len(foci)} focus/foci, {roi_size_um:.0f} µm")
    ax.set_xlabel("WSI x coordinate (px)")
    ax.set_ylabel("WSI y coordinate (px)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def overlay_on_wsi_thumbnail(
    wsi_path: Path,
    rois: pd.DataFrame,
    foci: pd.DataFrame,
    out_png: Path,
    thumb_width: int,
    roi_size_um: float,
    color_by_cluster: bool = True,
) -> Optional[float]:
    try:
        import openslide
    except ImportError as e:
        raise RuntimeError("openslide-python is required for WSI overlay.") from e

    slide = openslide.OpenSlide(str(wsi_path))
    w, h = slide.dimensions
    th = int(round(thumb_width * h / w))
    thumb = slide.get_thumbnail((thumb_width, th)).convert("RGB")
    sx, sy = thumb_width / w, th / h

    fig, ax = plt.subplots(figsize=(13, max(4, 13 * th / thumb_width)))
    ax.imshow(thumb)
    for r in foci.itertuples():
        color = FOCUS_COLORS[(int(r.focus_id) - 1) % len(FOCUS_COLORS)] if color_by_cluster else "#111111"
        for poly in iter_polygon_parts(r.geometry):
            xy = np.asarray(poly.exterior.coords, dtype=float)
            xy[:, 0] *= sx; xy[:, 1] *= sy
            ax.add_patch(MplPolygon(xy, closed=True, fill=False, edgecolor=color, linewidth=1.2))
    for r in rois.itertuples():
        color = FOCUS_COLORS[(int(r.focus_id) - 1) % len(FOCUS_COLORS)] if color_by_cluster else "#000000"
        ax.add_patch(Rectangle((r.x_min * sx, r.y_min * sy),
                               (r.x_max-r.x_min)*sx, (r.y_max-r.y_min)*sy,
                               fill=False, edgecolor=color, linewidth=1.2))
    draw_wsi_legend(ax)
    ax.axis("off")
    ax.set_title(f"Tumour foci + representative tumour ROIs ({roi_size_um:.0f} µm)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    try:
        return float(slide.properties.get("openslide.mpp-x"))
    except (TypeError, ValueError):
        return None


def run_tumor_roi_overlay(wsi_path: str, cfg: PipelineConfig = None) -> dict:
    """Detect tumour foci and generate representative square tumour ROIs.

    Important ROI config fields
    ---------------------------
    ROI_SIZE_UM
        Square image-sampling ROI edge length in microns.
    ROI_MIN_TUMOR_FRAC
        Minimum tile-level tumour fraction used to construct tumour foci.
    ROI_FOCUS_REPAIR_GAP_UM
        Small physical closing distance for segmentation-gap repair only.
    ROI_MIN_FOCUS_AREA_UM2
        Minimum physical tumour-focus area; raises this to suppress tiny foci.
    ROI_MIN_ROI_TUMOR_FRAC
        Minimum tumour fraction required inside an accepted square ROI.
    ROI_MAX_NECROSIS
        Maximum mean necrosis fraction allowed inside an accepted ROI.
    ROI_TUMOR_SIGNAL_WEIGHT
        Weight on the candidate centre tile's tumour probability/fraction.
    ROI_NEIGHBOR_TUMOR_WEIGHT
        Weight on local neighbouring tumour continuity.
    ROI_NECROSIS_PENALTY
        Penalty applied to candidate centres with necrosis.
    ROI_MAX_ROIS_PER_FOCUS
        Representative-sampling cap per focus.  Default 3; 0 means unlimited.

    Presentation settings are intentionally not PipelineConfig fields.
    """
    if cfg is None:
        cfg = default_cfg

    slide_name = Path(wsi_path).stem
    manifest_csv = Path(cfg.OUT_DIR) / slide_name / "segmentation" / "manifest.csv"
    out_dir = Path(cfg.OUT_DIR) / slide_name / "spatial_feature_results" / "tumor_roi_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_csv}\n"
            f"Run run_segmentation(wsi_path, cfg) first."
        )

    mpp = float(getattr(cfg, "MPP", 0.25))
    roi_size_um = float(getattr(cfg, "ROI_SIZE_UM", 200.0))
    min_tumor_frac = float(getattr(cfg, "ROI_MIN_TUMOR_FRAC", 0.20))
    min_roi_tumor_frac = float(getattr(cfg, "ROI_MIN_ROI_TUMOR_FRAC", min_tumor_frac))
    max_necrosis = float(getattr(cfg, "ROI_MAX_NECROSIS", 0.50))

    # Legacy fallbacks are read only to avoid breaking existing configs.
    legacy_min_tiles = int(getattr(cfg, "ROI_MIN_CLUSTER_TILES", 3))
    legacy_merge = float(getattr(cfg, "ROI_MERGE_GAP_UM", 0.0))
    focus_repair_gap_um = float(getattr(
        cfg, "ROI_FOCUS_REPAIR_GAP_UM",
        min(legacy_merge, 50.0) if legacy_merge > 0 else 25.0,
    ))

    tumor_signal_weight = float(getattr(cfg, "ROI_TUMOR_SIGNAL_WEIGHT", 1.0))
    neighbor_weight = float(getattr(cfg, "ROI_NEIGHBOR_TUMOR_WEIGHT", 0.5))
    necrosis_penalty = float(getattr(cfg, "ROI_NECROSIS_PENALTY", 0.5))
    max_rois_per_focus = int(getattr(cfg, "ROI_MAX_ROIS_PER_FOCUS", 3))

    df = pd.read_csv(manifest_csv)
    validate_manifest(df)
    step_x, step_y = infer_tile_step(df)
    default_min_area = legacy_min_tiles * step_x * step_y * mpp * mpp
    min_focus_area_um2 = float(getattr(cfg, "ROI_MIN_FOCUS_AREA_UM2", default_min_area))

    print(f"\n{'='*60}")
    print("  Tumour focus + representative ROI generation")
    print(f"  Slide                   : {slide_name}")
    print(f"  ROI size                : {roi_size_um:.0f} µm")
    print(f"  Min tumour tile frac    : {min_tumor_frac:.3f}")
    print(f"  Focus repair gap        : {focus_repair_gap_um:.1f} µm")
    print(f"  Min focus area          : {min_focus_area_um2:.0f} µm²")
    print(f"  Min final ROI tumour    : {min_roi_tumor_frac:.3f}")
    print(f"  Max ROIs / focus        : {max_rois_per_focus if max_rois_per_focus > 0 else 'unlimited'}")
    print(
        "  Candidate score weights : "
        f"tumour={tumor_signal_weight:g}, "
        f"neighbour={neighbor_weight:g}, "
        f"necrosis_penalty={necrosis_penalty:g}"
    )
    print(f"  Output                  : {out_dir}")
    print(f"{'='*60}")

    # Single source of truth: the public runner delegates all focus/ROI
    # generation to generate_rois().
    rois, tum, foci = generate_rois(
        df=df,
        min_tumor_frac=min_tumor_frac,
        roi_size_um=roi_size_um,
        mpp=mpp,
        max_necrosis=max_necrosis,
        min_cluster_tiles=legacy_min_tiles,
        merge_gap_um=legacy_merge,
        focus_repair_gap_um=focus_repair_gap_um,
        min_focus_area_um2=min_focus_area_um2,
        min_roi_tumor_frac=min_roi_tumor_frac,
        tumor_signal_weight=tumor_signal_weight,
        neighbor_weight=neighbor_weight,
        necrosis_penalty=necrosis_penalty,
        max_rois_per_focus=max_rois_per_focus,
        return_foci=True,
    )

    roi_csv = out_dir / "tumor_roi_boxes.csv"
    foci_geojson = out_dir / "tumor_foci.geojson"
    focus_tiles_csv = out_dir / "tumor_focus_tiles.csv"
    pseudo_png = out_dir / "tumor_roi_boxes_pseudo_thumbnail.png"
    wsi_png: Optional[str] = None

    # cluster_id is intentionally retained everywhere for the TSR scorer and
    # for backward compatibility with existing downstream tables.
    if not rois.empty and "cluster_id" not in rois.columns:
        rois["cluster_id"] = rois["focus_id"]
    if not tum.empty and "cluster_id" not in tum.columns:
        tum["cluster_id"] = tum["focus_id"]
    if not foci.empty and "cluster_id" not in foci.columns:
        foci["cluster_id"] = foci["focus_id"]

    rois.to_csv(roi_csv, index=False)
    export_foci_geojson(foci, foci_geojson)
    tum.to_csv(focus_tiles_csv, index=False)

    n_candidate_tiles = int((df["frac_Tumour"] >= min_tumor_frac).sum())
    if n_candidate_tiles == 0:
        print(
            "  WARNING: no tumour-positive tiles passed ROI_MIN_TUMOR_FRAC. "
            "Lower ROI_MIN_TUMOR_FRAC only if this is inconsistent with the segmentation."
        )
    elif foci.empty:
        print(
            "  WARNING: tumour-positive tiles were found, but every focus was removed. "
            "Check ROI_MIN_FOCUS_AREA_UM2 and ROI_FOCUS_REPAIR_GAP_UM."
        )
    elif rois.empty:
        print(
            "  WARNING: tumour foci were detected but no representative ROI passed QC. "
            "Check ROI_MIN_ROI_TUMOR_FRAC, ROI_MAX_NECROSIS, and ROI_SIZE_UM."
        )

    overlay_on_pseudo_thumbnail(
        df, rois, foci, pseudo_png, roi_size_um,
        color_by_cluster=_COLOR_BY_FOCUS,
    )

    if Path(wsi_path).exists():
        wsi_png_path = out_dir / "tumor_roi_boxes_wsi_thumbnail.png"
        try:
            slide_mpp = overlay_on_wsi_thumbnail(
                Path(wsi_path), rois, foci, wsi_png_path,
                _THUMBNAIL_WIDTH_PX, roi_size_um,
                color_by_cluster=_COLOR_BY_FOCUS,
            )
            wsi_png = str(wsi_png_path)
            if slide_mpp and abs(slide_mpp - mpp) / max(mpp, 1e-12) > 0.10:
                print(
                    "  WARNING: WSI metadata MPP differs from cfg.MPP by >10% "
                    f"(slide={slide_mpp:.4f}, cfg={mpp:.4f})."
                )
        except Exception as exc:
            print(f"  WARNING: WSI thumbnail overlay skipped: {exc}")
    else:
        print("  WARNING: WSI file not found on disk; skipped WSI thumbnail overlay.")

    print(f"  Tumour foci : {len(foci)}")
    print(f"  Tumour ROIs : {len(rois)}")
    print(f"  Wrote       : {roi_csv}")
    print(f"  Wrote       : {foci_geojson}")

    return {
        "slide_name": slide_name,
        "manifest_csv": str(manifest_csv),
        "roi_csv": str(roi_csv),
        "foci_geojson": str(foci_geojson),
        "focus_tiles_csv": str(focus_tiles_csv),
        "pseudo_png": str(pseudo_png),
        "wsi_png": wsi_png,
        "n_boxes": int(len(rois)),
        "n_foci": int(len(foci)),
        "n_clusters": int(len(foci)),  # compatibility
    }


def main(argv=None) -> None:
    from .config import config_from_args
    cfg, _ = config_from_args(argv)
    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit("--wsi-path is required")
    run_tumor_roi_overlay(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
