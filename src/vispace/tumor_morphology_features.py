#!/usr/bin/env python3
"""
tumor_morphology_features.py
============================
Stage 8 of the ViSpace pipeline — tumour morphology feature extraction.

Reads cluster polygons produced by run_cluster_tils_tsr_score() (stage 5)
and the segmentation GeoJSON produced by run_stitching() (stage 3).

Output directory
----------------
    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_morphology/
        tumor_core_features_by_cluster.csv
        tumor_core_wsi_summary.csv
        tumor_island_qc.csv   (only when cfg.MORPHOLOGY_SAVE_ISLAND_QC = True)

Pipeline position
-----------------
    tessellate.py → segmenter.py → stitch.py → tumor_roi_overlay.py
        → cluster_tils_tsr_score.py → immune_proximity_features.py
        → necrosis_proximity_features.py → tumor_morphology_features.py

Usage (as a library)
---------------------
    from vispace import run_tumor_morphology_features
    from vispace import cfg

    run_tumor_morphology_features("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
------------------------------
Same shared flags as the rest of the pipeline — every PipelineConfig field
is available here too, including the MORPHOLOGY_* knobs. Also supports
--from-json to pick up a config saved earlier via `config.py --print-config`.

    # minimal — requires cluster_tils_tsr_score.py to have already run
    python tumor_morphology_features.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python tumor_morphology_features.py --from-json run_config.json

    # continue from a saved config but override one knob
    python tumor_morphology_features.py --from-json run_config.json \\
        --morphology-min-island-area-um2 500

    # save per-island QC output alongside the standard outputs
    python tumor_morphology_features.py --from-json run_config.json \\
        --morphology-save-island-qc true

    # see every available flag
    python tumor_morphology_features.py --help
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import shape, Polygon as SPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from .config import cfg as default_cfg, PipelineConfig


# ---------------------------------------------------------------------------
# Constants  (unchanged)
# ---------------------------------------------------------------------------

DEFAULT_TUMOR_CLASS_NAMES      = {"Tumour", "Tumor", "tumour", "tumor"}
RECOMMENDED_MIN_ISLAND_AREA_UM2 = 1000.0


# ---------------------------------------------------------------------------
# Geometry helpers  (unchanged)
# ---------------------------------------------------------------------------

def fix_geom(geom: BaseGeometry) -> BaseGeometry:
    if geom is None or geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def iter_polygon_parts(geom: BaseGeometry) -> Iterable[BaseGeometry]:
    if geom is None or geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for g in geom.geoms:
            yield from iter_polygon_parts(g)


def safe_div(num: float, denom: float, default: float = np.nan) -> float:
    try:
        if denom is None or (isinstance(denom, float) and np.isnan(denom)) or denom == 0:
            return default
        return float(num) / float(denom)
    except Exception:
        return default


def px_to_um(px: float, mpp: float) -> float:
    return float(px) * mpp


def px2_to_um2(px2: float, mpp: float) -> float:
    return float(px2) * mpp * mpp


def px2_to_mm2(px2: float, mpp: float) -> float:
    return px2_to_um2(px2, mpp) / 1e6


def get_class_name(props: dict) -> Optional[str]:
    if not isinstance(props, dict):
        return None
    cls = props.get("class") or props.get("name") or props.get("label")
    if cls is not None:
        return str(cls).strip()
    classification = props.get("classification")
    if isinstance(classification, dict):
        cls = classification.get("name") or classification.get("label")
        if cls is not None:
            return str(cls).strip()
    return None


# ---------------------------------------------------------------------------
# Per-island shape metrics  (unchanged)
# ---------------------------------------------------------------------------

def island_compactness(poly: BaseGeometry) -> float:
    P = float(poly.length)
    return float(4.0 * math.pi * poly.area / (P * P)) if P > 0 else np.nan


def island_solidity(poly: BaseGeometry) -> float:
    hull_area = poly.convex_hull.area
    return safe_div(poly.area, hull_area) if hull_area > 0 else np.nan


def island_elongation(poly: BaseGeometry) -> float:
    rect = poly.minimum_rotated_rectangle
    if rect.is_empty or rect.geom_type != "Polygon":
        return np.nan
    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return np.nan
    edges = sorted([
        math.hypot(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1])
        for i in range(4)
    ])
    minor = float(np.mean(edges[:2]))
    major = float(np.mean(edges[2:]))
    return safe_div(major, minor, default=np.nan)


def island_hole_area_px2(poly: BaseGeometry) -> float:
    if poly.geom_type != "Polygon":
        return 0.0
    return float(sum(SPolygon(ring).area for ring in poly.interiors))


def aggregate_island_shape_metrics(
    islands: List[BaseGeometry], mpp: float
) -> Dict[str, float]:
    if not islands:
        return {k: np.nan for k in [
            "tumor_compactness_mean", "tumor_solidity_mean",
            "tumor_elongation_mean",  "tumor_hole_fraction",
            "tumor_major_axis_um_mean", "tumor_minor_axis_um_mean",
        ]}

    areas   = np.array([p.area for p in islands], dtype=float)
    total_A = float(areas.sum())
    weights = areas / total_A if total_A > 0 else np.ones(len(islands)) / len(islands)

    comps  = np.array([island_compactness(p) for p in islands])
    solds  = np.array([island_solidity(p)    for p in islands])
    elongs = np.array([island_elongation(p)  for p in islands])
    holes  = np.array([island_hole_area_px2(p) for p in islands])

    majors, minors = [], []
    for p in islands:
        rect = p.minimum_rotated_rectangle
        if rect.is_empty or rect.geom_type != "Polygon":
            majors.append(np.nan); minors.append(np.nan); continue
        coords = list(rect.exterior.coords)
        edges  = sorted([math.hypot(coords[i+1][0]-coords[i][0],
                                    coords[i+1][1]-coords[i][1]) for i in range(4)])
        minors.append(float(np.mean(edges[:2])))
        majors.append(float(np.mean(edges[2:])))
    majors = np.array(majors)
    minors = np.array(minors)

    def wavg(vals: np.ndarray) -> float:
        mask = np.isfinite(vals)
        return float(np.average(vals[mask], weights=weights[mask])) if mask.any() else np.nan

    total_hole  = float(holes.sum())
    total_gross = total_A + total_hole

    return {
        "tumor_compactness_mean":   wavg(comps),
        "tumor_solidity_mean":      wavg(solds),
        "tumor_elongation_mean":    wavg(elongs),
        "tumor_hole_fraction":      safe_div(total_hole, total_gross),
        "tumor_major_axis_um_mean": wavg(majors) * mpp if np.any(np.isfinite(majors)) else np.nan,
        "tumor_minor_axis_um_mean": wavg(minors) * mpp if np.any(np.isfinite(minors)) else np.nan,
    }


# ---------------------------------------------------------------------------
# Fragmentation + spread + NND  (unchanged)
# ---------------------------------------------------------------------------

def compute_fragmentation(
    islands:      List[BaseGeometry],
    cluster_geom: BaseGeometry,
    mpp:          float,
) -> Dict[str, float]:
    n = len(islands)
    cluster_area_mm2 = px2_to_mm2(cluster_geom.area, mpp)

    _nan_keys = [
        "tumor_n_islands", "tumor_largest_patch_index", "tumor_fragmentation_index",
        "tumor_effective_patches", "tumor_patch_density_per_mm2", "tumor_island_area_cv",
        "tumor_island_area_mean_um2", "tumor_island_area_median_um2",
        "tumor_island_area_p90_um2", "tumor_island_area_max_um2",
        "tumor_island_nnd_mean_um", "tumor_island_nnd_median_um", "tumor_island_nnd_std_um",
        "tumor_spread_frac", "tumor_convex_hull_fill",
    ]
    out = {k: (0 if k == "tumor_n_islands" else np.nan) for k in _nan_keys}
    if n == 0:
        return out

    areas_px2 = np.array([g.area for g in islands], dtype=float)
    areas_um2 = areas_px2 * mpp * mpp
    total_px2 = float(areas_px2.sum())
    pts = np.array([[g.representative_point().x, g.representative_point().y]
                    for g in islands], dtype=float)

    lpi     = safe_div(float(areas_px2.max()), total_px2)
    p       = areas_px2 / total_px2 if total_px2 > 0 else np.ones(n) / n
    eff     = float(1.0 / np.sum(p**2)) if np.sum(p**2) > 0 else np.nan
    area_cv = float(np.std(areas_px2, ddof=1) / np.mean(areas_px2)) if n > 1 else 0.0

    out["tumor_n_islands"]              = int(n)
    out["tumor_largest_patch_index"]    = lpi
    out["tumor_fragmentation_index"]    = (1.0 - lpi) if pd.notna(lpi) else np.nan
    out["tumor_effective_patches"]      = eff
    out["tumor_patch_density_per_mm2"]  = safe_div(n, cluster_area_mm2)
    out["tumor_island_area_cv"]         = area_cv
    out["tumor_island_area_mean_um2"]   = float(np.mean(areas_um2))
    out["tumor_island_area_median_um2"] = float(np.median(areas_um2))
    out["tumor_island_area_p90_um2"]    = float(np.percentile(areas_um2, 90))
    out["tumor_island_area_max_um2"]    = float(np.max(areas_um2))

    if n > 1:
        try:
            from scipy.spatial import cKDTree
            kd = cKDTree(pts)
            dist, _ = kd.query(pts, k=2)
            nnd_um  = dist[:, 1] * mpp
            out["tumor_island_nnd_mean_um"]   = float(np.mean(nnd_um))
            out["tumor_island_nnd_median_um"] = float(np.median(nnd_um))
            out["tumor_island_nnd_std_um"]    = float(np.std(nnd_um))
        except Exception:
            pass
        x_ext = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_ext = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
    else:
        x_ext = y_ext = 0.0

    minx, miny, maxx, maxy = cluster_geom.bounds
    cw = max(maxx - minx, 1e-12)
    ch = max(maxy - miny, 1e-12)
    out["tumor_spread_frac"] = 0.5 * (x_ext / cw + y_ext / ch)

    if n >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull_area = float(ConvexHull(pts).volume)
            out["tumor_convex_hull_fill"] = safe_div(total_px2, hull_area)
        except Exception:
            pass

    return out


def size_stratified_counts(
    islands: List[BaseGeometry], mpp: float
) -> Dict[str, int]:
    if not islands:
        return {
            "tumor_n_islands_fragment": 0,
            "tumor_n_islands_micro":    0,
            "tumor_n_islands_small":    0,
            "tumor_n_islands_large":    0,
        }
    areas_um2 = np.array([g.area * mpp * mpp for g in islands])
    return {
        "tumor_n_islands_fragment": int((areas_um2 <  1_000).sum()),
        "tumor_n_islands_micro":    int(((areas_um2 >=  1_000) & (areas_um2 <  10_000)).sum()),
        "tumor_n_islands_small":    int(((areas_um2 >= 10_000) & (areas_um2 < 100_000)).sum()),
        "tumor_n_islands_large":    int((areas_um2 >= 100_000).sum()),
    }


# ---------------------------------------------------------------------------
# Per-cluster feature computation  (unchanged)
# ---------------------------------------------------------------------------

def compute_cluster_features(
    cluster_id,
    cluster_geom:        BaseGeometry,
    tumor_intersections: List[BaseGeometry],
    mpp:                 float,
    min_island_area_px2: float,
) -> Tuple[Dict, List[Dict]]:
    cl_area_px2 = float(cluster_geom.area)
    cl_perim_px = float(cluster_geom.length)
    base = {
        "cluster_id":           cluster_id,
        "cluster_area_px2":     cl_area_px2,
        "cluster_area_um2":     px2_to_um2(cl_area_px2, mpp),
        "cluster_area_mm2":     px2_to_mm2(cl_area_px2, mpp),
        "cluster_perimeter_px": cl_perim_px,
        "cluster_perimeter_um": px_to_um(cl_perim_px, mpp),
    }
    _empty_shape = {
        "tumor_compactness_mean": np.nan, "tumor_solidity_mean": np.nan,
        "tumor_elongation_mean":  np.nan, "tumor_hole_fraction": np.nan,
        "tumor_major_axis_um_mean": np.nan, "tumor_minor_axis_um_mean": np.nan,
    }

    if not tumor_intersections:
        row = {**base,
               "tumor_area_px2": 0.0, "tumor_area_um2": 0.0, "tumor_area_mm2": 0.0,
               "tumor_perimeter_px": 0.0, "tumor_perimeter_um": 0.0,
               "tumor_fraction_of_cluster": 0.0,
               "tumor_boundary_per_area_um_per_mm2": np.nan,
               **_empty_shape,
               "tumor_valid_morphology": False,
               "tumor_invalid_reason": "no_tumor_intersection"}
        row.update(compute_fragmentation([], cluster_geom, mpp))
        row.update(size_stratified_counts([], mpp))
        return row, []

    dissolved = fix_geom(unary_union(tumor_intersections))
    if dissolved is None or dissolved.is_empty:
        row = {**base,
               "tumor_area_px2": 0.0, "tumor_area_um2": 0.0, "tumor_area_mm2": 0.0,
               "tumor_perimeter_px": 0.0, "tumor_perimeter_um": 0.0,
               "tumor_fraction_of_cluster": 0.0,
               "tumor_boundary_per_area_um_per_mm2": np.nan,
               **_empty_shape,
               "tumor_valid_morphology": False,
               "tumor_invalid_reason": "dissolved_empty"}
        row.update(compute_fragmentation([], cluster_geom, mpp))
        row.update(size_stratified_counts([], mpp))
        return row, []

    tumor_area_px2 = float(dissolved.area)
    tumor_perim_px = float(dissolved.length)
    tumor_area_mm2 = px2_to_mm2(tumor_area_px2, mpp)
    tumor_perim_um = px_to_um(tumor_perim_px, mpp)

    all_islands = list(iter_polygon_parts(dissolved))
    islands     = [p for p in all_islands if p.area >= min_island_area_px2]

    shape_metrics = aggregate_island_shape_metrics(islands, mpp)
    frag          = compute_fragmentation(islands, cluster_geom, mpp)
    strata        = size_stratified_counts(all_islands, mpp)

    row = {
        **base,
        "tumor_area_px2":                     tumor_area_px2,
        "tumor_area_um2":                     px2_to_um2(tumor_area_px2, mpp),
        "tumor_area_mm2":                     tumor_area_mm2,
        "tumor_perimeter_px":                 tumor_perim_px,
        "tumor_perimeter_um":                 tumor_perim_um,
        "tumor_fraction_of_cluster":          safe_div(tumor_area_px2, cl_area_px2),
        "tumor_boundary_per_area_um_per_mm2": safe_div(tumor_perim_um, tumor_area_mm2),
        **shape_metrics,
        **frag,
        **strata,
        "min_island_area_um2_used":           min_island_area_px2 * mpp * mpp,
        "tumor_valid_morphology":             True,
        "tumor_invalid_reason":               "",
    }

    island_rows = []
    for iid, island in enumerate(islands, start=1):
        rp = island.representative_point()
        island_rows.append({
            "cluster_id":                     cluster_id,
            "tumor_island_id":                iid,
            "tumor_island_area_px2":          float(island.area),
            "tumor_island_area_um2":          px2_to_um2(island.area, mpp),
            "tumor_island_area_mm2":          px2_to_mm2(island.area, mpp),
            "tumor_island_perimeter_px":      float(island.length),
            "tumor_island_perimeter_um":      px_to_um(island.length, mpp),
            "tumor_island_representative_x":  float(rp.x),
            "tumor_island_representative_y":  float(rp.y),
            "tumor_island_compactness":       island_compactness(island),
            "tumor_island_solidity":          island_solidity(island),
            "tumor_island_elongation":        island_elongation(island),
            "tumor_island_fraction_of_tumor": safe_div(island.area, tumor_area_px2),
        })

    return row, island_rows


# ---------------------------------------------------------------------------
# Loading  (unchanged)
# ---------------------------------------------------------------------------

def load_cluster_polygons(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        geom  = fix_geom(shape(feat["geometry"]))
        if geom is None or geom.is_empty:
            continue
        if "cluster_id" not in props:
            raise ValueError("Cluster GeoJSON missing required property 'cluster_id'.")
        rows.append({
            "cluster_id":   props["cluster_id"],
            "cluster_geom": geom,
            **{k: v for k, v in props.items() if k != "cluster_id"},
        })
    if not rows:
        raise RuntimeError(f"No cluster polygons loaded from {path}")
    return pd.DataFrame(rows)


def load_tumor_regions(path: Path, tumor_names: set) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows, n_total = [], 0
    for idx, feat in enumerate(data.get("features", [])):
        n_total += 1
        props = feat.get("properties", {}) or {}
        cls   = get_class_name(props)
        if cls not in tumor_names:
            continue
        geom = fix_geom(shape(feat["geometry"]))
        if geom is None or geom.is_empty:
            continue
        rows.append({"source_id": idx, "geometry": geom})
    if not rows:
        raise RuntimeError(
            f"No tumour polygons found in {path}. "
            f"Tried class names: {sorted(tumor_names)}"
        )
    df = pd.DataFrame(rows)
    print(f"  Loaded {len(df):,} tumour polygons from {n_total:,} GeoJSON features.")
    return df


# ---------------------------------------------------------------------------
# Main extraction  (unchanged)
# ---------------------------------------------------------------------------

def query_tree(tree: STRtree, geoms: list, query_geom: BaseGeometry) -> List[int]:
    hits = tree.query(query_geom)
    if len(hits) == 0:
        return []
    if isinstance(hits[0], (int, np.integer)):
        return [int(i) for i in hits]
    id_map = {id(g): i for i, g in enumerate(geoms)}
    return [id_map[id(g)] for g in hits]


def extract_all_clusters(
    cluster_polygons:    pd.DataFrame,
    tumor_regions:       pd.DataFrame,
    mpp:                 float,
    min_island_area_um2: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tumor_geoms    = tumor_regions["geometry"].tolist()
    tree           = STRtree(tumor_geoms)
    min_island_px2 = min_island_area_um2 / (mpp * mpp) if min_island_area_um2 > 0 else 0.0

    feature_rows, island_rows = [], []

    for _, c in tqdm(
        cluster_polygons.iterrows(), total=len(cluster_polygons),
        desc="Computing tumour morphology", unit="cluster",
    ):
        cid      = c["cluster_id"]
        cgeom    = c["cluster_geom"]
        cand_idx = query_tree(tree, tumor_geoms, cgeom)
        intersected = []
        for i in cand_idx:
            g = tumor_geoms[i]
            if not cgeom.intersects(g):
                continue
            inter = cgeom.intersection(g)
            for part in iter_polygon_parts(inter):
                if part.area > 0:
                    intersected.append(part)

        row, irows = compute_cluster_features(cid, cgeom, intersected, mpp, min_island_px2)
        for meta in ["n_roi_boxes", "buffer_um", "priority_nonoverlap_assignment"]:
            if meta in c.index:
                row[meta] = c[meta]

        feature_rows.append(row)
        island_rows.extend(irows)

    return pd.DataFrame(feature_rows), pd.DataFrame(island_rows)


# ---------------------------------------------------------------------------
# WSI summary  (unchanged)
# ---------------------------------------------------------------------------

CORE_FEATURE_COLUMNS = [
    "cluster_id", "n_roi_boxes", "cluster_area_mm2", "cluster_perimeter_um",
    "tumor_area_px2", "tumor_area_um2", "tumor_area_mm2",
    "tumor_perimeter_px", "tumor_perimeter_um",
    "tumor_fraction_of_cluster",
    "tumor_boundary_per_area_um_per_mm2",
    "tumor_compactness_mean", "tumor_solidity_mean", "tumor_elongation_mean",
    "tumor_major_axis_um_mean", "tumor_minor_axis_um_mean",
    "tumor_hole_fraction",
    "tumor_n_islands", "tumor_n_islands_fragment", "tumor_n_islands_micro",
    "tumor_n_islands_small", "tumor_n_islands_large",
    "tumor_largest_patch_index", "tumor_fragmentation_index",
    "tumor_effective_patches", "tumor_patch_density_per_mm2",
    "tumor_island_area_cv", "tumor_island_area_mean_um2",
    "tumor_island_area_median_um2", "tumor_island_area_p90_um2", "tumor_island_area_max_um2",
    "tumor_island_nnd_mean_um", "tumor_island_nnd_median_um",
    "tumor_spread_frac", "tumor_convex_hull_fill",
    "min_island_area_um2_used", "tumor_valid_morphology", "tumor_invalid_reason",
]


def select_core_features(features: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CORE_FEATURE_COLUMNS if c in features.columns]
    return features[cols].copy()


def compute_wsi_summary(features: pd.DataFrame, islands: pd.DataFrame) -> pd.DataFrame:
    total_cl_px2      = float(features["cluster_area_px2"].sum())
    total_tu_px2      = float(features["tumor_area_px2"].sum())
    total_tu_perim_um = float(features["tumor_perimeter_um"].sum())
    total_tu_mm2      = float(features["tumor_area_mm2"].sum())
    total_cl_mm2      = float(features["cluster_area_mm2"].sum())

    out: dict = {
        "n_clusters":                            int(len(features)),
        "n_clusters_with_tumor":                 int((features["tumor_area_px2"] > 0).sum()),
        "wsi_cluster_area_mm2":                  total_cl_mm2,
        "wsi_tumor_area_mm2":                    total_tu_mm2,
        "wsi_tumor_fraction_of_cluster":         safe_div(total_tu_px2, total_cl_px2),
        "wsi_tumor_perimeter_um":                total_tu_perim_um,
        "wsi_tumor_boundary_per_area_um_per_mm2": safe_div(total_tu_perim_um, total_tu_mm2),
        "wsi_tumor_n_islands":                   int(features["tumor_n_islands"].fillna(0).sum()),
        "wsi_tumor_patch_density_per_mm2":       safe_div(
            float(features["tumor_n_islands"].fillna(0).sum()), total_cl_mm2),
    }

    for tier in ["fragment", "micro", "small", "large"]:
        col = f"tumor_n_islands_{tier}"
        if col in features.columns:
            out[f"wsi_{col}"] = int(features[col].fillna(0).sum())

    weights = features["tumor_area_mm2"].to_numpy(float)
    valid_w = weights > 0
    for col in [
        "tumor_compactness_mean", "tumor_solidity_mean", "tumor_elongation_mean",
        "tumor_hole_fraction", "tumor_largest_patch_index", "tumor_fragmentation_index",
        "tumor_effective_patches", "tumor_island_nnd_median_um", "tumor_spread_frac",
        "tumor_convex_hull_fill",
    ]:
        if col in features.columns and valid_w.any():
            vals = features[col].to_numpy(float)
            mask = valid_w & np.isfinite(vals)
            out[f"wsi_area_weighted_{col}"] = (
                float(np.average(vals[mask], weights=weights[mask])) if mask.any() else np.nan
            )

    if islands is not None and not islands.empty:
        a = islands["tumor_island_area_um2"].to_numpy(float)
        out.update({
            "wsi_island_area_mean_um2":   float(np.mean(a)),
            "wsi_island_area_median_um2": float(np.median(a)),
            "wsi_island_area_p90_um2":    float(np.percentile(a, 90)),
            "wsi_island_area_max_um2":    float(np.max(a)),
        })

    return pd.DataFrame([out])




# ---------------------------------------------------------------------------
# User-facing overall tumour morphology analysis PNG
# ---------------------------------------------------------------------------

def fmt_metric(v: float, decimals: int = 2, suffix: str = "") -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.{decimals}f}{suffix}"
    except Exception:
        return "—"


def fmt_percent(v: float, decimals: int = 1) -> str:
    try:
        if pd.isna(v):
            return "—"
        return f"{float(v):.{decimals}f}%"
    except Exception:
        return "—"


def row_get(row, key: str, default=np.nan):
    try:
        return row.get(key, default)
    except Exception:
        return default


def classify_overall_tumor_morphology(wsi_row: pd.Series, cluster_df: pd.DataFrame) -> Tuple[str, str]:
    """
    Convert segmentation-derived morphology numbers into a short, transparent
    user-facing interpretation. This is descriptive only, not diagnostic.
    """
    tumor_fraction = row_get(wsi_row, "wsi_tumor_fraction_of_cluster")
    fragmentation = row_get(wsi_row, "wsi_area_weighted_tumor_fragmentation_index")
    n_islands      = row_get(wsi_row, "wsi_tumor_n_islands")
    solidity       = row_get(wsi_row, "wsi_area_weighted_tumor_solidity_mean")
    spread         = row_get(wsi_row, "wsi_area_weighted_tumor_spread_frac")

    if pd.isna(tumor_fraction) or tumor_fraction <= 0:
        return (
            "Overall pattern: no measurable tumour morphology",
            "No tumour-positive morphology features were detected in the analysed cluster regions."
        )

    descriptors = []

    if pd.notna(fragmentation) and fragmentation >= 0.65:
        descriptors.append("fragmented")
    elif pd.notna(fragmentation) and fragmentation <= 0.40:
        descriptors.append("cohesive")
    else:
        descriptors.append("moderately fragmented")

    if pd.notna(spread) and spread >= 0.75:
        descriptors.append("spatially dispersed")
    elif pd.notna(spread) and spread <= 0.35:
        descriptors.append("spatially localized")

    if pd.notna(solidity) and solidity >= 0.70:
        descriptors.append("solid")
    elif pd.notna(solidity) and solidity < 0.60:
        descriptors.append("irregular")

    if pd.notna(tumor_fraction) and tumor_fraction >= 0.30:
        descriptors.append("tumour-rich")
    elif pd.notna(tumor_fraction) and tumor_fraction < 0.10:
        descriptors.append("low tumour fraction")

    headline = "Overall pattern: " + ", ".join(descriptors) + " tumour architecture"

    dominant_sentence = ""
    if not cluster_df.empty and "tumor_area_mm2" in cluster_df.columns:
        positive = cluster_df[cluster_df["tumor_area_mm2"].fillna(0) > 0].copy()
        if not positive.empty:
            dom = positive.sort_values("tumor_area_mm2", ascending=False).iloc[0]
            total_area = float(positive["tumor_area_mm2"].sum())
            dom_area = row_get(dom, "tumor_area_mm2")
            area_share = 100.0 * float(dom_area) / total_area if total_area > 0 else np.nan
            dominant_sentence = (
                f"Cluster {int(dom['cluster_id'])} is the dominant tumour-bearing region, "
                f"contributing {fmt_percent(area_share)} of tumour area, with "
                f"{fmt_metric(row_get(dom, 'tumor_n_islands'), 0)} tumour islands and "
                f"fragmentation index {fmt_metric(row_get(dom, 'tumor_fragmentation_index'))}."
            )

    explanation = (
        f"The slide contains {fmt_metric(n_islands, 0)} tumour islands across "
        f"{fmt_metric(row_get(wsi_row, 'n_clusters_with_tumor'), 0)} tumour-positive clusters. "
        f"The WSI fragmentation index is {fmt_metric(fragmentation)}, with solidity "
        f"{fmt_metric(solidity)} and spread fraction {fmt_metric(spread)}. "
        f"{dominant_sentence}"
    )

    return headline, explanation


def make_tumor_morphology_interpretation_bullets(wsi_row: pd.Series, cluster_df: pd.DataFrame) -> str:
    frag = row_get(wsi_row, "wsi_area_weighted_tumor_fragmentation_index")
    tumor_fraction = row_get(wsi_row, "wsi_tumor_fraction_of_cluster")
    n_islands = row_get(wsi_row, "wsi_tumor_n_islands")
    solidity = row_get(wsi_row, "wsi_area_weighted_tumor_solidity_mean")
    spread = row_get(wsi_row, "wsi_area_weighted_tumor_spread_frac")

    bullets = []

    if pd.notna(frag) and frag >= 0.65:
        bullets.append("• High fragmentation: tumour is distributed across many separated islands.")
    elif pd.notna(frag) and frag <= 0.40:
        bullets.append("• Low fragmentation: tumour architecture appears comparatively cohesive.")
    else:
        bullets.append("• Intermediate fragmentation: tumour shows partial separation into islands.")

    if pd.notna(tumor_fraction) and tumor_fraction >= 0.30:
        bullets.append("• Tumour-rich analysed region: tumour occupies a substantial fraction of cluster area.")
    elif pd.notna(tumor_fraction) and tumor_fraction < 0.10:
        bullets.append("• Low tumour fraction: tumour occupies a small fraction of cluster area.")
    else:
        bullets.append("• Moderate tumour fraction within analysed tumour clusters.")

    if pd.notna(spread) and spread >= 0.75:
        bullets.append("• High spread fraction: tumour islands are spatially dispersed across the cluster envelope.")
    else:
        bullets.append("• Lower spread fraction: tumour islands are more spatially localized.")

    if pd.notna(solidity) and solidity < 0.60:
        bullets.append("• Lower solidity suggests irregular or less filled tumour island shapes.")
    else:
        bullets.append("• Solidity is moderate to high, suggesting more filled tumour island shapes.")

    if not cluster_df.empty and "tumor_area_mm2" in cluster_df.columns:
        positive = cluster_df[cluster_df["tumor_area_mm2"].fillna(0) > 0].copy()
        if not positive.empty:
            dom = positive.sort_values("tumor_area_mm2", ascending=False).iloc[0]
            total_area = float(positive["tumor_area_mm2"].sum())
            share = 100.0 * float(dom["tumor_area_mm2"]) / total_area if total_area > 0 else np.nan
            bullets.append(f"• Cluster {int(dom['cluster_id'])} dominates tumour burden with {fmt_percent(share)} of tumour area.")

    bullets.append(f"• Total tumour islands detected: {fmt_metric(n_islands, 0)}.")
    return "\n".join(bullets)


def draw_metric_card(ax, title: str, value: str, subtitle: str = "") -> None:
    ax.axis("off")
    card = plt.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#d9dee7",
        linewidth=1.2,
    )
    ax.add_patch(card)
    ax.text(0.06, 0.72, title, transform=ax.transAxes, fontsize=10,
            color="#5b6678", weight="bold", va="center")
    ax.text(0.06, 0.42, value, transform=ax.transAxes, fontsize=18,
            color="#111827", weight="bold", va="center")
    if subtitle:
        ax.text(0.06, 0.18, subtitle, transform=ax.transAxes, fontsize=8.5,
                color="#6b7280", va="center")


def add_bar_labels(ax, values, decimals: int = 2) -> None:
    if len(values) == 0:
        return
    ymax = max(float(np.nanmax(values)), 1e-12)
    offset = ymax * 0.03
    for i, v in enumerate(values):
        if pd.isna(v):
            continue
        ax.text(i, float(v) + offset, f"{float(v):.{decimals}f}",
                ha="center", va="bottom", fontsize=8)


def generate_tumor_morphology_analysis_png(
    core_features: pd.DataFrame,
    wsi_summary: pd.DataFrame,
    out_png: Path,
    slide_name: Optional[str] = None,
) -> Optional[Path]:
    """
    Generate one user-facing PNG panel summarizing overall tumour morphology.

    The PNG is intended for reports and dashboards. It is produced directly
    by this morphology stage so the Quarto report can simply display it later.
    """
    if core_features is None or core_features.empty:
        print("  Skipped morphology analysis PNG: no cluster-level morphology features.")
        return None
    if wsi_summary is None or wsi_summary.empty:
        print("  Skipped morphology analysis PNG: no WSI morphology summary.")
        return None

    required = [
        "cluster_id", "tumor_area_mm2", "tumor_fraction_of_cluster",
        "tumor_n_islands", "tumor_fragmentation_index",
    ]
    missing = [c for c in required if c not in core_features.columns]
    if missing:
        print(f"  Skipped morphology analysis PNG: missing columns {missing}")
        return None

    # Local import keeps the extraction stage usable in minimal environments.
    global plt
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plot_df = core_features.copy()
    plot_df = plot_df[plot_df["tumor_area_mm2"].fillna(0) > 0].copy()
    if plot_df.empty:
        print("  Skipped morphology analysis PNG: no tumour-positive clusters.")
        return None

    plot_df = plot_df.sort_values("tumor_area_mm2", ascending=False)
    plot_df["cluster_label"] = plot_df["cluster_id"].apply(lambda x: f"C{int(x)}")

    wsi_row = wsi_summary.iloc[0]
    headline, explanation = classify_overall_tumor_morphology(wsi_row, plot_df)
    bullets = make_tumor_morphology_interpretation_bullets(wsi_row, plot_df)

    fig = plt.figure(figsize=(18, 12), dpi=180)
    fig.patch.set_facecolor("white")

    gs = GridSpec(
        5, 4,
        figure=fig,
        height_ratios=[0.9, 1.05, 1.05, 1.75, 1.15],
        hspace=0.68,
        wspace=0.38,
    )

    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")
    title = "Tumour Morphology Overall Analysis"
    if slide_name:
        title += f" · {slide_name}"

    ax_header.text(0, 0.92, title, fontsize=24, weight="bold", color="#111827", va="top")
    ax_header.text(0, 0.54, headline, fontsize=16, weight="bold", color="#374151", va="top")
    ax_header.text(0, 0.04, textwrap.fill(explanation, width=165), fontsize=10.5,
                   color="#4b5563", va="bottom")

    metrics = [
        ("Tumour area", f"{fmt_metric(row_get(wsi_row, 'wsi_tumor_area_mm2'))} mm²", "Total segmented tumour area"),
        ("Tumour fraction", fmt_percent(row_get(wsi_row, "wsi_tumor_fraction_of_cluster") * 100), "Fraction of analysed cluster area"),
        ("Tumour islands", fmt_metric(row_get(wsi_row, "wsi_tumor_n_islands"), 0), "Separated tumour components"),
        ("Fragmentation index", fmt_metric(row_get(wsi_row, "wsi_area_weighted_tumor_fragmentation_index")), "Higher means more fragmented"),
        ("Solidity", fmt_metric(row_get(wsi_row, "wsi_area_weighted_tumor_solidity_mean")), "Higher means more filled/solid"),
        ("Spread fraction", fmt_metric(row_get(wsi_row, "wsi_area_weighted_tumor_spread_frac")), "Spatial distribution of islands"),
        ("Patch density", f"{fmt_metric(row_get(wsi_row, 'wsi_tumor_patch_density_per_mm2'))}/mm²", "Tumour patches per area"),
        ("Median island distance", f"{fmt_metric(row_get(wsi_row, 'wsi_area_weighted_tumor_island_nnd_median_um'))} µm", "Nearest-neighbour distance"),
    ]

    for i, metric in enumerate(metrics):
        r = 1 + i // 4
        c = i % 4
        draw_metric_card(fig.add_subplot(gs[r, c]), metric[0], metric[1], metric[2])

    # Cluster bar charts
    area_vals = plot_df["tumor_area_mm2"].astype(float).to_numpy()
    frac_vals = plot_df["tumor_fraction_of_cluster"].astype(float).to_numpy() * 100.0
    island_vals = plot_df["tumor_n_islands"].astype(float).to_numpy()
    frag_vals = plot_df["tumor_fragmentation_index"].astype(float).to_numpy()
    labels = plot_df["cluster_label"].tolist()

    chart_specs = [
        ("Tumour area by cluster", "Area (mm²)", area_vals, 2),
        ("Tumour fraction by cluster", "Fraction (%)", frac_vals, 1),
        ("Tumour islands by cluster", "Island count", island_vals, 0),
        ("Fragmentation by cluster", "Fragmentation index", frag_vals, 2),
    ]

    for idx, (title, ylabel, vals, decimals) in enumerate(chart_specs):
        ax = fig.add_subplot(gs[3, idx])
        ax.bar(labels, vals)
        add_bar_labels(ax, vals, decimals=decimals)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        if "Fragmentation" in title:
            ax.set_ylim(0, max(1.0, float(np.nanmax(vals)) * 1.2))

    # Bubble plot: islands vs fragmentation, size by tumour area.
    ax_bubble = fig.add_subplot(gs[4, 0:2])
    sizes = plot_df["tumor_area_mm2"].astype(float).to_numpy()
    size_scaled = 250 + 1800 * sizes / max(float(np.nanmax(sizes)), 1e-12)
    ax_bubble.scatter(
        plot_df["tumor_n_islands"],
        plot_df["tumor_fragmentation_index"],
        s=size_scaled,
        alpha=0.65,
        edgecolors="#111827",
        linewidths=0.8,
    )
    for _, row in plot_df.iterrows():
        ax_bubble.text(
            row["tumor_n_islands"],
            row["tumor_fragmentation_index"],
            f"C{int(row['cluster_id'])}",
            fontsize=9,
            ha="center",
            va="center",
            weight="bold",
        )
    ax_bubble.set_title("Fragmentation landscape", fontsize=12, weight="bold")
    ax_bubble.set_xlabel("Number of tumour islands")
    ax_bubble.set_ylabel("Fragmentation index")
    ax_bubble.spines[["top", "right"]].set_visible(False)
    ax_bubble.grid(alpha=0.25)

    ax_note = fig.add_subplot(gs[4, 2:])
    ax_note.axis("off")
    ax_note.text(0, 0.98, "Interpretation notes", fontsize=13, weight="bold",
                 color="#111827", va="top")
    ax_note.text(0, 0.78, bullets, fontsize=10.5, color="#374151", va="top", linespacing=1.5)
    ax_note.text(
        0, 0.05,
        "Note: Computational morphology summary from segmentation-derived features; not a standalone diagnosis.",
        fontsize=8.5,
        color="#6b7280",
        va="bottom",
    )

    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")
    return out_png


def save_tumor_morphology_plots(
    core_features: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, Optional[str]]:
    """
    Save separate tumour morphology visualization PNGs.

    These are intentionally plot-only outputs. Any text interpretation,
    cards, or explanatory report layout can be added later in the QMD.

    Writes, when the required columns are available:
        tumor_area_by_cluster.png
        tumor_fraction_by_cluster.png
        tumor_islands_by_cluster.png
        tumor_fragmentation_by_cluster.png
        tumor_fragmentation_landscape.png
    """
    plot_paths: Dict[str, Optional[str]] = {
        "tumor_area_by_cluster_png": None,
        "tumor_fraction_by_cluster_png": None,
        "tumor_islands_by_cluster_png": None,
        "tumor_fragmentation_by_cluster_png": None,
        "tumor_fragmentation_landscape_png": None,
    }

    if core_features is None or core_features.empty:
        print("  Skipped morphology plots: no cluster-level morphology features.")
        return plot_paths

    required = ["cluster_id", "tumor_area_mm2"]
    missing = [c for c in required if c not in core_features.columns]
    if missing:
        print(f"  Skipped morphology plots: missing columns {missing}")
        return plot_paths

    # Local import keeps the extraction stage usable in minimal environments.
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_df = core_features.copy()
    plot_df = plot_df[plot_df["tumor_area_mm2"].fillna(0) > 0].copy()

    if plot_df.empty:
        print("  Skipped morphology plots: no tumour-positive clusters.")
        return plot_paths

    plot_df = plot_df.sort_values("tumor_area_mm2", ascending=False)
    plot_df["cluster_label"] = plot_df["cluster_id"].apply(lambda x: f"C{int(x)}")

    def _save_bar(column: str, ylabel: str, title: str, filename: str, decimals: int = 2, scale: float = 1.0):
        if column not in plot_df.columns:
            print(f"  Skipped {filename}: missing column {column}")
            return None

        vals = plot_df[column].astype(float).to_numpy() * scale
        labels = plot_df["cluster_label"].tolist()

        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.bar(labels, vals)
        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Tumour cluster")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

        ymax = max(float(np.nanmax(vals)), 1e-12)
        offset = ymax * 0.03
        for i, v in enumerate(vals):
            if pd.isna(v):
                continue
            ax.text(i, float(v) + offset, f"{float(v):.{decimals}f}", ha="center", va="bottom", fontsize=9)

        if "Fragmentation" in title:
            ax.set_ylim(0, max(1.0, ymax * 1.2))
        else:
            ax.set_ylim(0, ymax * 1.18)

        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Wrote: {out_path.name}")
        return str(out_path)

    plot_paths["tumor_area_by_cluster_png"] = _save_bar(
        column="tumor_area_mm2",
        ylabel="Tumour area (mm²)",
        title="Tumour area by cluster",
        filename="tumor_area_by_cluster.png",
        decimals=2,
    )

    plot_paths["tumor_fraction_by_cluster_png"] = _save_bar(
        column="tumor_fraction_of_cluster",
        ylabel="Tumour fraction (%)",
        title="Tumour fraction by cluster",
        filename="tumor_fraction_by_cluster.png",
        decimals=1,
        scale=100.0,
    )

    plot_paths["tumor_islands_by_cluster_png"] = _save_bar(
        column="tumor_n_islands",
        ylabel="Tumour island count",
        title="Tumour islands by cluster",
        filename="tumor_islands_by_cluster.png",
        decimals=0,
    )

    plot_paths["tumor_fragmentation_by_cluster_png"] = _save_bar(
        column="tumor_fragmentation_index",
        ylabel="Fragmentation index",
        title="Fragmentation by cluster",
        filename="tumor_fragmentation_by_cluster.png",
        decimals=2,
    )

    landscape_required = ["tumor_n_islands", "tumor_fragmentation_index", "tumor_area_mm2"]
    missing_landscape = [c for c in landscape_required if c not in plot_df.columns]
    if missing_landscape:
        print(f"  Skipped tumor_fragmentation_landscape.png: missing columns {missing_landscape}")
        return plot_paths

    fig, ax = plt.subplots(figsize=(8.5, 5.6), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    sizes = plot_df["tumor_area_mm2"].astype(float).to_numpy()
    size_scaled = 220 + 1500 * sizes / max(float(np.nanmax(sizes)), 1e-12)

    ax.scatter(
        plot_df["tumor_n_islands"],
        plot_df["tumor_fragmentation_index"],
        s=size_scaled,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.8,
    )

    for _, row in plot_df.iterrows():
        ax.text(
            row["tumor_n_islands"],
            row["tumor_fragmentation_index"],
            f"C{int(row['cluster_id'])}",
            fontsize=9,
            ha="center",
            va="center",
            weight="bold",
        )

    ax.set_title("Fragmentation landscape", fontsize=14, weight="bold")
    ax.set_xlabel("Number of tumour islands")
    ax.set_ylabel("Fragmentation index")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "tumor_fragmentation_landscape.png"
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plot_paths["tumor_fragmentation_landscape_png"] = str(out_path)
    print(f"  Wrote: {out_path.name}")

    return plot_paths


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------

def run_tumor_morphology_features(
    wsi_path: str,
    cfg:      PipelineConfig = None,
) -> dict:
    """
    Run tumour morphology feature extraction for one WSI.

    Reads
    -----
    cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
        cluster_scoring_polygons.geojson
    cfg.OUT_DIR/<slide>/segmentation/
        segmentation_all_classes.geojson

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_morphology/
        tumor_core_features_by_cluster.csv
        tumor_core_wsi_summary.csv
        tumor_island_qc.csv   (only when cfg.MORPHOLOGY_SAVE_ISLAND_QC = True)
        tumor_area_by_cluster.png
        tumor_fraction_by_cluster.png
        tumor_islands_by_cluster.png
        tumor_fragmentation_by_cluster.png
        tumor_fragmentation_landscape.png

    Parameters
    ----------
    wsi_path : str
    cfg      : PipelineConfig
        Knobs (all with fallbacks):
            cfg.MPP
            cfg.MORPHOLOGY_MIN_ISLAND_AREA_UM2  (default 1000.0)
            cfg.MORPHOLOGY_SAVE_ISLAND_QC       (default False)
            cfg.MORPHOLOGY_TUMOR_CLASS_NAMES    (default {"Tumour","Tumor","tumour","tumor"})

    Returns
    -------
    dict : slide_name, cluster_geojson, segmentation_geojson,
           core_csv, summary_csv, island_qc_csv (or None), plot PNG paths
    """
    if cfg is None:
        cfg = default_cfg

    slide_name = Path(wsi_path).stem

    cluster_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "cluster_tils_tsr_score"
        / "cluster_scoring_polygons.geojson"
    )
    seg_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    out_dir = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "tumor_morphology"
    )

    if not cluster_geojson.exists():
        raise FileNotFoundError(
            f"Cluster GeoJSON not found: {cluster_geojson}\n"
            f"Run run_cluster_tils_tsr_score(wsi_path, cfg) first."
        )
    if not seg_geojson.exists():
        raise FileNotFoundError(
            f"Segmentation GeoJSON not found: {seg_geojson}\n"
            f"Run run_stitching(wsi_path, cfg) first."
        )

    mpp               = getattr(cfg, "MPP",                          0.25)
    min_island_area   = getattr(cfg, "MORPHOLOGY_MIN_ISLAND_AREA_UM2", RECOMMENDED_MIN_ISLAND_AREA_UM2)
    save_island_qc    = getattr(cfg, "MORPHOLOGY_SAVE_ISLAND_QC",      False)
    tumor_class_names = getattr(cfg, "MORPHOLOGY_TUMOR_CLASS_NAMES",   DEFAULT_TUMOR_CLASS_NAMES)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Tumour morphology features")
    print(f"  Slide              : {slide_name}")
    print(f"  min_island_area_um2: {min_island_area:.0f}")
    print(f"  Clusters           : {cluster_geojson.name}")
    print(f"  Output             : {out_dir}")
    print(f"{'='*55}")

    clusters = load_cluster_polygons(cluster_geojson)
    tumors   = load_tumor_regions(seg_geojson, tumor_class_names)

    features, islands = extract_all_clusters(clusters, tumors, mpp, min_island_area)
    core    = select_core_features(features)
    summary = compute_wsi_summary(features, islands)

    core_csv    = out_dir / "tumor_core_features_by_cluster.csv"
    summary_csv = out_dir / "tumor_core_wsi_summary.csv"
    core.to_csv(core_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"  Wrote: {core_csv.name}")
    print(f"  Wrote: {summary_csv.name}")

    morphology_plots = save_tumor_morphology_plots(
        core_features=core,
        out_dir=out_dir,
    )

    island_qc_csv = None
    if save_island_qc and not islands.empty:
        island_qc_csv = out_dir / "tumor_island_qc.csv"
        islands.to_csv(island_qc_csv, index=False)
        print(f"  Wrote: {island_qc_csv.name}")

    valid = (
        core[core["tumor_valid_morphology"]]
        if "tumor_valid_morphology" in core.columns else core
    )
    print(f"\n  === Tumour morphology summary ===")
    print(f"  Clusters scored:         {len(core)}")
    if "tumor_area_mm2" in core.columns:
        print(f"  Total tumour area:       {core['tumor_area_mm2'].sum():.3f} mm²")
    if "tumor_n_islands" in core.columns:
        print(f"  Total islands (filtered):{int(core['tumor_n_islands'].fillna(0).sum())}")
    if "tumor_fragmentation_index" in valid.columns:
        print(f"  Mean fragmentation index:{valid['tumor_fragmentation_index'].mean():.3f}")
    if "tumor_compactness_mean" in valid.columns:
        print(f"  Mean compactness:        {valid['tumor_compactness_mean'].mean():.3f}")
    print(f"\n  Done.")

    return {
        "slide_name":           slide_name,
        "cluster_geojson":      str(cluster_geojson),
        "segmentation_geojson": str(seg_geojson),
        "core_csv":             str(core_csv),
        "summary_csv":          str(summary_csv),
        "island_qc_csv":        str(island_qc_csv) if island_qc_csv else None,
        **morphology_plots,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) — every PipelineConfig
# field (including the MORPHOLOGY_* knobs) is available as a flag, plus
# --from-json to pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python tumor_morphology_features.py --from-json run_config.json

def main(argv=None) -> None:
    from .config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    slide_name = Path(cfg.WSI_PATH).stem
    cluster_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "cluster_tils_tsr_score" / "cluster_scoring_polygons.geojson"
    )
    seg_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    if not cluster_geojson.exists():
        raise SystemExit(
            f"Cluster GeoJSON not found: {cluster_geojson}. "
            f"Run cluster_tils_tsr_score.py for this slide (with the same --out-dir) first."
        )
    if not seg_geojson.exists():
        raise SystemExit(
            f"Segmentation GeoJSON not found: {seg_geojson}. "
            f"Run stitch.py for this slide (with the same --out-dir) first."
        )

    run_tumor_morphology_features(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()