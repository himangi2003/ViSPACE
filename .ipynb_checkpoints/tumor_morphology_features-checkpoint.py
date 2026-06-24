#!/usr/bin/env python3
"""
tumor_morphology_features.py
============================
Spatial-analysis stage — tumour morphology feature extraction.

Reads cluster polygons produced by run_cluster_tils_tsr_score() and the
segmentation GeoJSON produced by run_stitching().

Output directory
----------------
    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_morphology/
        tumor_core_features_by_cluster.csv
        tumor_core_wsi_summary.csv
        tumor_island_qc.csv   (only when cfg.MORPHOLOGY_SAVE_ISLAND_QC = True)

Usage
-----
    from tumor_morphology_features import run_tumor_morphology_features
    from config import cfg

    run_tumor_morphology_features("slides/TCGA-A1-A0SP.svs", cfg)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import shape, Polygon as SPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from config import cfg as default_cfg, PipelineConfig


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
           core_csv, summary_csv, island_qc_csv (or None)
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
    }