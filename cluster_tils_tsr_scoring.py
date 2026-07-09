#!/usr/bin/env python3
"""
cluster_tils_tsr_score.py
=========================
Step 6 of the ViP-SegD pipeline (Step 2 of the spatial-analysis stage).

Builds non-overlapping cluster scoring polygons from the ROI boxes produced
by run_tumor_roi_overlay(), then computes TSR and sTILs from the segmentation
GeoJSON produced by run_stitching().

Background handling
-------------------
The cluster scoring polygon is the dissolved + buffered hull of tumor ROI boxes.
It may contain unsegmented pixels (background = no GeoJSON annotation).
Background is NEVER included in any TSR or sTILs denominator:

  TSR   = stroma_area / (tumor_area + stroma_area)          [only T+S pixels]
  sTILs = inflammatory_area / stroma_area × 100             [Salgado 2015]

Background only appears as:
  background_area_px2 / _um2  — QC / tissue coverage reporting
  tissue_fraction             — fraction of cluster polygon that is segmented
  reliability flags           — clusters with low tissue_fraction are flagged

sTILs variants (all three always computed):
  sTILs_salgado   = inflam / stroma           (recommended default)
  sTILs_stromal   = inflam / (stroma + inflam)
  sTILs_tissue    = inflam / viable_tissue    (viable = T+S+I+O, excl. necrosis)

Inputs (all inferred from wsi_path + cfg)
-----------------------------------------
  roi_boxes_csv : cfg.OUT_DIR/<slide>/spatial_feature_results/
                      tumor_roi_overlay/tumor_roi_boxes.csv
                  Written by run_tumor_roi_overlay() (step 5).
  geojson       : cfg.OUT_DIR/<slide>/segmentation/
                      segmentation_all_classes.geojson
                  Written by run_stitching() (step 4).

Outputs
-------
  cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
      cluster_scoring_polygons.geojson
      tils_tsr_by_cluster.csv
      tils_tsr_wsi_summary.csv
      cluster_tils_tsr_overlay.png

Pipeline position
-----------------
    tessellate.py  →  segmenter.py  →  stitch.py  →  tumor_roi_overlay.py  →  cluster_tils_tsr_score.py

Usage (as a library)
---------------------
    from cluster_tils_tsr_score import run_cluster_tils_tsr_score
    from config import cfg

    run_cluster_tils_tsr_score("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
------------------------------
Same shared flags as config.py / tessellate.py / segmenter.py / stitch.py /
tumor_roi_overlay.py — every PipelineConfig field is available here too,
including the CLUSTER_* / TILS_DENOMINATOR knobs. Also supports --from-json
to pick up a config saved earlier via `config.py --print-config`.

    # minimal — requires tumor_roi_overlay.py to have already run for this slide
    python cluster_tils_tsr_score.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python cluster_tils_tsr_score.py --from-json run_config.json

    # continue from a saved config but override the sTILs denominator
    python cluster_tils_tsr_score.py --from-json run_config.json \\
        --tils-denominator tissue

    # see every available flag
    python cluster_tils_tsr_score.py --help
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from tqdm import tqdm

from shapely.geometry import shape, box as shapely_box, mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

from config import cfg as default_cfg, PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level constants  (unchanged from original)
# ---------------------------------------------------------------------------

CLASS_ORDER = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]

CLASS_COLORS_RGB = {
    "Tumour":       np.array([220,  30,  30], dtype=float) / 255.0,
    "Stroma":       np.array([ 30, 180,  30], dtype=float) / 255.0,
    "Inflammatory": np.array([ 30, 100, 255], dtype=float) / 255.0,
    "Necrosis":     np.array([255, 165,   0], dtype=float) / 255.0,
    "Others":       np.array([200,   0, 200], dtype=float) / 255.0,
}

# Draw order: Tumour last → drawn on top of all other classes
DRAW_ORDER = ["Others", "Necrosis", "Stroma", "Inflammatory", "Tumour"]

TILS_LEVELS = [
    (0.0,   5.0,   "very low"),
    (5.0,  10.0,   "low"),
    (10.0, 30.0,   "intermediate"),
    (30.0, 101.0,  "high"),
]
TILS_LEVEL_COLORS = {
    "very low":     "#2c7bb6",
    "low":          "#abd9e9",
    "intermediate": "#fdae61",
    "high":         "#d7191c",
    "unreliable":   "#888888",
}


# ---------------------------------------------------------------------------
# Internal ScoringConfig  (not exposed to callers of run_cluster_tils_tsr_score)
# ---------------------------------------------------------------------------

@dataclass
class _ScoringConfig:
    roi_boxes:  Path
    geojson:    Path
    outdir:     Path    = field(default_factory=lambda: Path("cluster_tils_tsr_output"))
    tils_denominator:                str   = "salgado"
    cluster_buffer_um:               float = 200.0
    mpp:                             float = 0.25
    min_roi_boxes_per_cluster:       int   = 5
    min_polygon_area_px2:            float = 1.0
    max_area_quantile:               float = 1.0
    max_area_mad_z:                  float = 0.0
    min_tissue_fraction:             float = 0.10
    min_tsr_denom_area_px2:          float = 5000.0
    min_tils_denom_area_px2:         float = 5000.0
    no_overlay:                      bool  = False
    cross_class_overlap_warn_fraction: float = 0.05

    def validate(self) -> None:
        if self.mpp <= 0:
            raise ValueError("mpp must be > 0")
        valid = ("salgado", "stroma_plus_inflammatory", "tissue")
        if self.tils_denominator not in valid:
            raise ValueError(f"tils_denominator must be one of {valid}")
        if not self.roi_boxes.exists():
            raise FileNotFoundError(f"ROI boxes CSV not found: {self.roi_boxes}")
        if not self.geojson.exists():
            raise FileNotFoundError(f"GeoJSON not found: {self.geojson}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_class_name(feat: dict) -> Optional[str]:
    props = feat.get("properties") or {}
    cls   = (props.get("classification") or {}).get("name")
    if cls in CLASS_ORDER:
        return cls
    idx = props.get("class_index")
    if isinstance(idx, int) and 0 <= idx < len(CLASS_ORDER):
        return CLASS_ORDER[idx]
    return None


def make_valid(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom


def tils_level(value: float, reliable: bool) -> str:
    if not reliable or not np.isfinite(value) or value < 0:
        return "unreliable"
    for lo, hi, label in TILS_LEVELS:
        if lo <= value < hi:
            return label
    return "high"


def fmt_tsr(tumor_pct: float, stroma_pct: float) -> str:
    """Format TSR as 'T%/S%', e.g. '20/80'."""
    if not (np.isfinite(tumor_pct) and np.isfinite(stroma_pct)):
        return "NA"
    return f"{tumor_pct:.0f}/{stroma_pct:.0f}"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_regions(
    geojson_path: Path,
    min_polygon_area_px2: float = 1.0,
    max_area_quantile: float    = 1.0,
    max_area_mad_z: float       = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    with open(geojson_path) as f:
        data = json.load(f)

    rows, skipped_unknown, skipped_bad = [], 0, 0

    features = data.get("features", [])
    for feat in tqdm(features, desc="Loading GeoJSON regions", unit="feature", leave=False):
        cls = normalize_class_name(feat)
        if cls is None:
            skipped_unknown += 1
            continue
        try:
            geom = shape(feat.get("geometry"))
        except Exception:
            skipped_bad += 1
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty or geom.area <= 0:
            skipped_bad += 1
            continue
        rows.append({
            "class":             cls,
            "geometry":          geom,
            "geometry_area_px2": float(geom.area),
        })

    regions = pd.DataFrame(rows)
    before  = len(regions)
    if before:
        keep_parts = []
        for cls, sub in regions.groupby("class", sort=False):
            s    = sub["geometry_area_px2"].astype(float)
            keep = s >= min_polygon_area_px2
            if 0 < max_area_quantile < 1 and len(s) >= 20:
                keep &= s <= s.quantile(max_area_quantile)
            if max_area_mad_z > 0 and len(s) >= 20:
                log_s = np.log1p(s)
                med   = float(np.median(log_s))
                mad   = float(np.median(np.abs(log_s - med)))
                if mad > 0:
                    z     = 0.6745 * np.abs(log_s - med) / mad
                    keep &= z <= max_area_mad_z
            keep_parts.append(sub.loc[keep])
        regions = (
            pd.concat(keep_parts, ignore_index=True)
            if keep_parts else regions.iloc[0:0].copy()
        )

    after = len(regions)
    return regions.reset_index(drop=True), {
        "features_seen":         int(len(features)),
        "regions_loaded":        int(after),
        "regions_filtered_out":  int(before - after),
        "skipped_unknown_class": int(skipped_unknown),
        "skipped_bad_geometry":  int(skipped_bad),
    }


def load_roi_boxes(roi_boxes_csv: Path) -> pd.DataFrame:
    rois    = pd.read_csv(roi_boxes_csv)
    required = ["cluster_id", "x_min", "y_min", "x_max", "y_max"]
    missing  = [c for c in required if c not in rois.columns]
    if missing:
        raise ValueError(f"ROI box CSV missing columns: {missing}")
    if "roi_id" not in rois.columns:
        rois.insert(0, "roi_id", np.arange(1, len(rois) + 1))
    for c in required + ["roi_id"]:
        rois[c] = pd.to_numeric(rois[c], errors="raise")
    rois = rois.dropna(subset=required).copy()
    rois = rois[
        (rois["x_max"] > rois["x_min"]) & (rois["y_max"] > rois["y_min"])
    ].copy()
    rois["cluster_id"] = rois["cluster_id"].astype(int)
    rois["roi_id"]     = rois["roi_id"].astype(int)
    return rois


def export_cluster_polygons_geojson(
    clusters: pd.DataFrame, out_path: Path, mpp: float
) -> None:
    features = []
    for row in clusters.itertuples():
        features.append({
            "type": "Feature",
            "properties": {
                "cluster_id":                   int(row.cluster_id),
                "n_roi_boxes":                  int(row.n_roi_boxes),
                "buffer_um":                    float(row.buffer_px * mpp),
                "cluster_scoring_area_px2":     float(row.cluster_scoring_area_px2),
                "cluster_scoring_area_um2":     float(row.cluster_scoring_area_px2 * mpp * mpp),
                "cluster_scoring_perimeter_px": float(row.cluster_scoring_perimeter_px),
                "cluster_scoring_perimeter_um": float(row.cluster_scoring_perimeter_px * mpp),
                "priority_nonoverlap":          int(row.priority_nonoverlap_assignment),
            },
            "geometry": mapping(row.geometry),
        })
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


# ---------------------------------------------------------------------------
# Geometry: dissolve → buffer → non-overlap
# ---------------------------------------------------------------------------

def dissolve_buffer_filter_nonoverlap(
    rois: pd.DataFrame,
    buffer_px: float,
    min_roi_boxes_per_cluster: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_records = []

    for cid, sub in tqdm(
        rois.groupby("cluster_id", sort=True),
        desc="Dissolving ROI boxes into cluster polygons",
        unit="cluster",
    ):
        boxes = [
            shapely_box(float(r.x_min), float(r.y_min), float(r.x_max), float(r.y_max))
            for r in sub.itertuples()
        ]
        orig = make_valid(unary_union(boxes))
        exp  = make_valid(orig.buffer(buffer_px, join_style=1)) if buffer_px > 0 else orig
        raw_records.append({
            "cluster_id":                               int(cid),
            "n_roi_boxes":                              int(len(sub)),
            "roi_ids":                                  ";".join(map(str, sub["roi_id"].tolist())),
            "cluster_original_union_area_px2":          float(orig.area),
            "cluster_expanded_pre_nonoverlap_area_px2": float(exp.area),
            "buffer_px":                                float(buffer_px),
            "geometry_original":                        orig,
            "geometry_expanded":                        exp,
            "kept_after_min_roi_filter":                bool(len(sub) >= min_roi_boxes_per_cluster),
        })

    all_clusters = pd.DataFrame(raw_records)
    retained = (
        all_clusters[all_clusters["kept_after_min_roi_filter"]]
        .copy()
        .sort_values(
            ["cluster_original_union_area_px2", "cluster_id"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    assigned: list = []
    final_records  = []
    for priority, row in enumerate(retained.itertuples(), start=1):
        geom = row.geometry_expanded
        if assigned:
            geom = make_valid(geom.difference(unary_union(assigned)))
        if geom.is_empty or geom.area <= 0:
            log.warning(
                "Cluster %d swallowed by higher-priority clusters — skipped.",
                row.cluster_id,
            )
            continue
        assigned.append(geom)
        minx, miny, maxx, maxy = geom.bounds
        final_records.append({
            "cluster_id":                               int(row.cluster_id),
            "priority_nonoverlap_assignment":           int(priority),
            "n_roi_boxes":                              int(row.n_roi_boxes),
            "roi_ids":                                  row.roi_ids,
            "x_min": float(minx), "y_min": float(miny),
            "x_max": float(maxx), "y_max": float(maxy),
            "cluster_original_union_area_px2":          float(row.cluster_original_union_area_px2),
            "cluster_expanded_pre_nonoverlap_area_px2": float(row.cluster_expanded_pre_nonoverlap_area_px2),
            "cluster_scoring_area_px2":                 float(geom.area),
            "cluster_scoring_perimeter_px":             float(geom.length),
            "inter_cluster_overlap_removed_area_px2":   max(
                0.0, float(row.cluster_expanded_pre_nonoverlap_area_px2 - geom.area)
            ),
            "buffer_px":                                float(row.buffer_px),
            "geometry":                                 geom,
        })

    clusters = pd.DataFrame(final_records)

    # Sanity-check: measure residual pairwise overlap
    n_inter, max_inter = 0, 0.0
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a = float(
                clusters.iloc[i]["geometry"]
                .intersection(clusters.iloc[j]["geometry"]).area
            )
            if a > 1e-6:
                n_inter  += 1
                max_inter = max(max_inter, a)

    clusters.attrs["n_area_intersections"]           = int(n_inter)
    clusters.attrs["max_pairwise_intersection_area"] = float(max_inter)
    clusters.attrs["min_roi_boxes_per_cluster"]      = int(min_roi_boxes_per_cluster)

    return (
        clusters.sort_values("cluster_id").reset_index(drop=True),
        all_clusters.sort_values("cluster_id").reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Spatial index
# ---------------------------------------------------------------------------

def build_spatial_index(regions: pd.DataFrame):
    geoms      = regions["geometry"].tolist()
    tree       = STRtree(geoms)
    id_to_idx  = {id(g): i for i, g in enumerate(geoms)}
    return geoms, tree, id_to_idx


def query_indices(tree, id_to_idx, geom) -> Iterable[int]:
    for item in tree.query(geom):
        yield (
            int(item) if isinstance(item, (int, np.integer))
            else id_to_idx[id(item)]
        )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_clusters(
    clusters:    pd.DataFrame,
    regions:     pd.DataFrame,
    min_tissue_fraction:             float,
    min_tsr_denom_area_px2:          float,
    min_tils_denom_area_px2:         float,
    tils_denominator:                str,
    mpp:                             float,
    cross_class_overlap_warn_fraction: float = 0.05,
) -> pd.DataFrame:
    """
    Compute TSR and sTILs for each cluster polygon.

    Background handling
    -------------------
    cluster_scoring_area_px2  = full polygon area (includes unsegmented background).
    Segmentation only covers pixels labelled by the GeoJSON model.  Any pixel not
    covered by a GeoJSON polygon is background and contributes to background_area_px2.

    Background is NEVER part of any TSR or sTILs denominator:
      TSR denom   = tumor_area + stroma_area            (segmented pixels only)
      sTILs denom = stroma_area (Salgado) or stroma+inflam or viable_tissue
    Background enters only the reliability gate via tissue_fraction.
    """
    geoms, tree, id_to_idx = build_spatial_index(regions)
    classes = regions["class"].to_numpy()
    records = []

    for c in tqdm(
        clusters.itertuples(), total=len(clusters),
        desc="Scoring clusters (TSR / sTILs)", unit="cluster",
    ):
        cluster_geom = c.geometry
        area_px  = {cls: 0.0 for cls in CLASS_ORDER}
        perim_px = {cls: 0.0 for cls in CLASS_ORDER}

        for i in query_indices(tree, id_to_idx, cluster_geom):
            g = geoms[i]
            if not cluster_geom.intersects(g):
                continue
            inter = make_valid(cluster_geom.intersection(g))
            if inter.is_empty:
                continue
            a = float(inter.area)
            if a <= 0:
                continue
            cls = classes[i]
            area_px[cls]  += a
            perim_px[cls] += float(inter.length)

        tumor    = area_px["Tumour"]
        stroma   = area_px["Stroma"]
        inflam   = area_px["Inflammatory"]
        necrosis = area_px["Necrosis"]
        others   = area_px["Others"]

        # ── Background ──────────────────────────────────────────────────────
        cluster_area_px = float(c.cluster_scoring_area_px2)
        segmented_area  = tumor + stroma + inflam + necrosis + others

        cross_class_overlap = max(0.0, segmented_area - cluster_area_px)
        if (
            cluster_area_px > 0
            and cross_class_overlap / cluster_area_px > cross_class_overlap_warn_fraction
        ):
            warnings.warn(
                f"Cluster {c.cluster_id}: cross-class GeoJSON overlap is "
                f"{cross_class_overlap / cluster_area_px:.1%} of cluster area. "
                "Tissue areas may be slightly inflated.",
                UserWarning, stacklevel=2,
            )

        background_area      = max(0.0, cluster_area_px - segmented_area)
        raw_tissue_fraction  = segmented_area / cluster_area_px if cluster_area_px > 0 else 0.0
        tissue_fraction      = min(raw_tissue_fraction, 1.0)
        cross_class_overlap_frac = max(0.0, raw_tissue_fraction - 1.0)

        viable_tissue = tumor + stroma + inflam + others

        # ── TSR ─────────────────────────────────────────────────────────────
        tsr_denom    = tumor + stroma
        tsr_raw      = stroma / tsr_denom if tsr_denom > 0 else np.nan
        tsr_reliable = bool(
            tsr_denom > 0
            and tsr_denom >= min_tsr_denom_area_px2
            and tissue_fraction >= min_tissue_fraction
        )
        t_pct = tumor  / tsr_denom * 100 if tsr_denom > 0 else np.nan
        s_pct = stroma / tsr_denom * 100 if tsr_denom > 0 else np.nan
        tsr_display_str = fmt_tsr(t_pct, s_pct)

        # ── sTILs (all three variants) ───────────────────────────────────────
        salgado_tils = inflam / stroma              * 100 if stroma > 0              else np.nan
        stromal_tils = inflam / (stroma + inflam)   * 100 if (stroma + inflam) > 0  else np.nan
        tissue_tils  = inflam / viable_tissue        * 100 if viable_tissue > 0      else np.nan

        if tils_denominator == "salgado":
            tils_raw, tils_denom_area = salgado_tils, stroma
        elif tils_denominator == "stroma_plus_inflammatory":
            tils_raw, tils_denom_area = stromal_tils, stroma + inflam
        else:  # "tissue"
            tils_raw, tils_denom_area = tissue_tils, viable_tissue

        tils_reliable = bool(
            tils_denom_area > 0
            and tils_denom_area >= min_tils_denom_area_px2
            and tissue_fraction >= min_tissue_fraction
        )

        m = mpp
        rec = {
            # identifiers
            "cluster_id":                       int(c.cluster_id),
            "n_roi_boxes":                      int(c.n_roi_boxes),
            "priority_nonoverlap":              int(c.priority_nonoverlap_assignment),
            # cluster polygon geometry
            "cluster_scoring_area_px2":         cluster_area_px,
            "cluster_scoring_area_um2":         cluster_area_px * m * m,
            "cluster_scoring_perimeter_px":     float(c.cluster_scoring_perimeter_px),
            "cluster_scoring_perimeter_um":     float(c.cluster_scoring_perimeter_px) * m,
            "buffer_um":                        float(c.buffer_px) * m,
            # tissue class areas
            "tumor_area_px2":                   tumor,
            "tumor_area_um2":                   tumor    * m * m,
            "stroma_area_px2":                  stroma,
            "stroma_area_um2":                  stroma   * m * m,
            "inflammatory_area_px2":            inflam,
            "inflammatory_area_um2":            inflam   * m * m,
            "necrosis_area_px2":                necrosis,
            "necrosis_area_um2":                necrosis * m * m,
            "others_area_px2":                  others,
            "others_area_um2":                  others   * m * m,
            "background_area_px2":              background_area,
            "background_area_um2":              background_area * m * m,
            "segmented_area_px2":               segmented_area,
            "segmented_area_um2":               segmented_area  * m * m,
            # tissue class perimeters
            "tumor_perimeter_px":               perim_px["Tumour"],
            "tumor_perimeter_um":               perim_px["Tumour"]       * m,
            "stroma_perimeter_px":              perim_px["Stroma"],
            "stroma_perimeter_um":              perim_px["Stroma"]       * m,
            "inflammatory_perimeter_px":        perim_px["Inflammatory"],
            "inflammatory_perimeter_um":        perim_px["Inflammatory"] * m,
            "necrosis_perimeter_px":            perim_px["Necrosis"],
            "necrosis_perimeter_um":            perim_px["Necrosis"]     * m,
            "others_perimeter_px":              perim_px["Others"],
            "others_perimeter_um":              perim_px["Others"]       * m,
            # tissue composition fractions
            "tissue_fraction":                  tissue_fraction,
            "background_fraction":              1.0 - tissue_fraction,
            "tumor_pct_of_segmented":           tumor    / segmented_area * 100 if segmented_area > 0 else np.nan,
            "stroma_pct_of_segmented":          stroma   / segmented_area * 100 if segmented_area > 0 else np.nan,
            "inflammatory_pct_of_segmented":    inflam   / segmented_area * 100 if segmented_area > 0 else np.nan,
            "necrosis_pct_of_segmented":        necrosis / segmented_area * 100 if segmented_area > 0 else np.nan,
            "others_pct_of_segmented":          others   / segmented_area * 100 if segmented_area > 0 else np.nan,
            # TSR
            "TSR_display":                      tsr_display_str,
            "TSR_tumor_pct":                    t_pct,
            "TSR_stroma_pct":                   s_pct,
            "TSR_stroma_fraction":              tsr_raw,
            "TSR_category": (
                "stroma-high" if tsr_reliable and np.isfinite(tsr_raw) and tsr_raw > 0.5
                else "stroma-low" if tsr_reliable and np.isfinite(tsr_raw)
                else "unreliable"
            ),
            "TSR_reliable":                     tsr_reliable,
            "TSR_denom_area_px2":               tsr_denom,
            # sTILs
            "sTILs_pct_salgado":                salgado_tils,
            "sTILs_pct_stromal":                stromal_tils,
            "sTILs_pct_tissue":                 tissue_tils,
            "sTILs_pct":                        tils_raw,
            "sTILs_denominator":                tils_denominator,
            "sTILs_level":                      tils_level(tils_raw, tils_reliable),
            "sTILs_reliable":                   tils_reliable,
            "sTILs_denom_area_px2":             tils_denom_area,
            # QC
            "cross_class_overlap_area_px2":     cross_class_overlap,
            "cross_class_overlap_frac":         cross_class_overlap_frac,
        }
        records.append(rec)

    return pd.DataFrame(records)


def area_weighted_wsi(
    cs: pd.DataFrame,
    all_pre: pd.DataFrame,
    tils_denominator: str,
    mpp: float,
) -> dict:
    """WSI-level TSR and sTILs by summing segmented areas across all clusters."""
    m = mpp

    tumor    = float(cs["tumor_area_px2"].sum())
    stroma   = float(cs["stroma_area_px2"].sum())
    inflam   = float(cs["inflammatory_area_px2"].sum())
    necrosis = float(cs["necrosis_area_px2"].sum())
    others   = float(cs["others_area_px2"].sum())
    segm     = float(cs["segmented_area_px2"].sum())
    bg       = float(cs["background_area_px2"].sum())
    cl_area  = float(cs["cluster_scoring_area_px2"].sum())
    viable   = tumor + stroma + inflam + others

    tsr_denom    = tumor + stroma
    tsr_raw      = stroma / tsr_denom if tsr_denom > 0 else np.nan
    t_pct        = tumor  / tsr_denom * 100 if tsr_denom > 0 else np.nan
    s_pct        = stroma / tsr_denom * 100 if tsr_denom > 0 else np.nan

    salgado_tils = inflam / stroma            * 100 if stroma > 0           else np.nan
    stromal_tils = inflam / (stroma + inflam) * 100 if (stroma + inflam) > 0 else np.nan
    tissue_tils  = inflam / viable            * 100 if viable > 0           else np.nan
    final_tils   = {
        "salgado":                 salgado_tils,
        "stroma_plus_inflammatory": stromal_tils,
        "tissue":                  tissue_tils,
    }[tils_denominator]

    filtered = all_pre[~all_pre["kept_after_min_roi_filter"]]

    return {
        # run metadata
        "tils_denominator_used":          tils_denominator,
        "mpp":                            mpp,
        # cluster counts
        "n_clusters_input":               int(len(all_pre)),
        "n_clusters_scored":              int(len(cs)),
        "n_clusters_filtered_min_roi":    int(len(filtered)),
        "filtered_cluster_ids":           ";".join(
            map(str, filtered["cluster_id"].astype(int).tolist())
        ),
        "n_roi_boxes_input":              int(all_pre["n_roi_boxes"].sum()),
        "n_roi_boxes_scored":             int(cs["n_roi_boxes"].sum()),
        "n_clusters_TSR_reliable":        int(cs["TSR_reliable"].sum()),
        "n_clusters_sTILs_reliable":      int(cs["sTILs_reliable"].sum()),
        # WSI TSR
        "WSI_TSR_display":                fmt_tsr(t_pct, s_pct),
        "WSI_TSR_tumor_pct":              t_pct,
        "WSI_TSR_stroma_pct":             s_pct,
        "WSI_TSR_stroma_fraction":        tsr_raw,
        "WSI_TSR_category": (
            "stroma-high" if np.isfinite(tsr_raw) and tsr_raw > 0.5
            else "stroma-low" if np.isfinite(tsr_raw) else "NA"
        ),
        # WSI sTILs
        "WSI_sTILs_pct_salgado":          salgado_tils,
        "WSI_sTILs_pct_stromal":          stromal_tils,
        "WSI_sTILs_pct_tissue":           tissue_tils,
        "WSI_sTILs_pct_selected":         final_tils,
        "WSI_sTILs_level":                tils_level(
            final_tils, bool(np.isfinite(final_tils))
        ),
        # WSI area sums (px² and µm²)
        "total_cluster_area_px2":         cl_area,
        "total_cluster_area_um2":         cl_area  * m * m,
        "total_cluster_area_mm2":         cl_area  * m * m / 1e6,
        "total_segmented_area_px2":       segm,
        "total_segmented_area_um2":       segm     * m * m,
        "total_background_area_px2":      bg,
        "total_background_area_um2":      bg       * m * m,
        "total_tumor_area_px2":           tumor,
        "total_tumor_area_um2":           tumor    * m * m,
        "total_stroma_area_px2":          stroma,
        "total_stroma_area_um2":          stroma   * m * m,
        "total_inflammatory_area_px2":    inflam,
        "total_inflammatory_area_um2":    inflam   * m * m,
        "total_necrosis_area_px2":        necrosis,
        "total_necrosis_area_um2":        necrosis * m * m,
        "total_others_area_px2":          others,
        "total_others_area_um2":          others   * m * m,
        "total_tissue_fraction":          min(segm / cl_area, 1.0) if cl_area > 0 else np.nan,
        "WSI_frac_Tumour":                tumor    / cl_area if cl_area > 0 else np.nan,
        "WSI_frac_Stroma":                stroma   / cl_area if cl_area > 0 else np.nan,
        "WSI_frac_Inflammatory":          inflam   / cl_area if cl_area > 0 else np.nan,
        "WSI_frac_Necrosis":              necrosis / cl_area if cl_area > 0 else np.nan,
        "WSI_frac_Others":                others   / cl_area if cl_area > 0 else np.nan,
        "total_viable_area_px2":          viable,
        "total_viable_area_um2":          viable   * m * m,
        # QC
        "total_cross_class_overlap_area_px2": float(cs["cross_class_overlap_area_px2"].sum()),
        "mean_cross_class_overlap_frac":      float(cs["cross_class_overlap_frac"].mean()),
        "n_clusters_with_overlap":            int((cs["cross_class_overlap_frac"] > 0.01).sum()),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def polygon_patches(geom: BaseGeometry):
    from matplotlib.patches import Polygon as MplPolygon
    polys = (
        list(geom.geoms) if geom.geom_type == "MultiPolygon"
        else [geom] if geom.geom_type == "Polygon" else []
    )
    out = []
    for poly in polys:
        if poly.is_empty:
            continue
        coords = np.asarray(poly.exterior.coords)
        if len(coords) >= 3:
            out.append(MplPolygon(coords, closed=True))
    return out


def plot_cluster_overlay(
    regions: pd.DataFrame,
    clusters: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    out_png: Path,
    draw_region_map: bool = True,
    max_regions_to_draw: int = 50_000,
) -> None:
    fig = plt.figure(figsize=(20, 13))
    gs  = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.6], wspace=0.02)
    ax       = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_panel.axis("off")

    # Draw tissue in DRAW_ORDER so Tumour sits on top of everything
    if draw_region_map:
        for cls in tqdm(
            DRAW_ORDER, desc="Rendering tissue overlay", unit="class", leave=False
        ):
            sub = regions[regions["class"] == cls]
            if sub.empty:
                continue
            if len(sub) > max_regions_to_draw:
                sub = sub.sample(max_regions_to_draw, random_state=7)
            patches = []
            for geom in sub["geometry"]:
                patches.extend(polygon_patches(geom))
            if patches:
                ax.add_collection(PatchCollection(
                    patches,
                    facecolor=CLASS_COLORS_RGB[cls],
                    edgecolor="none", alpha=0.85, match_original=False,
                ))

    score_map = (
        cluster_scores
        .drop_duplicates(subset="cluster_id", keep="first")
        .set_index("cluster_id")
    )

    panel_rows = []
    for c in clusters.itertuples():
        cid = int(c.cluster_id)
        if cid not in score_map.index:
            continue
        s    = score_map.loc[cid]
        edge = TILS_LEVEL_COLORS.get(str(s["sTILs_level"]), "#888888")

        patches = polygon_patches(c.geometry)
        if patches:
            ax.add_collection(PatchCollection(
                patches, facecolor="none", edgecolor=edge,
                linewidth=3.5, alpha=1.0, match_original=False,
            ))

        cx, cy = c.geometry.representative_point().coords[0]
        ax.text(
            cx, cy, f"C{cid}",
            fontsize=9, weight="bold", color="black", ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.78, edgecolor=edge,
                      boxstyle="round,pad=0.20"),
        )

        panel_rows.append({
            "cluster_id": cid, "edge": edge,
            "tsr_disp":  str(s["TSR_display"]),
            "tsr_cat":   str(s["TSR_category"]),
            "tils_val":  float(s["sTILs_pct"]),
            "tils_lvl":  str(s["sTILs_level"]),
            "area_mm2":  float(s["cluster_scoring_area_um2"]) / 1e6,
            "bg_frac":   float(s["background_fraction"]),
            "tsr_rel":   bool(s["TSR_reliable"]),
            "til_rel":   bool(s["sTILs_reliable"]),
        })

    # Axis limits
    all_bounds = [g.bounds for g in clusters["geometry"]]
    if draw_region_map and len(regions):
        all_bounds.extend([g.bounds for g in regions["geometry"]])
    b = np.array(all_bounds)
    x0, y0 = b[:, 0].min(), b[:, 1].min()
    x1, y1 = b[:, 2].max(), b[:, 3].max()
    pad_x, pad_y = (x1 - x0) * 0.02, (y1 - y0) * 0.02
    ax.set_xlim(x0 - pad_x, x1 + pad_x)
    ax.set_ylim(y1 + pad_y, y0 - pad_y)
    ax.set_aspect("equal")
    ax.set_xlabel("WSI x (px)", fontsize=10)
    ax.set_ylabel("WSI y (px)", fontsize=10)
    ax.set_title(
        "Non-overlapping cluster polygons — TSR & sTILs\n"
        "TSR = Tumour%/Stroma%;  sTILs = Inflammatory/Stroma (Salgado 2015)\n"
        "Background pixels excluded from all TSR/sTILs denominators",
        fontsize=9,
    )

    class_handles = [
        mpatches.Patch(facecolor=CLASS_COLORS_RGB[c], edgecolor="none", label=c)
        for c in CLASS_ORDER
    ]
    level_handles = [
        mpatches.Patch(facecolor="none", edgecolor=col, linewidth=3.0,
                       label=f"sTILs {lvl}")
        for lvl, col in TILS_LEVEL_COLORS.items()
    ]
    leg1 = ax.legend(handles=class_handles, loc="upper right", fontsize=8,
                     title="Tissue class", framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=level_handles, loc="lower right", fontsize=8,
              title="Cluster outline", framealpha=0.9)

    # Score panel
    ax_panel.text(0.02, 0.990, "Cluster scores",
                  fontsize=13, weight="bold", va="top")
    ax_panel.text(
        0.02, 0.955,
        "TSR  =  Tumour% / Stroma%\n"
        "sTILs = Inflam / Stroma × 100\n"
        "Background excluded from denominators",
        fontsize=7.5, va="top", color="#444444",
    )

    y     = 0.885
    n     = max(1, len(panel_rows))
    r_gap = 0.145 if n <= 5 else max(0.09, 0.84 / n)

    for row in sorted(panel_rows, key=lambda r: r["cluster_id"]):
        ax_panel.text(
            0.02, y, f"C{row['cluster_id']}",
            fontsize=10, weight="bold", va="top",
            bbox=dict(facecolor="white", edgecolor=row["edge"],
                      linewidth=2.2, boxstyle="round,pad=0.25"),
        )
        tsr_flag  = "" if row["tsr_rel"]  else " ⚠"
        tils_flag = "" if row["til_rel"]  else " ⚠"
        ax_panel.text(
            0.20, y,
            f"TSR {row['tsr_disp']} (T%/S%){tsr_flag}  [{row['tsr_cat']}]\n"
            f"sTILs {row['tils_val']:.1f}%{tils_flag}  [{row['tils_lvl']}]\n"
            f"Area {row['area_mm2']:.3f} mm²   Bg {row['bg_frac']:.1%}",
            fontsize=8.5, va="top",
        )
        y -= r_gap

    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Internal orchestrator  (mirrors original run_scoring, path-unaware)
# ---------------------------------------------------------------------------

def _run_scoring(scoring_cfg: _ScoringConfig) -> dict:
    """
    Execute the full scoring pipeline for one slide.
    Returns a plain dict of result paths + summary; no ScoringResult wrapper.
    """
    scoring_cfg.validate()
    scoring_cfg.outdir.mkdir(parents=True, exist_ok=True)

    cluster_buffer_px = scoring_cfg.cluster_buffer_um / scoring_cfg.mpp
    rois = load_roi_boxes(scoring_cfg.roi_boxes)

    clusters, all_pre = dissolve_buffer_filter_nonoverlap(
        rois=rois,
        buffer_px=cluster_buffer_px,
        min_roi_boxes_per_cluster=scoring_cfg.min_roi_boxes_per_cluster,
    )
    if clusters.empty:
        raise RuntimeError(
            "No clusters remained after filtering / non-overlap clipping."
        )

    cluster_geojson = scoring_cfg.outdir / "cluster_scoring_polygons.geojson"
    export_cluster_polygons_geojson(clusters, cluster_geojson, scoring_cfg.mpp)
    print(f"  Wrote: {cluster_geojson.name}")

    regions, filter_info = load_regions(
        scoring_cfg.geojson,
        min_polygon_area_px2=scoring_cfg.min_polygon_area_px2,
        max_area_quantile=scoring_cfg.max_area_quantile,
        max_area_mad_z=scoring_cfg.max_area_mad_z,
    )
    if regions.empty:
        raise RuntimeError("No usable GeoJSON regions after loading / filtering.")

    cluster_scores = score_clusters(
        clusters=clusters,
        regions=regions,
        min_tissue_fraction=scoring_cfg.min_tissue_fraction,
        min_tsr_denom_area_px2=scoring_cfg.min_tsr_denom_area_px2,
        min_tils_denom_area_px2=scoring_cfg.min_tils_denom_area_px2,
        tils_denominator=scoring_cfg.tils_denominator,
        mpp=scoring_cfg.mpp,
        cross_class_overlap_warn_fraction=scoring_cfg.cross_class_overlap_warn_fraction,
    )

    cluster_csv = scoring_cfg.outdir / "tils_tsr_by_cluster.csv"
    cluster_scores.to_csv(cluster_csv, index=False)
    print(f"  Wrote: {cluster_csv.name}")

    wsi_summary = area_weighted_wsi(
        cluster_scores, all_pre, scoring_cfg.tils_denominator, scoring_cfg.mpp
    )
    wsi_summary.update({
        "cluster_buffer_um":             scoring_cfg.cluster_buffer_um,
        "min_roi_boxes_per_cluster":     scoring_cfg.min_roi_boxes_per_cluster,
        "n_area_intersections_final":    clusters.attrs.get("n_area_intersections", 0),
        "max_pairwise_intersection_px2": clusters.attrs.get("max_pairwise_intersection_area", 0.0),
        **filter_info,
    })

    wsi_csv = scoring_cfg.outdir / "tils_tsr_wsi_summary.csv"
    pd.DataFrame([wsi_summary]).to_csv(wsi_csv, index=False)
    print(f"  Wrote: {wsi_csv.name}")

    overlay_png = None
    if not scoring_cfg.no_overlay:
        overlay_png = scoring_cfg.outdir / "cluster_tils_tsr_overlay.png"
        with tqdm(total=1, desc="Saving cluster overlay", unit="image"):
            plot_cluster_overlay(regions, clusters, cluster_scores, overlay_png)
        print(f"  Wrote: {overlay_png.name}")

    return {
        "cluster_csv":       str(cluster_csv),
        "wsi_csv":           str(wsi_csv),
        "cluster_geojson":   str(cluster_geojson),
        "overlay_png":       str(overlay_png) if overlay_png else None,
        "wsi_summary":       wsi_summary,
        "cluster_scores":    cluster_scores,
    }


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------

def run_cluster_tils_tsr_score(
    wsi_path: str,
    cfg: PipelineConfig = None,
) -> dict:
    """
    Run TSR / sTILs cluster scoring for one WSI.

    All inputs and outputs are derived from wsi_path and cfg — the caller
    never needs to pass manifest_path, roi_csv, geojson, out_dir, or
    slide_name manually.

    Reads
    -----
    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_roi_overlay/
        tumor_roi_boxes.csv
            Written by run_tumor_roi_overlay().

    cfg.OUT_DIR/<slide>/segmentation/
        segmentation_all_classes.geojson
            Written by run_stitching().

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
        cluster_scoring_polygons.geojson
        tils_tsr_by_cluster.csv
        tils_tsr_wsi_summary.csv
        cluster_tils_tsr_overlay.png

    Parameters
    ----------
    wsi_path : str
        Path to .svs / .tif — used to derive slide_name only.
    cfg : PipelineConfig
        Pipeline config.  Tuning knobs read from cfg (with fallbacks):
            cfg.MPP                          microns-per-pixel (0.25)
            cfg.TILS_DENOMINATOR             "salgado" | "stroma_plus_inflammatory"
                                             | "tissue"  (default "salgado")
            cfg.CLUSTER_BUFFER_UM            buffer around ROI boxes (200.0)
            cfg.CLUSTER_MIN_ROI_BOXES        min ROI boxes per cluster (5)
            cfg.CLUSTER_MIN_POLYGON_AREA_PX2 min polygon area filter (1.0)
            cfg.CLUSTER_MAX_AREA_QUANTILE    upper area quantile filter (1.0)
            cfg.CLUSTER_MAX_AREA_MAD_Z       MAD-z outlier rejection (0.0)
            cfg.CLUSTER_MIN_TISSUE_FRACTION  reliability gate (0.10)
            cfg.CLUSTER_MIN_TSR_DENOM_PX2    min TSR denom area (5000.0)
            cfg.CLUSTER_MIN_TILS_DENOM_PX2   min sTILs denom area (5000.0)
            cfg.CLUSTER_NO_OVERLAY           skip PNG overlay (False)

    Returns
    -------
    dict
        slide_name       str
        roi_csv          str   input ROI boxes CSV
        geojson          str   input segmentation GeoJSON
        cluster_csv      str   tils_tsr_by_cluster.csv
        wsi_csv          str   tils_tsr_wsi_summary.csv
        cluster_geojson  str   cluster_scoring_polygons.geojson
        overlay_png      str | None
        wsi_summary      dict  WSI-level scores
    """
    if cfg is None:
        cfg = default_cfg

    slide_name = Path(wsi_path).stem

    # ── Infer all input / output paths ──────────────────────────────────────
    roi_csv = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "tumor_roi_overlay"
        / "tumor_roi_boxes.csv"
    )
    geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    out_dir = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "cluster_tils_tsr_score"
    )

    if not roi_csv.exists():
        raise FileNotFoundError(
            f"ROI boxes CSV not found: {roi_csv}\n"
            f"Run run_tumor_roi_overlay(wsi_path, cfg) first."
        )
    if not geojson.exists():
        raise FileNotFoundError(
            f"Segmentation GeoJSON not found: {geojson}\n"
            f"Run run_stitching(wsi_path, cfg) first."
        )

    # ── Config knobs with safe fallbacks ────────────────────────────────────
    mpp                  = getattr(cfg, "MPP",                          0.25)
    tils_denominator     = getattr(cfg, "TILS_DENOMINATOR",             "salgado")
    cluster_buffer_um    = getattr(cfg, "CLUSTER_BUFFER_UM",            200.0)
    min_roi_boxes        = getattr(cfg, "CLUSTER_MIN_ROI_BOXES",        5)
    min_poly_area        = getattr(cfg, "CLUSTER_MIN_POLYGON_AREA_PX2", 1.0)
    max_area_quantile    = getattr(cfg, "CLUSTER_MAX_AREA_QUANTILE",    1.0)
    max_area_mad_z       = getattr(cfg, "CLUSTER_MAX_AREA_MAD_Z",       0.0)
    min_tissue_frac      = getattr(cfg, "CLUSTER_MIN_TISSUE_FRACTION",  0.10)
    min_tsr_denom        = getattr(cfg, "CLUSTER_MIN_TSR_DENOM_PX2",   5000.0)
    min_tils_denom       = getattr(cfg, "CLUSTER_MIN_TILS_DENOM_PX2",  5000.0)
    no_overlay           = getattr(cfg, "CLUSTER_NO_OVERLAY",           False)

    print(f"\n{'='*55}")
    print(f"  Cluster TSR / sTILs scoring")
    print(f"  Slide      : {slide_name}")
    print(f"  ROI boxes  : {roi_csv.name}")
    print(f"  GeoJSON    : {geojson.name}")
    print(f"  sTILs denom: {tils_denominator}")
    print(f"  mpp        : {mpp}")
    print(f"  Output     : {out_dir}")
    print(f"{'='*55}")

    scoring_cfg = _ScoringConfig(
        roi_boxes=roi_csv,
        geojson=geojson,
        outdir=out_dir,
        tils_denominator=tils_denominator,
        cluster_buffer_um=cluster_buffer_um,
        mpp=mpp,
        min_roi_boxes_per_cluster=min_roi_boxes,
        min_polygon_area_px2=min_poly_area,
        max_area_quantile=max_area_quantile,
        max_area_mad_z=max_area_mad_z,
        min_tissue_fraction=min_tissue_frac,
        min_tsr_denom_area_px2=min_tsr_denom,
        min_tils_denom_area_px2=min_tils_denom,
        no_overlay=no_overlay,
    )

    result = _run_scoring(scoring_cfg)

    # ── Console summary ──────────────────────────────────────────────────────
    s = result["wsi_summary"]
    print(f"\n  === WSI scoring summary ===")
    print(f"  Clusters input / scored : {s['n_clusters_input']} / {s['n_clusters_scored']}")
    print(f"  ROI boxes  input / scored: {s['n_roi_boxes_input']} / {s['n_roi_boxes_scored']}")
    print(f"  WSI TSR                 : {s['WSI_TSR_display']}  ({s['WSI_TSR_category']})")
    print(f"  WSI sTILs (Salgado)    : {s['WSI_sTILs_pct_salgado']:.2f}%")
    print(f"  WSI sTILs (selected)   : {s['WSI_sTILs_pct_selected']:.2f}%  [{tils_denominator}]")
    print(f"  Total cluster area      : {s['total_cluster_area_mm2']:.4f} mm²")
    print(f"  Tissue fraction         : {s['total_tissue_fraction']:.3f}")
    if s["mean_cross_class_overlap_frac"] > 0.01:
        print(
            f"  ⚠ Cross-class GeoJSON overlap: mean={s['mean_cross_class_overlap_frac']:.1%} "
            f"in {s['n_clusters_with_overlap']} cluster(s). Tissue areas may be inflated."
        )
    print(f"\n  Done.")

    return {
        "slide_name":      slide_name,
        "roi_csv":         str(roi_csv),
        "geojson":         str(geojson),
        "cluster_csv":     result["cluster_csv"],
        "wsi_csv":         result["wsi_csv"],
        "cluster_geojson": result["cluster_geojson"],
        "overlay_png":     result["overlay_png"],
        "wsi_summary":     s,
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) — every PipelineConfig
# field (including TILS_DENOMINATOR and the CLUSTER_* knobs) is available
# as a flag, plus --from-json to pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python cluster_tils_tsr_score.py --from-json run_config.json

def main(argv=None) -> None:
    from config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    slide_name = Path(cfg.WSI_PATH).stem
    roi_csv = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "tumor_roi_overlay" / "tumor_roi_boxes.csv"
    )
    geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    if not roi_csv.exists():
        raise SystemExit(
            f"ROI boxes CSV not found: {roi_csv}. "
            f"Run tumor_roi_overlay.py for this slide (with the same --out-dir) first."
        )
    if not geojson.exists():
        raise SystemExit(
            f"Segmentation GeoJSON not found: {geojson}. "
            f"Run stitch.py for this slide (with the same --out-dir) first."
        )

    run_cluster_tils_tsr_score(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()