#!/usr/bin/env python3
"""
necrosis_proximity_features.py
================================
Stage 7 of the ViSpace pipeline — necrosis proximity feature extraction.

Reads cluster polygons produced by run_cluster_tils_tsr_score() (stage 5)
and the segmentation GeoJSON produced by run_stitching() (stage 3).

Output directory
----------------
    cfg.OUT_DIR/<slide>/spatial_feature_results/necrosis_proximity/
        necrosis_proximity_by_cluster.csv
        necrosis_proximity_wsi_summary.csv
        necrosis_distance_figure.png
        necrosis_tissue_context_figure.png

Pipeline position
-----------------
    tessellate.py → segmenter.py → stitch.py → tumor_roi_overlay.py
        → cluster_tils_tsr_score.py → immune_proximity_features.py
        → necrosis_proximity_features.py → tumor_morphology_features.py

Usage (as a library)
---------------------
    from vispace import run_necrosis_proximity_features
    from vispace import cfg

    run_necrosis_proximity_features("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
------------------------------
Same shared flags as the rest of the pipeline — every PipelineConfig field
is available here too, including the NECROSIS_* knobs. Also supports
--from-json to pick up a config saved earlier via `config.py --print-config`.

    # minimal — requires cluster_tils_tsr_score.py to have already run
    python necrosis_proximity_features.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python necrosis_proximity_features.py --from-json run_config.json

    # continue from a saved config but override one knob
    python necrosis_proximity_features.py --from-json run_config.json \\
        --necrosis-immune-coupling-threshold-um 150

    # see every available flag
    python necrosis_proximity_features.py --help
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from .config import cfg as default_cfg, PipelineConfig


# ---------------------------------------------------------------------------
# Internal config dataclass  (private — callers use run_necrosis_proximity_features)
# ---------------------------------------------------------------------------

@dataclass
class _NecrosisCfg:
    cluster_geojson:               Path
    segmentation_geojson:          Path
    outdir:                        Path  = field(default_factory=lambda: Path("necrosis_proximity_output"))
    mpp:                           float = 0.25
    proximity_thresholds_um:       List[float] = field(default_factory=lambda: [50.0, 100.0])
    contact_tolerance_um:          float = 5.0
    immune_coupling_threshold_um:  float = 100.0
    min_component_area_um2:        float = 500.0
    # Phenotype thresholds
    absent_max_area_um2:               float = 500.0
    central_min_intra_frac:            float = 0.50
    peritumoural_min_pct_within_100um: float = 60.0
    immune_adjacent_min_coupling:      float = 0.20
    distant_min_median_um:             float = 150.0

    def validate(self) -> None:
        if not self.cluster_geojson.exists():
            raise FileNotFoundError(f"Cluster GeoJSON not found: {self.cluster_geojson}")
        if not self.segmentation_geojson.exists():
            raise FileNotFoundError(f"Segmentation GeoJSON not found: {self.segmentation_geojson}")
        if self.mpp <= 0:
            raise ValueError("mpp must be > 0")


# ---------------------------------------------------------------------------
# Geometry helpers  (unchanged)
# ---------------------------------------------------------------------------

def fix_geom(g: BaseGeometry) -> Optional[BaseGeometry]:
    if g is None or g.is_empty:
        return g
    if not g.is_valid:
        g = g.buffer(0)
    return g


def polygon_parts(g: BaseGeometry) -> List[BaseGeometry]:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if g.geom_type in {"MultiPolygon", "GeometryCollection"}:
        out = []
        for part in g.geoms:
            out.extend(polygon_parts(part))
        return out
    return []


def safe_div(num: float, den: float, default: float = np.nan) -> float:
    try:
        return float(num) / float(den) if den and not np.isnan(den) and den != 0 else default
    except Exception:
        return default


def px2_to_um2(a: float, mpp: float) -> float:
    return float(a) * mpp * mpp


def px_to_um(l: float, mpp: float) -> float:
    return float(l) * mpp


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    v = values[mask]; w = weights[mask]
    order = np.argsort(v)
    v = v[order]; w = w[order]
    csum = np.cumsum(w)
    return float(v[np.searchsorted(csum, 0.5 * w.sum())])


def build_strtree(geoms: List[BaseGeometry]) -> Optional[STRtree]:
    if not geoms:
        return None
    return STRtree(geoms)


def nearest_distance_um(
    comp: BaseGeometry,
    tree: Optional[STRtree],
    geoms: List[BaseGeometry],
    mpp:  float,
) -> float:
    if tree is None or not geoms or comp is None or comp.is_empty:
        return np.nan
    rep  = comp.representative_point()
    hits = tree.query(comp.buffer(comp.area ** 0.5 * 10))
    if len(hits) == 0:
        return np.nan
    if isinstance(hits[0], (int, np.integer)):
        indices = [int(i) for i in hits]
    else:
        id_map  = {id(g): i for i, g in enumerate(geoms)}
        indices = [id_map[id(g)] for g in hits]
    min_dist = min((rep.distance(geoms[i]) for i in indices
                    if i < len(geoms) and not geoms[i].is_empty), default=np.nan)
    return float(min_dist) * mpp if np.isfinite(min_dist) else np.nan


# ---------------------------------------------------------------------------
# Loading  (unchanged)
# ---------------------------------------------------------------------------

def load_clusters(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for i, feat in enumerate(data.get("features", [])):
        props = feat.get("properties", {}) or {}
        g     = fix_geom(shape(feat["geometry"]))
        if g is None or g.is_empty:
            continue
        cid = props.get("cluster_id", i)
        rows.append({"cluster_id": int(cid), "geometry": g,
                     "n_roi_boxes": props.get("n_roi_boxes", np.nan)})
    if not rows:
        raise RuntimeError(f"No cluster polygons in {path}")
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


def load_segmentation(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows      = []
    valid     = {"Tumour", "Stroma", "Inflammatory", "Necrosis", "Others",
                 "Tumor", "tumor", "tumour"}
    class_map = {"Tumor": "Tumour", "tumor": "Tumour", "tumour": "Tumour"}
    for feat in data.get("features", []):
        props = feat.get("properties", {}) or {}
        cls   = (props.get("classification") or {}).get("name") or props.get("class")
        if cls is None or cls not in valid:
            continue
        cls = class_map.get(cls, cls)
        g   = fix_geom(shape(feat["geometry"]))
        if g is None or g.is_empty:
            continue
        rows.append({"class": cls, "geometry": g})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Component table  (unchanged)
# ---------------------------------------------------------------------------

def compute_component_table(
    necrosis_parts: List[BaseGeometry],
    tumor_union:    Optional[BaseGeometry],
    inflam_geoms:   List[BaseGeometry],
    stroma_geoms:   List[BaseGeometry],
    inflam_tree:    Optional[STRtree],
    stroma_tree:    Optional[STRtree],
    mpp:            float,
) -> pd.DataFrame:
    rows = []
    tumor_empty = tumor_union is None or tumor_union.is_empty
    if not tumor_empty:
        tumor_boundary = tumor_union.boundary

    for j, comp in enumerate(necrosis_parts):
        if comp is None or comp.is_empty or comp.area <= 0:
            continue
        area_um2 = px2_to_um2(comp.area, mpp)

        if tumor_empty:
            signed_um   = np.nan
            compartment = "no_tumor_reference"
        else:
            rep     = comp.representative_point()
            dist_um = px_to_um(rep.distance(tumor_boundary), mpp)
            if tumor_union.covers(rep):
                signed_um   = -dist_um
                compartment = "intratumoral"
            else:
                signed_um   = dist_um
                compartment = "extratumoral"

        immune_dist_um = nearest_distance_um(comp, inflam_tree, inflam_geoms, mpp)
        stroma_dist_um = nearest_distance_um(comp, stroma_tree, stroma_geoms, mpp)

        rows.append({
            "component_id":      j,
            "area_um2":          area_um2,
            "signed_dist_um":    signed_um,
            "abs_dist_um":       abs(signed_um) if np.isfinite(signed_um) else np.nan,
            "compartment":       compartment,
            "immune_dist_um":    immune_dist_um,
            "stroma_dist_um":    stroma_dist_um,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["component_id", "area_um2", "signed_dist_um", "abs_dist_um",
                 "compartment", "immune_dist_um", "stroma_dist_um"]
    )


# ---------------------------------------------------------------------------
# Phenotype classification  (unchanged)
# ---------------------------------------------------------------------------

def classify_phenotype(
    total_area_um2:   float,
    intra_frac:       float,
    pct_within_100um: float,
    immune_coupling:  float,
    median_extra_um:  float,
    cfg:              _NecrosisCfg,
) -> str:
    if not np.isfinite(total_area_um2) or total_area_um2 < cfg.absent_max_area_um2:
        return "necrosis-absent"
    if np.isfinite(intra_frac) and intra_frac >= cfg.central_min_intra_frac:
        return "tumour-central"
    if np.isfinite(pct_within_100um) and pct_within_100um >= cfg.peritumoural_min_pct_within_100um:
        return "peritumoural"
    if np.isfinite(immune_coupling) and immune_coupling >= cfg.immune_adjacent_min_coupling:
        return "immune-adjacent"
    return "stromal-distant"


# ---------------------------------------------------------------------------
# Per-cluster feature extraction  (unchanged)
# ---------------------------------------------------------------------------

def extract_cluster_features(
    cluster_id:      int,
    cluster_geom:    BaseGeometry,
    n_roi_boxes,
    necrosis_parts:  List[BaseGeometry],
    tumor_union:     Optional[BaseGeometry],
    inflam_geoms:    List[BaseGeometry],
    stroma_geoms:    List[BaseGeometry],
    inflam_tree:     Optional[STRtree],
    stroma_tree:     Optional[STRtree],
    mpp:             float,
    cfg:             _NecrosisCfg,
) -> dict:
    cluster_area_um2 = px2_to_um2(float(cluster_geom.area), mpp)

    comp_df = compute_component_table(
        necrosis_parts, tumor_union,
        inflam_geoms, stroma_geoms,
        inflam_tree, stroma_tree, mpp,
    )

    empty_row = {
        "cluster_id":                            cluster_id,
        "n_roi_boxes":                           n_roi_boxes,
        "cluster_area_um2":                      cluster_area_um2,
        "cluster_area_mm2":                      cluster_area_um2 / 1e6,
        "necrosis_area_total_um2":               0.0,
        "necrosis_area_total_mm2":               0.0,
        "necrosis_fraction_of_cluster":          0.0,
        "necrosis_to_tumor_ratio":               np.nan,
        "n_necrosis_components":                 0,
        "necrosis_area_cv":                      np.nan,
        "necrosis_area_intratumoral_um2":        0.0,
        "necrosis_fraction_intratumoral":        np.nan,
        "necrosis_area_extratumoral_um2":        0.0,
        "necrosis_fraction_extratumoral":        np.nan,
        "necrosis_extratumoral_distance_aw_median_um": np.nan,
        "necrosis_contact_area_um2":             0.0,
        "necrosis_contact_fraction":             np.nan,
        "necrosis_immune_min_dist_aw_median_um": np.nan,
        "necrosis_immune_coupling_index":        0.0,
        "necrosis_stroma_min_dist_aw_median_um": np.nan,
        "necrosis_phenotype":                    "necrosis-absent",
    }
    for t in cfg.proximity_thresholds_um:
        empty_row[f"necrosis_pct_within_{int(t)}um"] = np.nan

    if comp_df.empty or comp_df["area_um2"].sum() == 0:
        return empty_row

    total_area_um2 = float(comp_df["area_um2"].sum())
    n_comp         = int(len(comp_df))
    area_cv        = (float(comp_df["area_um2"].std(ddof=1) / comp_df["area_um2"].mean())
                      if n_comp > 1 else 0.0)

    intra_mask = comp_df["compartment"] == "intratumoral"
    extra_mask = comp_df["compartment"] == "extratumoral"
    intra_area = float(comp_df.loc[intra_mask, "area_um2"].sum())
    extra_area = float(comp_df.loc[extra_mask, "area_um2"].sum())

    tumor_area_um2 = (
        px2_to_um2(float(tumor_union.area), mpp)
        if tumor_union is not None and not tumor_union.is_empty else 0.0
    )

    extra_df = comp_df[extra_mask].dropna(subset=["abs_dist_um"])
    aw_med = (
        weighted_median(extra_df["abs_dist_um"].to_numpy(float),
                        extra_df["area_um2"].to_numpy(float))
        if not extra_df.empty and extra_df["area_um2"].sum() > 0 else np.nan
    )

    contact_mask = comp_df["abs_dist_um"] <= cfg.contact_tolerance_um
    contact_area = float(comp_df.loc[contact_mask, "area_um2"].sum())

    pcts: dict = {}
    for t in cfg.proximity_thresholds_um:
        within_mask = comp_df["abs_dist_um"] <= t
        area_within = float(comp_df.loc[within_mask, "area_um2"].sum())
        pcts[f"necrosis_pct_within_{int(t)}um"] = 100.0 * safe_div(area_within, total_area_um2)

    # Immune proximity (TNF-proxy)
    imm_valid = comp_df.dropna(subset=["immune_dist_um"])
    imm_median = (
        weighted_median(imm_valid["immune_dist_um"].to_numpy(float),
                        imm_valid["area_um2"].to_numpy(float))
        if not imm_valid.empty else np.nan
    )
    coupled_mask = comp_df["immune_dist_um"] <= cfg.immune_coupling_threshold_um
    immune_coupling = safe_div(
        float(comp_df.loc[coupled_mask, "area_um2"].sum()), total_area_um2, default=0.0
    )

    # Stroma proximity (remodelling proxy)
    str_valid  = comp_df.dropna(subset=["stroma_dist_um"])
    str_median = (
        weighted_median(str_valid["stroma_dist_um"].to_numpy(float),
                        str_valid["area_um2"].to_numpy(float))
        if not str_valid.empty else np.nan
    )

    pct_within_100 = pcts.get("necrosis_pct_within_100um", np.nan)
    phenotype = classify_phenotype(
        total_area_um2,
        safe_div(intra_area, total_area_um2),
        pct_within_100,
        immune_coupling,
        aw_med,
        cfg,
    )

    row = {
        "cluster_id":                            cluster_id,
        "n_roi_boxes":                           n_roi_boxes,
        "cluster_area_um2":                      cluster_area_um2,
        "cluster_area_mm2":                      cluster_area_um2 / 1e6,
        "necrosis_area_total_um2":               total_area_um2,
        "necrosis_area_total_mm2":               total_area_um2 / 1e6,
        "necrosis_fraction_of_cluster":          safe_div(total_area_um2, cluster_area_um2),
        "necrosis_to_tumor_ratio":               safe_div(total_area_um2, tumor_area_um2),
        "n_necrosis_components":                 n_comp,
        "necrosis_area_cv":                      area_cv,
        "necrosis_area_intratumoral_um2":        intra_area,
        "necrosis_fraction_intratumoral":        safe_div(intra_area, total_area_um2),
        "necrosis_area_extratumoral_um2":        extra_area,
        "necrosis_fraction_extratumoral":        safe_div(extra_area, total_area_um2),
        "necrosis_extratumoral_distance_aw_median_um": aw_med,
        "necrosis_contact_area_um2":             contact_area,
        "necrosis_contact_fraction":             safe_div(contact_area, total_area_um2),
        "necrosis_immune_min_dist_aw_median_um": imm_median,
        "necrosis_immune_coupling_index":        immune_coupling,
        "necrosis_stroma_min_dist_aw_median_um": str_median,
        "necrosis_phenotype":                    phenotype,
    }
    row.update(pcts)
    return row


# ---------------------------------------------------------------------------
# WSI summary  (unchanged)
# ---------------------------------------------------------------------------

def compute_wsi_summary(cluster_df: pd.DataFrame, cfg: _NecrosisCfg) -> pd.DataFrame:
    total_cl_um2       = float(cluster_df["cluster_area_um2"].sum())
    total_nec_um2      = float(cluster_df["necrosis_area_total_um2"].sum())
    total_intra_um2    = float(cluster_df["necrosis_area_intratumoral_um2"].sum())
    total_extra_um2    = float(cluster_df["necrosis_area_extratumoral_um2"].sum())
    total_contact_um2  = float(cluster_df["necrosis_contact_area_um2"].sum())

    out: dict = {
        "n_clusters":                                 int(len(cluster_df)),
        "total_cluster_area_mm2":                     total_cl_um2 / 1e6,
        "wsi_necrosis_area_total_mm2":                total_nec_um2 / 1e6,
        "wsi_necrosis_fraction_of_cluster":           safe_div(total_nec_um2, total_cl_um2),
        "wsi_n_necrosis_components":                  int(cluster_df["n_necrosis_components"].sum()),
        "wsi_necrosis_area_intratumoral_um2":         total_intra_um2,
        "wsi_necrosis_fraction_intratumoral":         safe_div(total_intra_um2, total_nec_um2),
        "wsi_necrosis_area_extratumoral_um2":         total_extra_um2,
        "wsi_necrosis_fraction_extratumoral":         safe_div(total_extra_um2, total_nec_um2),
        "wsi_necrosis_contact_area_um2":              total_contact_um2,
        "wsi_necrosis_contact_fraction":              safe_div(total_contact_um2, total_nec_um2),
    }

    for t in cfg.proximity_thresholds_um:
        col = f"necrosis_pct_within_{int(t)}um"
        if col in cluster_df.columns:
            area_w = cluster_df[col].fillna(0) / 100.0 * cluster_df["necrosis_area_total_um2"]
            out[f"wsi_{col}"] = 100.0 * safe_div(float(area_w.sum()), total_nec_um2)

    valid_ext = cluster_df[cluster_df["necrosis_extratumoral_distance_aw_median_um"].notna()].copy()
    out["wsi_necrosis_extratumoral_distance_aw_median_um"] = (
        weighted_median(
            valid_ext["necrosis_extratumoral_distance_aw_median_um"].to_numpy(float),
            valid_ext["necrosis_area_extratumoral_um2"].to_numpy(float),
        )
        if not valid_ext.empty else np.nan
    )

    nec_present = cluster_df[cluster_df["necrosis_area_total_um2"] > 0]
    for col in ("necrosis_to_tumor_ratio", "necrosis_immune_coupling_index"):
        if col in nec_present.columns and not nec_present.empty:
            w    = nec_present["necrosis_area_total_um2"].to_numpy(float)
            v    = nec_present[col].to_numpy(float)
            mask = np.isfinite(v) & (w > 0)
            out[f"wsi_{col}"] = float(np.average(v[mask], weights=w[mask])) if mask.any() else np.nan
        else:
            out[f"wsi_{col}"] = np.nan

    for col in ("necrosis_immune_min_dist_aw_median_um", "necrosis_stroma_min_dist_aw_median_um"):
        valid_c = cluster_df[cluster_df[col].notna()].copy() if col in cluster_df.columns else pd.DataFrame()
        out[f"wsi_{col}"] = (
            weighted_median(
                valid_c[col].to_numpy(float),
                valid_c["necrosis_area_total_um2"].to_numpy(float),
            )
            if not valid_c.empty else np.nan
        )

    if "necrosis_phenotype" in cluster_df.columns and total_nec_um2 > 0:
        pheno_area = cluster_df.groupby("necrosis_phenotype")["necrosis_area_total_um2"].sum()
        out["wsi_dominant_necrosis_phenotype"] = str(pheno_area.idxmax())
    else:
        out["wsi_dominant_necrosis_phenotype"] = "necrosis-absent"

    out["contact_tolerance_um_used"]       = cfg.contact_tolerance_um
    out["immune_coupling_threshold_um"]    = cfg.immune_coupling_threshold_um
    out["proximity_thresholds_um"]         = ";".join(str(int(t)) for t in cfg.proximity_thresholds_um)

    return pd.DataFrame([out])


# ---------------------------------------------------------------------------
# Plots  (unchanged rendering logic, paths come from cfg.outdir)
# ---------------------------------------------------------------------------

PHENOTYPE_COLORS = {
    "necrosis-absent":   "#aaaaaa",
    "tumour-central":    "#c0392b",
    "peritumoural":      "#e67e22",
    "immune-adjacent":   "#8e44ad",
    "stromal-distant":   "#2980b9",
}


def plot_distance_figure(
    cluster_df: pd.DataFrame,
    comp_dfs:   List[pd.DataFrame],
    cfg:        _NecrosisCfg,
    out_png:    Path,
) -> None:
    fig = plt.figure(figsize=(16, 6), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.07, right=0.97,
                            top=0.88, bottom=0.14, wspace=0.32)
    ax_hist  = fig.add_subplot(gs[0, 0])
    ax_stack = fig.add_subplot(gs[0, 1])

    all_dists, all_areas = [], []
    for cdf in comp_dfs:
        if cdf is not None and not cdf.empty:
            valid = cdf.dropna(subset=["signed_dist_um", "area_um2"])
            all_dists.extend(valid["signed_dist_um"].tolist())
            all_areas.extend(valid["area_um2"].tolist())

    if all_dists:
        all_dists   = np.array(all_dists)
        all_areas   = np.array(all_areas)
        xlim, bin_w = 300.0, 25.0
        bins        = np.arange(-xlim, xlim + bin_w, bin_w)
        bc          = (bins[:-1] + bins[1:]) / 2
        ih, _       = np.histogram(all_dists[all_dists <  0], bins=bins,
                                   weights=all_areas[all_dists <  0] / 1e6)
        eh, _       = np.histogram(all_dists[all_dists >= 0], bins=bins,
                                   weights=all_areas[all_dists >= 0] / 1e6)
        ax_hist.bar(bc, ih, width=bin_w * 0.88, color="#c0392b", alpha=0.85, label="Intratumoral")
        ax_hist.bar(bc, eh, width=bin_w * 0.88, color="#e67e22", alpha=0.85, label="Extratumoral")
    ax_hist.axvline(0, color="black", linewidth=1.8)
    ax_hist.set_xlim(-300, 300)
    ax_hist.set_xlabel("Signed distance to tumour boundary (µm)", fontsize=10)
    ax_hist.set_ylabel("Necrosis area (mm²)", fontsize=10)
    ax_hist.set_title("Necrosis distribution relative to tumour boundary", fontsize=10, pad=6)
    ax_hist.legend(fontsize=9); ax_hist.grid(axis="y", alpha=0.2)

    cids   = sorted(cluster_df["cluster_id"].tolist())
    x      = np.arange(len(cids))
    zones  = [("#c0392b", "Intratumoral"), ("#2ecc71", "0–50µm"),
              ("#f1c40f", "50–100µm"), ("#95a5a6", ">100µm")]

    def zone_areas(row):
        tot = row.get("necrosis_area_total_um2", 0) or 0
        if tot == 0:
            return [0] * 4
        intra = row.get("necrosis_area_intratumoral_um2", 0) or 0
        p50   = (row.get("necrosis_pct_within_50um",  0) or 0)
        p100  = (row.get("necrosis_pct_within_100um", 0) or 0)
        a50   = tot * p50  / 100.0
        a100  = tot * p100 / 100.0
        return [z * 1e-6 for z in [intra, max(0, a50 - intra),
                                    max(0, a100 - a50), max(0, tot - a100)]]

    bottom = np.zeros(len(cids))
    for zi, (color, label) in enumerate(zones):
        vals = [zone_areas(cluster_df[cluster_df["cluster_id"] == c].iloc[0].to_dict())[zi]
                for c in cids]
        ax_stack.bar(x, vals, bottom=bottom, color=color, label=label, width=0.6)
        bottom += np.array(vals)
    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels([f"C{c}" for c in cids], fontsize=9)
    ax_stack.set_xlabel("Cluster ID", fontsize=10)
    ax_stack.set_ylabel("Necrosis area (mm²)", fontsize=10)
    ax_stack.set_title("Necrosis by proximity zone (per cluster)", fontsize=10, pad=6)
    ax_stack.legend(fontsize=7.5, framealpha=0.9); ax_stack.grid(axis="y", alpha=0.2)

    fig.suptitle("Necrosis proximity — distance analysis", fontsize=12, fontweight="bold", y=0.97)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


def plot_tissue_context_figure(
    cluster_df: pd.DataFrame,
    cfg:        _NecrosisCfg,
    out_png:    Path,
) -> None:
    fig = plt.figure(figsize=(14, 6), facecolor="white")
    gs  = gridspec.GridSpec(1, 2, figure=fig, left=0.08, right=0.97,
                            top=0.88, bottom=0.14, wspace=0.36)
    ax_coup  = fig.add_subplot(gs[0, 0])
    ax_pheno = fig.add_subplot(gs[0, 1])

    cids = sorted(cluster_df["cluster_id"].tolist())
    x    = np.arange(len(cids))

    if "necrosis_immune_coupling_index" in cluster_df.columns:
        ic_vals = [
            float(cluster_df[cluster_df["cluster_id"] == c]["necrosis_immune_coupling_index"].iloc[0])
            if not cluster_df[cluster_df["cluster_id"] == c].empty else 0
            for c in cids
        ]
        ax_coup.bar(x, ic_vals, color="#8e44ad", alpha=0.85, width=0.6, label="Immune coupling")
    ax_coup.set_xticks(x)
    ax_coup.set_xticklabels([f"C{c}" for c in cids], fontsize=9)
    ax_coup.set_xlabel("Cluster ID", fontsize=10)
    ax_coup.set_ylabel("Immune coupling index", fontsize=10)
    ax_coup.set_title("Necrosis–immune coupling\n(fraction within threshold)", fontsize=10, pad=6)
    ax_coup.legend(fontsize=8); ax_coup.grid(axis="y", alpha=0.2)

    if "necrosis_phenotype" in cluster_df.columns:
        pheno_area  = cluster_df.groupby("necrosis_phenotype")["necrosis_area_total_um2"].sum()
        total_pheno = pheno_area.sum()
        if total_pheno > 0:
            pheno_pct = (pheno_area / total_pheno * 100).sort_values()
            colors    = [PHENOTYPE_COLORS.get(p, "#7f8c8d") for p in pheno_pct.index]
            ax_pheno.barh(range(len(pheno_pct)), pheno_pct.values, color=colors, height=0.6)
            ax_pheno.set_yticks(range(len(pheno_pct)))
            ax_pheno.set_yticklabels(pheno_pct.index.tolist(), fontsize=9)
            ax_pheno.set_xlabel("% necrosis area", fontsize=10)
            ax_pheno.set_title("Necrosis phenotype distribution", fontsize=10, pad=6)
            ax_pheno.grid(axis="x", alpha=0.2)

    fig.suptitle("Necrosis tissue-context analysis", fontsize=12, fontweight="bold", y=0.97)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


# ---------------------------------------------------------------------------
# Internal orchestrator
# ---------------------------------------------------------------------------

def _run_necrosis_proximity(ncfg: _NecrosisCfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ncfg.validate()
    ncfg.outdir.mkdir(parents=True, exist_ok=True)

    clusters    = load_clusters(ncfg.cluster_geojson)
    seg         = load_segmentation(ncfg.segmentation_geojson)
    print(f"  Clusters:             {len(clusters)}")
    print(f"  Segmentation regions: {len(seg)}")

    seg_geoms   = seg["geometry"].tolist()
    seg_classes = seg["class"].to_numpy()
    seg_tree    = STRtree(seg_geoms)

    def query_idx(query_geom):
        hits = seg_tree.query(query_geom)
        if len(hits) == 0:
            return []
        if isinstance(hits[0], (int, np.integer)):
            return [int(i) for i in hits]
        id_map = {id(g): i for i, g in enumerate(seg_geoms)}
        return [id_map[id(g)] for g in hits]

    feature_rows: list = []
    comp_dfs:     list = []

    for _, crow in tqdm(
        clusters.iterrows(), total=len(clusters),
        desc="Computing necrosis proximity", unit="cluster",
    ):
        cid          = int(crow["cluster_id"])
        cluster_geom = crow["geometry"]
        n_roi        = crow.get("n_roi_boxes", np.nan)

        class_parts: Dict[str, List[BaseGeometry]] = {
            "Tumour": [], "Necrosis": [], "Stroma": [], "Inflammatory": [], "Others": [],
        }

        for idx in query_idx(cluster_geom):
            g   = seg_geoms[idx]
            cls = seg_classes[idx]
            if cls not in class_parts:
                continue
            if not cluster_geom.intersects(g):
                continue
            inter = fix_geom(cluster_geom.intersection(g))
            if inter is None or inter.is_empty:
                continue
            for part in polygon_parts(inter):
                if part.area > 0:
                    class_parts[cls].append(part)

        def dissolve(parts):
            if not parts:
                return None
            u = fix_geom(unary_union(parts))
            return u if u is not None and not u.is_empty else None

        tumor_union  = dissolve(class_parts["Tumour"])
        inflam_geoms = [p for p in class_parts["Inflammatory"] if p is not None and not p.is_empty]
        stroma_geoms = [p for p in class_parts["Stroma"]       if p is not None and not p.is_empty]
        necrosis_parts = class_parts["Necrosis"]

        inflam_tree = build_strtree(inflam_geoms)
        stroma_tree = build_strtree(stroma_geoms)

        comp_df = compute_component_table(
            necrosis_parts, tumor_union,
            inflam_geoms, stroma_geoms,
            inflam_tree, stroma_tree, ncfg.mpp,
        )
        comp_dfs.append(comp_df)

        row = extract_cluster_features(
            cid, cluster_geom, n_roi,
            necrosis_parts, tumor_union,
            inflam_geoms, stroma_geoms,
            inflam_tree, stroma_tree,
            ncfg.mpp, ncfg,
        )
        feature_rows.append(row)

    cluster_df = pd.DataFrame(feature_rows).sort_values("cluster_id").reset_index(drop=True)
    wsi_df     = compute_wsi_summary(cluster_df, ncfg)

    cluster_csv = ncfg.outdir / "necrosis_proximity_by_cluster.csv"
    wsi_csv     = ncfg.outdir / "necrosis_proximity_wsi_summary.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    wsi_df.to_csv(wsi_csv, index=False)
    print(f"  Wrote: {cluster_csv.name}")
    print(f"  Wrote: {wsi_csv.name}")

    plot_distance_figure(cluster_df, comp_dfs, ncfg,
                         out_png=ncfg.outdir / "necrosis_distance_figure.png")
    plot_tissue_context_figure(cluster_df, ncfg,
                               out_png=ncfg.outdir / "necrosis_tissue_context_figure.png")

    return cluster_df, wsi_df


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------

def run_necrosis_proximity_features(
    wsi_path: str,
    cfg:      PipelineConfig = None,
) -> dict:
    """
    Run necrosis proximity feature extraction for one WSI.

    Reads
    -----
    cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
        cluster_scoring_polygons.geojson
    cfg.OUT_DIR/<slide>/segmentation/
        segmentation_all_classes.geojson

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/necrosis_proximity/
        necrosis_proximity_by_cluster.csv
        necrosis_proximity_wsi_summary.csv
        necrosis_distance_figure.png
        necrosis_tissue_context_figure.png

    Parameters
    ----------
    wsi_path : str
    cfg      : PipelineConfig
        Knobs (all with fallbacks):
            cfg.MPP
            cfg.NECROSIS_PROXIMITY_THRESHOLDS_UM        (default [50,100])
            cfg.NECROSIS_CONTACT_TOLERANCE_UM           (default 5.0)
            cfg.NECROSIS_IMMUNE_COUPLING_THRESHOLD_UM   (default 100.0)
            cfg.NECROSIS_MIN_COMPONENT_AREA_UM2         (default 500.0)
            cfg.NECROSIS_ABSENT_MAX_AREA_UM2            (default 500.0)
            cfg.NECROSIS_CENTRAL_MIN_INTRA_FRAC         (default 0.50)
            cfg.NECROSIS_PERITUMOURAL_MIN_PCT_100UM     (default 60.0)
            cfg.NECROSIS_IMMUNE_ADJACENT_MIN_COUPLING   (default 0.20)
            cfg.NECROSIS_DISTANT_MIN_MEDIAN_UM          (default 150.0)

    Returns
    -------
    dict : slide_name, cluster_geojson, segmentation_geojson,
           cluster_csv, wsi_csv, distance_png, context_png
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
        / "spatial_feature_results" / "necrosis_proximity"
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

    mpp = getattr(cfg, "MPP", 0.25)

    print(f"\n{'='*55}")
    print(f"  Necrosis proximity features")
    print(f"  Slide    : {slide_name}")
    print(f"  Clusters : {cluster_geojson.name}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*55}")

    ncfg = _NecrosisCfg(
        cluster_geojson=cluster_geojson,
        segmentation_geojson=seg_geojson,
        outdir=out_dir,
        mpp=mpp,
        proximity_thresholds_um=getattr(cfg, "NECROSIS_PROXIMITY_THRESHOLDS_UM", [50.0, 100.0]),
        contact_tolerance_um=getattr(cfg, "NECROSIS_CONTACT_TOLERANCE_UM", 5.0),
        immune_coupling_threshold_um=getattr(cfg, "NECROSIS_IMMUNE_COUPLING_THRESHOLD_UM", 100.0),
        min_component_area_um2=getattr(cfg, "NECROSIS_MIN_COMPONENT_AREA_UM2", 500.0),
        absent_max_area_um2=getattr(cfg, "NECROSIS_ABSENT_MAX_AREA_UM2", 500.0),
        central_min_intra_frac=getattr(cfg, "NECROSIS_CENTRAL_MIN_INTRA_FRAC", 0.50),
        peritumoural_min_pct_within_100um=getattr(cfg, "NECROSIS_PERITUMOURAL_MIN_PCT_100UM", 60.0),
        immune_adjacent_min_coupling=getattr(cfg, "NECROSIS_IMMUNE_ADJACENT_MIN_COUPLING", 0.20),
        distant_min_median_um=getattr(cfg, "NECROSIS_DISTANT_MIN_MEDIAN_UM", 150.0),
    )

    cluster_df, wsi_df = _run_necrosis_proximity(ncfg)

    s = wsi_df.iloc[0]
    print(f"\n  === Necrosis proximity summary ===")
    print(f"  Total necrosis area:          {s['wsi_necrosis_area_total_mm2']:.3f} mm²")
    print(f"  Necrosis fraction of cluster: {s['wsi_necrosis_fraction_of_cluster']:.3f}")
    print(f"  Intratumoral fraction:        {s['wsi_necrosis_fraction_intratumoral']:.3f}")
    print(f"  Immune coupling index:        {s['wsi_necrosis_immune_coupling_index']:.3f}")
    print(f"  Dominant phenotype:           {s['wsi_dominant_necrosis_phenotype']}")
    print(f"\n  Done.")

    return {
        "slide_name":           slide_name,
        "cluster_geojson":      str(cluster_geojson),
        "segmentation_geojson": str(seg_geojson),
        "cluster_csv":          str(out_dir / "necrosis_proximity_by_cluster.csv"),
        "wsi_csv":              str(out_dir / "necrosis_proximity_wsi_summary.csv"),
        "distance_png":         str(out_dir / "necrosis_distance_figure.png"),
        "context_png":          str(out_dir / "necrosis_tissue_context_figure.png"),
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) — every PipelineConfig
# field (including the NECROSIS_* knobs) is available as a flag, plus
# --from-json to pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python necrosis_proximity_features.py --from-json run_config.json

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

    run_necrosis_proximity_features(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()