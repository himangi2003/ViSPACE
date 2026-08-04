#!/usr/bin/env python3
"""
immune_proximity_features.py
============================
Stage 6 of the ViSpace pipeline — immune / TIL proximity feature extraction.

Reads cluster polygons produced by run_cluster_tils_tsr_score() (stage 5)
and the segmentation GeoJSON produced by run_stitching() (stage 3).

Output directory
----------------
    cfg.OUT_DIR/<slide>/spatial_feature_results/immune_proximity/
        immune_proximity_by_cluster.csv
        immune_proximity_wsi_summary.csv
        immune_proximity_plot.png

Pipeline position
-----------------
    tessellate.py → segmenter.py → stitch.py → tumor_roi_overlay.py
        → cluster_tils_tsr_score.py → immune_proximity_features.py
        → necrosis_proximity_features.py → tumor_morphology_features.py

Usage (as a library)
---------------------
    from vispace import run_immune_proximity_features
    from vispace import cfg

    run_immune_proximity_features("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
------------------------------
Same shared flags as the rest of the pipeline — every PipelineConfig field
is available here too, including the IMMUNE_* knobs. Also supports
--from-json to pick up a config saved earlier via `config.py --print-config`.

    # minimal — requires cluster_tils_tsr_score.py to have already run
    python immune_proximity_features.py --wsi-path slides/TCGA-A1-A0SP.svs \\
        --out-dir vipsegd_output

    # continue from a config saved earlier
    python immune_proximity_features.py --from-json run_config.json

    # continue from a saved config but override one knob
    python immune_proximity_features.py --from-json run_config.json \\
        --immune-contact-tolerance-um 10

    # see every available flag
    python immune_proximity_features.py --help
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

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
# Internal config dataclass
# ---------------------------------------------------------------------------

@dataclass
class _ImmuneCfg:
    cluster_geojson:          Path
    segmentation_geojson:     Path
    outdir:                   Path  = field(default_factory=lambda: Path("immune_proximity_output"))
    mpp:                      float = 0.25
    proximity_thresholds_um:  List[float] = field(default_factory=lambda: [20, 50, 100, 200])
    contact_tolerance_um:     float = 5.0
    # Phenotype thresholds
    desert_max_area_um2:       float = 1_000.0
    penetrated_min_intra_frac: float = 0.30
    margin_min_pct_within_50um: float = 50.0
    excluded_min_median_um:    float = 100.0
    peritumoral_max_median_um: float = 100.0

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

def fix_geom(g: BaseGeometry) -> BaseGeometry:
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


# ---------------------------------------------------------------------------
# Loading  (unchanged)
# ---------------------------------------------------------------------------

def load_clusters(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for i, feat in enumerate(data.get("features", [])):
        props = feat.get("properties", {}) or {}
        g = fix_geom(shape(feat["geometry"]))
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
    rows = []
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
# Core computation  (unchanged)
# ---------------------------------------------------------------------------

def compute_component_distances(
    inflam_parts: List[BaseGeometry],
    tumor_union:  Optional[BaseGeometry],
    mpp:          float,
) -> pd.DataFrame:
    rows = []
    tumor_empty = tumor_union is None or tumor_union.is_empty
    if not tumor_empty:
        tumor_boundary = tumor_union.boundary

    for j, comp in enumerate(inflam_parts):
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

        rows.append({
            "component_id":   j,
            "area_um2":       area_um2,
            "signed_dist_um": signed_um,
            "abs_dist_um":    abs(signed_um) if np.isfinite(signed_um) else np.nan,
            "compartment":    compartment,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["component_id", "area_um2", "signed_dist_um", "abs_dist_um", "compartment"]
    )


def classify_phenotype(
    total_area_um2:  float,
    intra_frac:      float,
    pct_within_50:   float,
    median_extra_um: float,
    cfg:             _ImmuneCfg,
) -> str:
    if not np.isfinite(total_area_um2) or total_area_um2 < cfg.desert_max_area_um2:
        return "immune-desert"
    if np.isfinite(intra_frac) and intra_frac >= cfg.penetrated_min_intra_frac:
        return "immune-penetrated"
    if np.isfinite(pct_within_50) and pct_within_50 >= cfg.margin_min_pct_within_50um:
        return "margin-localized"
    if np.isfinite(median_extra_um) and median_extra_um >= cfg.excluded_min_median_um:
        return "immune-excluded"
    if np.isfinite(median_extra_um) and median_extra_um < cfg.peritumoral_max_median_um:
        return "peritumoral"
    return "indeterminate"


def extract_cluster_features(
    cluster_id:   int,
    cluster_geom: BaseGeometry,
    n_roi_boxes,
    inflam_parts: List[BaseGeometry],
    tumor_union:  Optional[BaseGeometry],
    mpp:          float,
    cfg:          _ImmuneCfg,
) -> dict:
    cluster_area_px2 = float(cluster_geom.area)
    cluster_area_um2 = px2_to_um2(cluster_area_px2, mpp)

    comp_df = compute_component_distances(inflam_parts, tumor_union, mpp)

    if comp_df.empty or comp_df["area_um2"].sum() == 0:
        row = {
            "cluster_id":                              cluster_id,
            "n_roi_boxes":                             n_roi_boxes,
            "cluster_area_um2":                        cluster_area_um2,
            "cluster_area_mm2":                        cluster_area_um2 / 1e6,
            "til_area_total_um2":                      0.0,
            "til_fraction_of_cluster":                 0.0,
            "n_til_components":                        0,
            "til_area_cv":                             np.nan,
            "til_area_intratumoral_um2":               0.0,
            "til_fraction_intratumoral":               np.nan,
            "til_area_extratumoral_um2":               0.0,
            "til_fraction_extratumoral":               np.nan,
            "til_extratumoral_distance_aw_median_um":  np.nan,
            "til_contact_area_um2":                    0.0,
            "til_contact_fraction":                    np.nan,
        }
        for t in cfg.proximity_thresholds_um:
            row[f"til_pct_within_{int(t)}um"] = np.nan
        row["immune_phenotype"] = "immune-desert"
        return row

    total_area_um2 = float(comp_df["area_um2"].sum())
    n_comp         = int(len(comp_df))
    area_cv        = float(comp_df["area_um2"].std(ddof=1) / comp_df["area_um2"].mean()) \
                     if n_comp > 1 else 0.0

    intra_mask  = comp_df["compartment"] == "intratumoral"
    extra_mask  = comp_df["compartment"] == "extratumoral"
    intra_area  = float(comp_df.loc[intra_mask, "area_um2"].sum())
    extra_area  = float(comp_df.loc[extra_mask, "area_um2"].sum())

    extra_df    = comp_df[extra_mask].dropna(subset=["abs_dist_um"])
    aw_med      = (
        weighted_median(
            extra_df["abs_dist_um"].to_numpy(float),
            extra_df["area_um2"].to_numpy(float),
        )
        if not extra_df.empty and extra_df["area_um2"].sum() > 0 else np.nan
    )

    contact_mask = comp_df["abs_dist_um"] <= cfg.contact_tolerance_um
    contact_area = float(comp_df.loc[contact_mask, "area_um2"].sum())

    pcts = {}
    for t in cfg.proximity_thresholds_um:
        within_mask = comp_df["abs_dist_um"] <= t
        area_within = float(comp_df.loc[within_mask, "area_um2"].sum())
        pcts[f"til_pct_within_{int(t)}um"] = 100.0 * safe_div(area_within, total_area_um2)

    pct_50    = pcts.get("til_pct_within_50um", np.nan)
    phenotype = classify_phenotype(
        total_area_um2, safe_div(intra_area, total_area_um2), pct_50, aw_med, cfg
    )

    row = {
        "cluster_id":                              cluster_id,
        "n_roi_boxes":                             n_roi_boxes,
        "cluster_area_um2":                        cluster_area_um2,
        "cluster_area_mm2":                        cluster_area_um2 / 1e6,
        "til_area_total_um2":                      total_area_um2,
        "til_fraction_of_cluster":                 safe_div(total_area_um2, cluster_area_um2),
        "n_til_components":                        n_comp,
        "til_area_cv":                             area_cv,
        "til_area_intratumoral_um2":               intra_area,
        "til_fraction_intratumoral":               safe_div(intra_area, total_area_um2),
        "til_area_extratumoral_um2":               extra_area,
        "til_fraction_extratumoral":               safe_div(extra_area, total_area_um2),
        "til_extratumoral_distance_aw_median_um":  aw_med,
        "til_contact_area_um2":                    contact_area,
        "til_contact_fraction":                    safe_div(contact_area, total_area_um2),
        "immune_phenotype":                        phenotype,
    }
    row.update(pcts)
    return row


def compute_wsi_summary(cluster_df: pd.DataFrame, cfg: _ImmuneCfg) -> pd.DataFrame:
    total_cluster_um2 = float(cluster_df["cluster_area_um2"].sum())
    total_til_um2     = float(cluster_df["til_area_total_um2"].sum())
    total_intra_um2   = float(cluster_df["til_area_intratumoral_um2"].sum())
    total_extra_um2   = float(cluster_df["til_area_extratumoral_um2"].sum())
    total_contact_um2 = float(cluster_df["til_contact_area_um2"].sum())

    out = {
        "n_clusters":                             int(len(cluster_df)),
        "total_cluster_area_mm2":                 total_cluster_um2 / 1e6,
        "wsi_til_area_total_um2":                 total_til_um2,
        "wsi_til_area_total_mm2":                 total_til_um2 / 1e6,
        "wsi_til_fraction_of_cluster":            safe_div(total_til_um2, total_cluster_um2),
        "wsi_n_til_components":                   int(cluster_df["n_til_components"].sum()),
        "wsi_til_area_intratumoral_um2":          total_intra_um2,
        "wsi_til_fraction_intratumoral":          safe_div(total_intra_um2, total_til_um2),
        "wsi_til_area_extratumoral_um2":          total_extra_um2,
        "wsi_til_fraction_extratumoral":          safe_div(total_extra_um2, total_til_um2),
        "wsi_til_contact_area_um2":               total_contact_um2,
        "wsi_til_contact_fraction":               safe_div(total_contact_um2, total_til_um2),
    }

    for t in cfg.proximity_thresholds_um:
        col = f"til_pct_within_{int(t)}um"
        if col in cluster_df.columns:
            area_within = cluster_df[col].fillna(0) / 100.0 * cluster_df["til_area_total_um2"]
            out[f"wsi_til_pct_within_{int(t)}um"] = 100.0 * safe_div(
                float(area_within.sum()), total_til_um2
            )

    valid = cluster_df[cluster_df["til_extratumoral_distance_aw_median_um"].notna()].copy()
    out["wsi_til_extratumoral_distance_aw_median_um"] = (
        weighted_median(
            valid["til_extratumoral_distance_aw_median_um"].to_numpy(float),
            valid["til_area_extratumoral_um2"].to_numpy(float),
        )
        if not valid.empty else np.nan
    )

    if "immune_phenotype" in cluster_df.columns and total_til_um2 > 0:
        pheno_area = cluster_df.groupby("immune_phenotype")["til_area_total_um2"].sum()
        out["wsi_dominant_immune_phenotype"] = str(pheno_area.idxmax())
    else:
        out["wsi_dominant_immune_phenotype"] = "NA"

    out["contact_tolerance_um_used"] = cfg.contact_tolerance_um
    out["proximity_thresholds_um"]   = ";".join(str(int(t)) for t in cfg.proximity_thresholds_um)

    return pd.DataFrame([out])


# ---------------------------------------------------------------------------
# Plot  (unchanged rendering logic)
# ---------------------------------------------------------------------------

PHENOTYPE_COLORS = {
    "immune-desert":     "#aaaaaa",
    "immune-penetrated": "#1a7f2e",
    "margin-localized":  "#e8a020",
    "immune-excluded":   "#c0392b",
    "peritumoral":       "#2980b9",
    "indeterminate":     "#7f8c8d",
}


def plot_immune_proximity(
    cluster_df:  pd.DataFrame,
    comp_dfs:    List[pd.DataFrame],
    cluster_ids: List[int],
    cfg:         _ImmuneCfg,
    out_png:     Path,
) -> None:
    fig = plt.figure(figsize=(18, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97,
                            top=0.93, bottom=0.10,
                            hspace=0.42, wspace=0.32)
    ax_hist  = fig.add_subplot(gs[0, :])
    ax_stack = fig.add_subplot(gs[1, 0])
    ax_pheno = fig.add_subplot(gs[1, 1])

    all_dists, all_areas = [], []
    for comp_df in comp_dfs:
        if comp_df is not None and not comp_df.empty:
            valid = comp_df.dropna(subset=["signed_dist_um", "area_um2"])
            all_dists.extend(valid["signed_dist_um"].tolist())
            all_areas.extend(valid["area_um2"].tolist())

    if all_dists:
        all_dists   = np.array(all_dists)
        all_areas   = np.array(all_areas)
        xlim, bin_w = 300.0, 25.0
        bins        = np.arange(-xlim, xlim + bin_w, bin_w)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        intra_mask  = all_dists < 0
        extra_mask  = all_dists >= 0
        intra_hist, _ = np.histogram(all_dists[intra_mask], bins=bins,
                                     weights=all_areas[intra_mask] / 1e6)
        extra_hist, _ = np.histogram(all_dists[extra_mask], bins=bins,
                                     weights=all_areas[extra_mask] / 1e6)
        ax_hist.bar(bin_centers, intra_hist, width=bin_w * 0.88,
                    color="#3498db", alpha=0.85, label="Intratumoral")
        ax_hist.bar(bin_centers, extra_hist, width=bin_w * 0.88,
                    color="#e67e22", alpha=0.85, label="Extratumoral")
        ax2 = ax_hist.twinx()
        total_area = all_areas.sum()
        cum_pcts = []
        for b in bins[1:]:
            within = (all_areas[np.abs(all_dists) <= abs(b)].sum() if b >= 0
                      else all_areas[all_dists <= b].sum())
            cum_pcts.append(100.0 * within / total_area if total_area > 0 else 0)
        ax2.plot(bins[1:], cum_pcts, color="black", linewidth=2,
                 linestyle="--", label="Cumulative %")
        ax2.set_ylabel("Cumulative TIL area (%)", fontsize=10)
        ax2.set_ylim(0, 110)
        ax2.legend(loc="upper right", fontsize=8)

    ax_hist.axvline(0, color="black", linewidth=1.8)
    ax_hist.axvline(-50, color="#3498db", linewidth=1, linestyle=":")
    ax_hist.axvline( 50, color="#e67e22", linewidth=1, linestyle=":")
    ax_hist.set_xlim(-300, 300)
    ax_hist.set_xlabel(
        "Signed distance to tumour boundary (µm)\n"
        "← negative = inside tumour     positive = outside tumour →", fontsize=10)
    ax_hist.set_ylabel("TIL area (mm²)", fontsize=10)
    ax_hist.set_title(
        "Panel A — TIL distribution relative to tumour boundary\n"
        "Blue: intratumoral   |   Orange: extratumoral", fontsize=10, pad=6)
    ax_hist.legend(loc="upper left", fontsize=9)
    ax_hist.grid(axis="y", alpha=0.2)

    cids   = sorted(cluster_df["cluster_id"].tolist())
    x      = np.arange(len(cids))
    zones  = [
        ("#3498db", "Intratumoral"), ("#2ecc71", "0–50 µm"),
        ("#f1c40f", "50–100 µm"),   ("#e67e22", "100–200 µm"),
        ("#e74c3c", ">200 µm"),
    ]

    def zone_areas(row):
        tot = row.get("til_area_total_um2", 0) or 0
        if tot == 0:
            return [0] * 5
        intra = row.get("til_area_intratumoral_um2", 0) or 0
        p50  = (row.get("til_pct_within_50um",  0) or 0)
        p100 = (row.get("til_pct_within_100um", 0) or 0)
        p200 = (row.get("til_pct_within_200um", 0) or 0)
        a50  = tot * p50  / 100.0
        a100 = tot * p100 / 100.0
        a200 = tot * p200 / 100.0
        return [z * 1e-6 for z in [intra, max(0, a50 - intra),
                                    max(0, a100 - a50), max(0, a200 - a100),
                                    max(0, tot  - a200)]]

    bottom = np.zeros(len(cids))
    for zone_idx, (color, label) in enumerate(zones):
        vals = [zone_areas(
            cluster_df[cluster_df["cluster_id"] == cid].iloc[0].to_dict()
        )[zone_idx] for cid in cids]
        ax_stack.bar(x, vals, bottom=bottom, color=color, label=label, width=0.6)
        bottom += np.array(vals)

    ax_stack.set_xticks(x)
    ax_stack.set_xticklabels([f"C{c}" for c in cids], fontsize=9)
    ax_stack.set_xlabel("Cluster ID", fontsize=10)
    ax_stack.set_ylabel("TIL area (mm²)", fontsize=10)
    ax_stack.set_title("Panel B — TIL area by proximity zone\n(per cluster)",
                       fontsize=10, pad=6)
    ax_stack.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    ax_stack.grid(axis="y", alpha=0.2)

    if "immune_phenotype" in cluster_df.columns:
        pheno_area  = cluster_df.groupby("immune_phenotype")["til_area_total_um2"].sum()
        total_pheno = pheno_area.sum()
        pheno_pct   = (pheno_area / total_pheno * 100).sort_values(ascending=True)
        colors      = [PHENOTYPE_COLORS.get(p, "#7f8c8d") for p in pheno_pct.index]
        bars        = ax_pheno.barh(range(len(pheno_pct)), pheno_pct.values,
                                    color=colors, height=0.6)
        ax_pheno.set_yticks(range(len(pheno_pct)))
        ax_pheno.set_yticklabels(pheno_pct.index.tolist(), fontsize=9)
        ax_pheno.set_xlabel("% of total TIL area", fontsize=10)
        ax_pheno.set_title("Panel C — Immune phenotype distribution\n(by % TIL area)",
                           fontsize=10, pad=6)
        for bar, val in zip(bars, pheno_pct.values):
            ax_pheno.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                          f"{val:.1f}%", va="center", fontsize=8)
        ax_pheno.set_xlim(0, max(pheno_pct.values) * 1.25 if len(pheno_pct) else 100)
        ax_pheno.grid(axis="x", alpha=0.2)

    fig.suptitle("Immune proximity analysis — TIL spatial relationship to tumour boundary",
                 fontsize=13, fontweight="bold", y=0.97)
    fig.savefig(out_png, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {out_png.name}")


# ---------------------------------------------------------------------------
# Internal orchestrator
# ---------------------------------------------------------------------------

def _run_immune_proximity(icfg: _ImmuneCfg) -> Tuple[pd.DataFrame, pd.DataFrame]:
    icfg.validate()
    icfg.outdir.mkdir(parents=True, exist_ok=True)

    clusters    = load_clusters(icfg.cluster_geojson)
    seg         = load_segmentation(icfg.segmentation_geojson)
    print(f"  Clusters:             {len(clusters)}")
    print(f"  Segmentation regions: {len(seg)}")

    seg_geoms   = seg["geometry"].tolist()
    seg_classes = seg["class"].to_numpy()
    tree        = STRtree(seg_geoms)

    def query_idx(query_geom):
        hits = tree.query(query_geom)
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
        desc="Computing immune proximity", unit="cluster",
    ):
        cid          = int(crow["cluster_id"])
        cluster_geom = crow["geometry"]
        n_roi        = crow.get("n_roi_boxes", np.nan)

        tumor_parts  = []
        inflam_parts = []

        for idx in query_idx(cluster_geom):
            g   = seg_geoms[idx]
            cls = seg_classes[idx]
            if not cluster_geom.intersects(g):
                continue
            inter = fix_geom(cluster_geom.intersection(g))
            if inter is None or inter.is_empty:
                continue
            for part in polygon_parts(inter):
                if part.area <= 0:
                    continue
                if cls == "Tumour":
                    tumor_parts.append(part)
                elif cls == "Inflammatory":
                    inflam_parts.append(part)

        tumor_union = None
        if tumor_parts:
            u = fix_geom(unary_union(tumor_parts))
            if u is not None and not u.is_empty:
                tumor_union = u

        comp_df = compute_component_distances(inflam_parts, tumor_union, icfg.mpp)
        comp_dfs.append(comp_df)

        row = extract_cluster_features(
            cid, cluster_geom, n_roi, inflam_parts, tumor_union, icfg.mpp, icfg
        )
        feature_rows.append(row)

    cluster_df = pd.DataFrame(feature_rows).sort_values("cluster_id").reset_index(drop=True)
    wsi_df     = compute_wsi_summary(cluster_df, icfg)

    cluster_csv = icfg.outdir / "immune_proximity_by_cluster.csv"
    wsi_csv     = icfg.outdir / "immune_proximity_wsi_summary.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    wsi_df.to_csv(wsi_csv, index=False)
    print(f"  Wrote: {cluster_csv.name}")
    print(f"  Wrote: {wsi_csv.name}")

    plot_immune_proximity(
        cluster_df, comp_dfs,
        cluster_df["cluster_id"].tolist(),
        icfg,
        icfg.outdir / "immune_proximity_plot.png",
    )

    return cluster_df, wsi_df


# ---------------------------------------------------------------------------
# Top-level callable
# ---------------------------------------------------------------------------

def run_immune_proximity_features(
    wsi_path: str,
    cfg:      PipelineConfig = None,
) -> dict:
    """
    Run immune / TIL proximity feature extraction for one WSI.

    Reads
    -----
    cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
        cluster_scoring_polygons.geojson
    cfg.OUT_DIR/<slide>/segmentation/
        segmentation_all_classes.geojson

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/immune_proximity/
        immune_proximity_by_cluster.csv
        immune_proximity_wsi_summary.csv
        immune_proximity_plot.png

    Parameters
    ----------
    wsi_path : str
    cfg      : PipelineConfig
        Knobs (all with fallbacks):
            cfg.MPP
            cfg.IMMUNE_PROXIMITY_THRESHOLDS_UM   (default [20,50,100,200])
            cfg.IMMUNE_CONTACT_TOLERANCE_UM      (default 5.0)
            cfg.IMMUNE_DESERT_MAX_AREA_UM2       (default 1000.0)
            cfg.IMMUNE_PENETRATED_MIN_INTRA_FRAC (default 0.30)
            cfg.IMMUNE_MARGIN_MIN_PCT_50UM       (default 50.0)
            cfg.IMMUNE_EXCLUDED_MIN_MEDIAN_UM    (default 100.0)
            cfg.IMMUNE_PERITUMORAL_MAX_MEDIAN_UM (default 100.0)

    Returns
    -------
    dict : slide_name, cluster_geojson, segmentation_geojson,
           cluster_csv, wsi_csv, plot_png
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
        / "spatial_feature_results" / "immune_proximity"
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
    print(f"  Immune proximity features")
    print(f"  Slide    : {slide_name}")
    print(f"  Clusters : {cluster_geojson.name}")
    print(f"  Output   : {out_dir}")
    print(f"{'='*55}")

    icfg = _ImmuneCfg(
        cluster_geojson=cluster_geojson,
        segmentation_geojson=seg_geojson,
        outdir=out_dir,
        mpp=mpp,
        proximity_thresholds_um=getattr(cfg, "IMMUNE_PROXIMITY_THRESHOLDS_UM", [20, 50, 100, 200]),
        contact_tolerance_um=getattr(cfg, "IMMUNE_CONTACT_TOLERANCE_UM", 5.0),
        desert_max_area_um2=getattr(cfg, "IMMUNE_DESERT_MAX_AREA_UM2", 1_000.0),
        penetrated_min_intra_frac=getattr(cfg, "IMMUNE_PENETRATED_MIN_INTRA_FRAC", 0.30),
        margin_min_pct_within_50um=getattr(cfg, "IMMUNE_MARGIN_MIN_PCT_50UM", 50.0),
        excluded_min_median_um=getattr(cfg, "IMMUNE_EXCLUDED_MIN_MEDIAN_UM", 100.0),
        peritumoral_max_median_um=getattr(cfg, "IMMUNE_PERITUMORAL_MAX_MEDIAN_UM", 100.0),
    )

    cluster_df, wsi_df = _run_immune_proximity(icfg)

    s = wsi_df.iloc[0]
    print(f"\n  === Immune proximity summary ===")
    print(f"  Total TIL area:          {s['wsi_til_area_total_mm2']:.3f} mm²")
    print(f"  TIL fraction of cluster: {s['wsi_til_fraction_of_cluster']:.3f}")
    print(f"  Intratumoral fraction:   {s['wsi_til_fraction_intratumoral']:.3f}")
    print(f"  Median extratumoral dist:{s['wsi_til_extratumoral_distance_aw_median_um']:.1f} µm")
    print(f"  Dominant phenotype:      {s['wsi_dominant_immune_phenotype']}")
    print(f"\n  Done.")

    return {
        "slide_name":           slide_name,
        "cluster_geojson":      str(cluster_geojson),
        "segmentation_geojson": str(seg_geojson),
        "cluster_csv":          str(out_dir / "immune_proximity_by_cluster.csv"),
        "wsi_csv":              str(out_dir / "immune_proximity_wsi_summary.csv"),
        "plot_png":             str(out_dir / "immune_proximity_plot.png"),
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
# Reuses config.py's full CLI (config_from_args) — every PipelineConfig
# field (including the IMMUNE_* knobs) is available as a flag, plus
# --from-json to pick up a config saved earlier via:
#
#     python config.py --print-config > run_config.json
#     python immune_proximity_features.py --from-json run_config.json

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

    run_immune_proximity_features(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()