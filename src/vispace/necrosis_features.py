#!/usr/bin/env python3
"""
necrosis_features.py
====================
Necrosis feature extraction per tumor region from ViSpace GeoJSON output.

Regions are the Master ROI polygons written by the TSR/TIL stage
(cluster_scoring_polygons.geojson), so necrosis is measured over the same
regions, with the same region IDs, as the TSR, immune-proximity and
morphology stages. Necrosis is scored over the whole region, not over the
representative image ROIs (which are chosen to avoid necrosis).

Features produced (per region)
-------------------------------
- necrosis_area_um2       : total necrosis polygon area in µm²
- necrosis_perimeter_um   : total necrosis polygon perimeter in µm
- necrosis_frac           : necrosis area / total tissue area in cluster
- necrosis_phenotype      : absent | focal | present
- tissue_area_um2         : total tissue area in the cluster (context)

Phenotype classification
------------------------
- absent  : no necrosis detected in the cluster (after the noise filter)
- focal   : necrosis fraction < FOCAL_THRESHOLD
- present : necrosis fraction >= FOCAL_THRESHOLD

Necrosis polygons smaller than MIN_FRAGMENT_UM2 (default 500 µm²) are
treated as segmentation noise and dropped before any measurement.

No immune coupling or necrosis-to-immune distance features are computed.

All areas and perimeters are reported in physical units (µm², µm)
using the slide MPP; no pixel² values are written to the output.

Outputs
-------
cfg.OUT_DIR/<slide>/spatial_feature_results/necrosis_feature/
    necrosis_feature_by_cluster.csv   one row per tumor cluster
    necrosis_feature_wsi_summary.csv  one WSI-level summary row
    necrosis_summary_figure.png       area / perimeter / phenotype figure

Pipeline position
-----------------
    tessellate.py → segmenter.py → stitch.py → tumor_roi_overlay.py
        → cluster_tils_tsr_score.py → immune_proximity_features.py
        → necrosis_features.py → tumor_morphology_features.py

    Requires cluster_tils_tsr_score.py output (cluster_scoring_polygons.geojson).

Usage (as a library)
--------------------
    from vispace import run_necrosis_features
    from vispace import cfg

    run_necrosis_features("slides/TCGA-A1-A0SP.svs", cfg)

Usage (from the command line)
-----------------------------
Same shared flags as the rest of the pipeline. `config_from_args()` handles
--wsi-path, --out-dir, --from-json, and all PipelineConfig overrides.

    # minimal
    python -m vispace.necrosis_features \
        --wsi-path slides/TCGA-A1-A0SP.svs \
        --out-dir vipsegd_output

    # continue from saved config
    python -m vispace.necrosis_features --from-json run_config.json

    # see all shared options
    python -m vispace.necrosis_features --help
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from .config import cfg as default_cfg, PipelineConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NECROSIS_CLASS = "Necrosis"
TUMOR_CLASS = "Tumour"
STROMA_CLASS = "Stroma"
INFLAM_CLASS = "Inflammatory"
OTHERS_CLASS = "Others"

# Necrosis fraction threshold separating focal from present.
FOCAL_THRESHOLD = 0.05

# Necrosis fragments below this physical area (µm²) are treated as
# segmentation noise and excluded from all measurements.
MIN_FRAGMENT_UM2 = 500.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_area_um2(poly: Polygon, mpp: float) -> float:
    """Return polygon area in µm²."""
    return abs(poly.area) * (mpp ** 2)


def _polygon_perimeter_um(poly: Polygon, mpp: float) -> float:
    """Return polygon perimeter in µm."""
    return poly.length * mpp


def _load_polygons_by_class(
    geojson_path: str | Path,
    target_class: str,
) -> list[Polygon]:
    """Load all polygons of a given tissue class from a GeoJSON file."""
    with open(geojson_path, encoding="utf-8") as f:
        gj = json.load(f)

    polys: list[Polygon] = []

    for feat in gj.get("features", []):
        props = feat.get("properties", {}) or {}

        classification = props.get("classification") or {}
        cls = classification.get("name") or props.get("class")

        # Normalize tumor spelling if needed.
        if cls in {"Tumor", "tumor", "tumour"}:
            cls = "Tumour"

        if cls != target_class:
            continue

        geom = shape(feat["geometry"])

        if not geom.is_valid:
            geom = geom.buffer(0)

        if geom.is_empty:
            continue

        if isinstance(geom, MultiPolygon):
            polys.extend(
                p for p in geom.geoms
                if isinstance(p, Polygon) and not p.is_empty
            )
        elif isinstance(geom, Polygon):
            polys.append(geom)

    return polys


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def load_region_polygons(path: str | Path) -> dict[int, BaseGeometry]:
    """Load Master ROI polygons (cluster_scoring_polygons.geojson) keyed by region ID."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    regions: dict[int, BaseGeometry] = {}
    for i, feat in enumerate(gj.get("features", [])):
        props = feat.get("properties", {}) or {}
        rid = props.get("master_roi_id", props.get("cluster_id", i))
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        regions[int(rid)] = geom
    return regions


def extract_necrosis_features(
    geojson_path: str | Path,
    regions: dict[int, BaseGeometry],
    mpp: float,
    focal_threshold: float = FOCAL_THRESHOLD,
    min_fragment_um2: float = MIN_FRAGMENT_UM2,
) -> pd.DataFrame:
    """
    Extract necrosis features per tumor cluster.

    Parameters
    ----------
    geojson_path
        Path to segmentation_all_classes.geojson.

    regions
        Region ID -> region polygon (level-0 pixels), normally the Master ROI
        polygons from cluster_scoring_polygons.geojson. Each region is
        scored once over its full area.

    mpp
        Microns per pixel for the slide.

    focal_threshold
        Necrosis fraction below which phenotype is classified as "focal".

    min_fragment_um2
        Necrosis fragments smaller than this (µm²) are dropped as noise.

    Returns
    -------
    pd.DataFrame
        One row per region with columns (``cluster_id`` holds the region ID):

        cluster_id
        necrosis_area_um2
        necrosis_perimeter_um
        necrosis_frac
        necrosis_phenotype
        tissue_area_um2
    """
    geojson_path = Path(geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    if mpp <= 0:
        raise ValueError("mpp must be > 0")

    if not 0 <= focal_threshold <= 1:
        raise ValueError("focal_threshold must be between 0 and 1")

    # Load all tissue-class polygons once.
    necrosis_polys = _load_polygons_by_class(geojson_path, NECROSIS_CLASS)
    tumor_polys = _load_polygons_by_class(geojson_path, TUMOR_CLASS)
    stroma_polys = _load_polygons_by_class(geojson_path, STROMA_CLASS)
    inflam_polys = _load_polygons_by_class(geojson_path, INFLAM_CLASS)
    others_polys = _load_polygons_by_class(geojson_path, OTHERS_CLASS)

    # Drop necrosis fragments below the noise floor (physical area).
    min_fragment_px2 = min_fragment_um2 / (mpp ** 2) if mpp > 0 else 0.0
    necrosis_polys = [
        p for p in necrosis_polys if p.area >= min_fragment_px2
    ]

    all_tissue_polys = (
        necrosis_polys
        + tumor_polys
        + stroma_polys
        + inflam_polys
        + others_polys
    )

    # Spatial indexes so each region only tests nearby polygons.
    necro_tree = STRtree(necrosis_polys) if necrosis_polys else None
    tissue_tree = STRtree(all_tissue_polys) if all_tissue_polys else None

    def _clip(tree, polys, region):
        if tree is None:
            return []
        out = []
        for i in tree.query(region):
            p = polys[int(i)]
            if p.intersects(region):
                q = p.intersection(region)
                if not q.is_empty:
                    out.append(q)
        return out

    rows = []

    for cid, roi_region in regions.items():
        # Necrosis and tissue inside this region.
        necro_in_roi = _clip(necro_tree, necrosis_polys, roi_region)
        tissue_in_roi = _clip(tissue_tree, all_tissue_polys, roi_region)
        tissue_area_px2 = sum(p.area for p in tissue_in_roi)

        # Aggregate necrosis measurements.
        necro_area_px2 = sum(p.area for p in necro_in_roi)
        necro_perim_px = sum(p.length for p in necro_in_roi)

        # Physical units.
        necro_area_um2 = necro_area_px2 * (mpp ** 2)
        necro_perim_um = necro_perim_px * mpp
        tissue_area_um2 = tissue_area_px2 * (mpp ** 2)

        # Fraction of all tissue represented by necrosis.
        if tissue_area_px2 > 0:
            necro_frac = necro_area_px2 / tissue_area_px2
        else:
            necro_frac = 0.0

        # Phenotype.
        if necro_area_px2 <= 0:
            phenotype = "absent"
        elif necro_frac < focal_threshold:
            phenotype = "focal"
        else:
            phenotype = "present"

        rows.append({
            "cluster_id": cid,
            "necrosis_area_um2": round(necro_area_um2, 2),
            "necrosis_perimeter_um": round(necro_perim_um, 2),
            "necrosis_frac": round(necro_frac, 4),
            "necrosis_phenotype": phenotype,
            "tissue_area_um2": round(tissue_area_um2, 2),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("cluster_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# User-facing CSV column names
# ---------------------------------------------------------------------------

# Internal column names -> names written to the CSVs. µm² areas and µm
# lengths are converted to mm² / mm on export.
NECROSIS_REGION_COLUMN_NAMES = {
    "cluster_id":            "region_id",
    "necrosis_area_um2":     "necrosis_area_mm2",
    "necrosis_perimeter_um": "necrosis_boundary_length_mm",
    "necrosis_frac":         "necrosis_tissue_fraction",
    "necrosis_phenotype":    "necrosis_category",
    "tissue_area_um2":       "tissue_area_mm2",
}

NECROSIS_SUMMARY_COLUMN_NAMES = {
    "wsi_n_clusters_scored":            "region_count",
    "wsi_necrosis_area_um2":            "necrosis_area_mm2",
    "wsi_necrosis_perimeter_um":        "necrosis_boundary_length_mm",
    "wsi_necrosis_fraction_of_cluster": "necrosis_tissue_fraction",
    "wsi_dominant_necrosis_phenotype":  "necrosis_category",
}

_UM2_TO_MM2 = {"necrosis_area_um2", "tissue_area_um2", "wsi_necrosis_area_um2"}
_UM_TO_MM = {"necrosis_perimeter_um", "wsi_necrosis_perimeter_um"}


def rename_for_export(df: pd.DataFrame, names: dict) -> pd.DataFrame:
    """Convert µm² / µm to mm² / mm and apply the user-facing column names."""
    d = df.copy()
    for c in _UM2_TO_MM2 & set(d.columns):
        d[c] = pd.to_numeric(d[c], errors="coerce") / 1e6
    for c in _UM_TO_MM & set(d.columns):
        d[c] = pd.to_numeric(d[c], errors="coerce") / 1e3
    return d.rename(columns=names)


# ---------------------------------------------------------------------------
# WSI-level summary
# ---------------------------------------------------------------------------

def summarize_necrosis_wsi(
    cluster_df: pd.DataFrame,
    focal_threshold: float = FOCAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Collapse the per-cluster necrosis table into a single WSI-level row.

    The dominant phenotype is area-weighted: it is the phenotype implied by
    the whole-slide necrosis fraction (total necrosis area over total tissue
    area across all scored clusters), using the same focal threshold.

    Returns
    -------
    pd.DataFrame with one row and columns:
        wsi_n_clusters_scored
        wsi_necrosis_area_um2
        wsi_necrosis_perimeter_um
        wsi_necrosis_fraction_of_cluster
        wsi_dominant_necrosis_phenotype
    """
    if cluster_df.empty:
        return pd.DataFrame([{
            "wsi_n_clusters_scored": 0,
            "wsi_necrosis_area_um2": 0.0,
            "wsi_necrosis_perimeter_um": 0.0,
            "wsi_necrosis_fraction_of_cluster": 0.0,
            "wsi_dominant_necrosis_phenotype": "absent",
        }])

    total_necro_area = float(cluster_df["necrosis_area_um2"].sum())
    total_necro_perim = float(cluster_df["necrosis_perimeter_um"].sum())
    total_tissue_area = float(cluster_df.get(
        "tissue_area_um2", pd.Series(dtype=float)
    ).sum())

    if total_tissue_area > 0:
        wsi_frac = total_necro_area / total_tissue_area
    else:
        wsi_frac = 0.0

    if total_necro_area <= 0:
        dominant = "absent"
    elif wsi_frac < focal_threshold:
        dominant = "focal"
    else:
        dominant = "present"

    return pd.DataFrame([{
        "wsi_n_clusters_scored": int(len(cluster_df)),
        "wsi_necrosis_area_um2": round(total_necro_area, 2),
        "wsi_necrosis_perimeter_um": round(total_necro_perim, 2),
        "wsi_necrosis_fraction_of_cluster": round(wsi_frac, 4),
        "wsi_dominant_necrosis_phenotype": dominant,
    }])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

_PHENOTYPE_ORDER = ["absent", "focal", "present"]
_PHENOTYPE_COLORS = {
    "absent": "#9aa0a6",
    "focal": "#f9ab00",
    "present": "#d93025",
}


def plot_necrosis_summary(
    cluster_df: pd.DataFrame,
    out_png: str | Path,
) -> Path:
    """
    Render a three-panel necrosis summary figure:
        (1) necrosis area per cluster
        (2) necrosis perimeter per cluster
        (3) phenotype distribution across clusters

    Saves to out_png and returns the path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if cluster_df.empty:
        for ax in axes:
            ax.text(
                0.5, 0.5, "No regions scored",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return out_png

    df = cluster_df.sort_values("cluster_id")
    cids = df["cluster_id"].astype(str).tolist()
    bar_colors = [
        _PHENOTYPE_COLORS.get(p, "#9aa0a6")
        for p in df["necrosis_phenotype"]
    ]

    # Panel 1: necrosis area per cluster.
    axes[0].bar(cids, df["necrosis_area_um2"], color=bar_colors)
    axes[0].set_title("Necrosis area per region")
    axes[0].set_xlabel("region_id")
    axes[0].set_ylabel("area (µm²)")

    # Panel 2: necrosis perimeter per cluster.
    axes[1].bar(cids, df["necrosis_perimeter_um"], color=bar_colors)
    axes[1].set_title("Necrosis perimeter per region")
    axes[1].set_xlabel("region_id")
    axes[1].set_ylabel("perimeter (µm)")

    # Panel 3: phenotype distribution.
    counts = (
        df["necrosis_phenotype"]
        .value_counts()
        .reindex(_PHENOTYPE_ORDER, fill_value=0)
    )
    axes[2].bar(
        _PHENOTYPE_ORDER,
        counts.values,
        color=[_PHENOTYPE_COLORS[p] for p in _PHENOTYPE_ORDER],
    )
    axes[2].set_title("Necrosis category")
    axes[2].set_xlabel("category")
    axes[2].set_ylabel("number of regions")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return out_png


# ---------------------------------------------------------------------------
# CSV-level convenience wrapper
# ---------------------------------------------------------------------------

def extract_necrosis_features_from_regions(
    geojson_path: str | Path,
    regions_geojson_path: str | Path,
    mpp: float,
    out_csv: str | Path | None = None,
    focal_threshold: float = FOCAL_THRESHOLD,
    min_fragment_um2: float = MIN_FRAGMENT_UM2,
) -> pd.DataFrame:
    """
    Load region polygons and compute necrosis features per region.

    Parameters
    ----------
    geojson_path
        segmentation_all_classes.geojson

    regions_geojson_path
        cluster_scoring_polygons.geojson from the TSR/TIL stage (one Master
        ROI polygon per region).

    mpp
        Microns per pixel.

    out_csv
        Optional output CSV path for the per-region table.

    focal_threshold
        Fraction separating focal from present.

    min_fragment_um2
        Necrosis noise floor (µm²).
    """
    regions_geojson_path = Path(regions_geojson_path)

    if not regions_geojson_path.exists():
        raise FileNotFoundError(f"Region polygons not found: {regions_geojson_path}")

    regions = load_region_polygons(regions_geojson_path)

    df = extract_necrosis_features(
        geojson_path=geojson_path,
        regions=regions,
        mpp=mpp,
        focal_threshold=focal_threshold,
        min_fragment_um2=min_fragment_um2,
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rename_for_export(df, NECROSIS_REGION_COLUMN_NAMES).to_csv(out_csv, index=False)
        print(f"  Wrote: {out_csv.name}")

    return df


# ---------------------------------------------------------------------------
# Top-level pipeline callable
# ---------------------------------------------------------------------------

def run_necrosis_features(
    wsi_path: str,
    cfg: PipelineConfig = None,
) -> dict:
    """
    Run necrosis feature extraction for one WSI.

    Reads
    -----
    cfg.OUT_DIR/<slide>/segmentation/
        segmentation_all_classes.geojson

    cfg.OUT_DIR/<slide>/spatial_feature_results/cluster_tils_tsr_score/
        cluster_scoring_polygons.geojson

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/necrosis_feature/
        necrosis_feature_by_cluster.csv
        necrosis_feature_wsi_summary.csv
        necrosis_summary_figure.png

    Parameters
    ----------
    wsi_path : str
        Path to WSI.

    cfg : PipelineConfig
        Pipeline configuration.

        Relevant fields:
            cfg.OUT_DIR
            cfg.MPP
            cfg.NECROSIS_FOCAL_THRESHOLD
                Optional; defaults to 0.05.
            cfg.NECROSIS_MIN_FRAGMENT_UM2
                Optional; defaults to 500.0.

    Returns
    -------
    dict
        Paths and metadata for generated output.
    """
    if cfg is None:
        cfg = default_cfg

    slide_name = Path(wsi_path).stem

    segmentation_geojson = (
        Path(cfg.OUT_DIR)
        / slide_name
        / "segmentation"
        / "segmentation_all_classes.geojson"
    )

    regions_geojson = (
        Path(cfg.OUT_DIR)
        / slide_name
        / "spatial_feature_results"
        / "cluster_tils_tsr_score"
        / "cluster_scoring_polygons.geojson"
    )

    out_dir = (
        Path(cfg.OUT_DIR)
        / slide_name
        / "spatial_feature_results"
        / "necrosis_feature"
    )

    out_csv = out_dir / "necrosis_feature_by_cluster.csv"
    summary_csv = out_dir / "necrosis_feature_wsi_summary.csv"
    figure_png = out_dir / "necrosis_summary_figure.png"

    if not segmentation_geojson.exists():
        raise FileNotFoundError(
            f"Segmentation GeoJSON not found: "
            f"{segmentation_geojson}\n"
            f"Run run_stitching(wsi_path, cfg) first."
        )

    if not regions_geojson.exists():
        raise FileNotFoundError(
            f"Region polygons not found: "
            f"{regions_geojson}\n"
            f"Run cluster_tils_tsr_score for this slide first."
        )

    mpp = getattr(cfg, "MPP", 0.25)

    focal_threshold = getattr(
        cfg, "NECROSIS_FOCAL_THRESHOLD", FOCAL_THRESHOLD,
    )
    min_fragment_um2 = getattr(
        cfg, "NECROSIS_MIN_FRAGMENT_UM2", MIN_FRAGMENT_UM2,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 55}")
    print("  Necrosis features")
    print(f"  Slide       : {slide_name}")
    print(f"  Segmentation: {segmentation_geojson.name}")
    print(f"  Regions     : {regions_geojson.name}")
    print(f"  MPP         : {mpp}")
    print(f"  Focal cutoff: {focal_threshold:.3f}")
    print(f"  Noise floor : {min_fragment_um2:.0f} µm²")
    print(f"  Output      : {out_dir}")
    print(f"{'=' * 55}")

    df = extract_necrosis_features_from_regions(
        geojson_path=segmentation_geojson,
        regions_geojson_path=regions_geojson,
        mpp=mpp,
        out_csv=out_csv,
        focal_threshold=focal_threshold,
        min_fragment_um2=min_fragment_um2,
    )

    # WSI-level summary.
    summary_df = summarize_necrosis_wsi(df, focal_threshold=focal_threshold)
    rename_for_export(summary_df, NECROSIS_SUMMARY_COLUMN_NAMES).to_csv(summary_csv, index=False)
    print(f"  Wrote: {summary_csv.name}")

    # Figure.
    plot_necrosis_summary(df, figure_png)
    print(f"  Wrote: {figure_png.name}")

    # Console summary.
    n_clusters = len(df)
    if n_clusters:
        n_absent = int((df["necrosis_phenotype"] == "absent").sum())
        n_focal = int((df["necrosis_phenotype"] == "focal").sum())
        n_present = int((df["necrosis_phenotype"] == "present").sum())
        total_necrosis_um2 = float(df["necrosis_area_um2"].sum())
    else:
        n_absent = n_focal = n_present = 0
        total_necrosis_um2 = 0.0

    print("\n  === Necrosis summary ===")
    print(f"  Regions:               {n_clusters}")
    print(f"  Absent:                {n_absent}")
    print(f"  Focal:                 {n_focal}")
    print(f"  Present:               {n_present}")
    print(
        f"  Total necrosis area:   "
        f"{total_necrosis_um2 / 1e6:.3f} mm²"
    )
    print("\n  Done.")

    return {
        "slide_name": str(slide_name),
        "segmentation_geojson": str(segmentation_geojson),
        "regions_geojson": str(regions_geojson),
        "cluster_csv": str(out_csv),
        "wsi_csv": str(summary_csv),
        "figure_png": str(figure_png),
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
#
# Reuses config.py's full CLI exactly like immune_proximity_features.py.
#
# Examples:
#
#   python -m vispace.necrosis_features \
#       --wsi-path slides/TCGA-A1-A0SP.svs \
#       --out-dir vipsegd_output
#
#   python -m vispace.necrosis_features --from-json run_config.json
#
# If NECROSIS_FOCAL_THRESHOLD / NECROSIS_MIN_FRAGMENT_UM2 are added to
# PipelineConfig/config_from_args, they are automatically available through
# the shared configuration CLI.

def main(argv=None) -> None:
    from .config import config_from_args

    cfg, _ = config_from_args(argv)  # handles --from-json, per-field overrides, etc.

    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit(
            "--wsi-path is required (path to a .svs / .tif slide), "
            "either directly or via --from-json"
        )

    slide_name = Path(cfg.WSI_PATH).stem
    regions_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "spatial_feature_results" / "cluster_tils_tsr_score"
        / "cluster_scoring_polygons.geojson"
    )
    seg_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    if not regions_geojson.exists():
        raise SystemExit(
            f"Region polygons not found: {regions_geojson}. "
            f"Run cluster_tils_tsr_score for this slide (with the same --out-dir) first."
        )
    if not seg_geojson.exists():
        raise SystemExit(
            f"Segmentation GeoJSON not found: {seg_geojson}. "
            f"Run stitch.py for this slide (with the same --out-dir) first."
        )

    run_necrosis_features(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
