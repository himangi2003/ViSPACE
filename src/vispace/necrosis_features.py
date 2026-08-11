#!/usr/bin/env python3
"""
necrosis_proximity_features.py
==============================
Necrosis feature extraction per tumour cluster from ViSpace GeoJSON output.

Features produced (per cluster)
-------------------------------
- necrosis_area_um2       : total necrosis polygon area in µm²
- necrosis_perimeter_um   : total necrosis polygon perimeter in µm
- necrosis_frac           : necrosis area / total tissue area in cluster
- necrosis_phenotype      : absent | focal | present

Phenotype classification
------------------------
- absent  : no necrosis detected in the cluster
- focal   : necrosis fraction < FOCAL_THRESHOLD
- present : necrosis fraction >= FOCAL_THRESHOLD

No immune coupling or necrosis-to-immune distance features are computed.

All distances and areas are reported in physical units (µm, µm²)
using the slide MPP; no pixel² values are written to the output.

Pipeline position
-----------------
    tessellate.py → segmenter.py → stitch.py → tumor_roi_overlay.py
        → cluster_tils_tsr_score.py → immune_proximity_features.py
        → necrosis_proximity_features.py → tumor_morphology_features.py

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
    python necrosis_proximity_features.py \
        --wsi-path slides/TCGA-A1-A0SP.svs \
        --out-dir vipsegd_output

    # continue from saved config
    python necrosis_proximity_features.py --from-json run_config.json

    # see all shared options
    python necrosis_proximity_features.py --help
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, shape

from .config import cfg as default_cfg, PipelineConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NECROSIS_CLASS = "Necrosis"
TUMOUR_CLASS = "Tumour"
STROMA_CLASS = "Stroma"
INFLAM_CLASS = "Inflammatory"
OTHERS_CLASS = "Others"

# Necrosis fraction threshold separating focal from present
FOCAL_THRESHOLD = 0.05


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

        # Normalise tumour spelling if needed
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

def extract_necrosis_features(
    geojson_path: str | Path,
    cluster_roi_boxes: list[dict],
    mpp: float,
    focal_threshold: float = FOCAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Extract necrosis features per tumour cluster.

    Parameters
    ----------
    geojson_path
        Path to segmentation_all_classes.geojson.

    cluster_roi_boxes
        List of dictionaries containing:
            cluster_id, minx, miny, maxx, maxy

        Typically produced by tumor_roi_overlay.

    mpp
        Microns per pixel for the slide.

    focal_threshold
        Necrosis fraction below which phenotype is classified as "focal".

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns:

        cluster_id
        necrosis_area_um2
        necrosis_perimeter_um
        necrosis_frac
        necrosis_phenotype
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    geojson_path = Path(geojson_path)

    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    if mpp <= 0:
        raise ValueError("mpp must be > 0")

    if not 0 <= focal_threshold <= 1:
        raise ValueError("focal_threshold must be between 0 and 1")

    # Load all tissue-class polygons once.
    necrosis_polys = _load_polygons_by_class(
        geojson_path, NECROSIS_CLASS
    )
    tumour_polys = _load_polygons_by_class(
        geojson_path, TUMOUR_CLASS
    )
    stroma_polys = _load_polygons_by_class(
        geojson_path, STROMA_CLASS
    )
    inflam_polys = _load_polygons_by_class(
        geojson_path, INFLAM_CLASS
    )
    others_polys = _load_polygons_by_class(
        geojson_path, OTHERS_CLASS
    )

    all_tissue_polys = (
        necrosis_polys
        + tumour_polys
        + stroma_polys
        + inflam_polys
        + others_polys
    )

    # A single tumour cluster is tiled by one or more non-overlapping ROI
    # boxes in tumor_roi_boxes.csv, so group the boxes by cluster_id and
    # score each cluster once. Unioning a cluster's boxes into a single
    # region (rather than looping box-by-box) both collapses the output to
    # one row per cluster and avoids double-counting necrosis perimeter
    # along shared internal box edges.
    boxes_by_cluster: dict = {}
    for roi in cluster_roi_boxes:
        boxes_by_cluster.setdefault(roi["cluster_id"], []).append(
            shapely_box(
                roi["minx"],
                roi["miny"],
                roi["maxx"],
                roi["maxy"],
            )
        )

    rows = []

    for cid, cluster_boxes in boxes_by_cluster.items():
        roi_region = unary_union(cluster_boxes)

        # Necrosis intersecting this cluster region.
        necro_in_roi = [
            p.intersection(roi_region)
            for p in necrosis_polys
            if p.intersects(roi_region)
        ]
        necro_in_roi = [
            p for p in necro_in_roi
            if not p.is_empty
        ]

        # Total tissue area in cluster.
        tissue_in_roi = [
            p.intersection(roi_region)
            for p in all_tissue_polys
            if p.intersects(roi_region)
        ]
        tissue_in_roi = [
            p for p in tissue_in_roi
            if not p.is_empty
        ]

        tissue_area_px2 = sum(
            p.area for p in tissue_in_roi
        )

        # Aggregate necrosis measurements.
        necro_area_px2 = sum(
            p.area for p in necro_in_roi
        )
        necro_perim_px = sum(
            p.length for p in necro_in_roi
        )

        # Physical units.
        necro_area_um2 = necro_area_px2 * (mpp ** 2)
        necro_perim_um = necro_perim_px * mpp

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
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CSV-level convenience wrapper
# ---------------------------------------------------------------------------

def extract_necrosis_features_from_csv(
    geojson_path: str | Path,
    roi_csv_path: str | Path,
    mpp: float,
    out_csv: str | Path | None = None,
    focal_threshold: float = FOCAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Load cluster ROI boxes from CSV and compute necrosis features.

    Parameters
    ----------
    geojson_path
        segmentation_all_classes.geojson

    roi_csv_path
        CSV containing:
            cluster_id, minx, miny, maxx, maxy

    mpp
        Microns per pixel.

    out_csv
        Optional output CSV path.

    focal_threshold
        Fraction separating focal from present.
    """
    roi_csv_path = Path(roi_csv_path)

    if not roi_csv_path.exists():
        raise FileNotFoundError(f"ROI CSV not found: {roi_csv_path}")

    roi_df = pd.read_csv(roi_csv_path)

    # tumor_roi_overlay.py writes bounding-box columns as
    # x_min/y_min/x_max/y_max; the geometry code below expects
    # minx/miny/maxx/maxy. Accept either spelling and normalise.
    column_aliases = {
        "x_min": "minx",
        "y_min": "miny",
        "x_max": "maxx",
        "y_max": "maxy",
    }
    roi_df = roi_df.rename(
        columns={
            src: dst
            for src, dst in column_aliases.items()
            if src in roi_df.columns and dst not in roi_df.columns
        }
    )

    required = {
        "cluster_id",
        "minx",
        "miny",
        "maxx",
        "maxy",
    }

    missing = required - set(roi_df.columns)
    if missing:
        raise ValueError(
            f"ROI CSV missing required columns: {sorted(missing)}"
        )

    cluster_rois = roi_df.to_dict("records")

    df = extract_necrosis_features(
        geojson_path=geojson_path,
        cluster_roi_boxes=cluster_rois,
        mpp=mpp,
        focal_threshold=focal_threshold,
    )

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
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

    cfg.OUT_DIR/<slide>/spatial_feature_results/tumor_roi_overlay/
        tumor_roi_boxes.csv

    Writes
    ------
    cfg.OUT_DIR/<slide>/spatial_feature_results/necrosis_feature/
        necrosis_feature_by_cluster.csv

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

    roi_csv = (
        Path(cfg.OUT_DIR)
        / slide_name
        / "spatial_feature_results"
        / "tumor_roi_overlay"
        / "tumor_roi_boxes.csv"
    )

    out_dir = (
        Path(cfg.OUT_DIR)
        / slide_name
        / "spatial_feature_results"
        / "necrosis_feature"
    )

    out_csv = out_dir / "necrosis_feature_by_cluster.csv"

    if not segmentation_geojson.exists():
        raise FileNotFoundError(
            f"Segmentation GeoJSON not found: "
            f"{segmentation_geojson}\n"
            f"Run run_stitching(wsi_path, cfg) first."
        )

    if not roi_csv.exists():
        raise FileNotFoundError(
            f"Tumour cluster ROI CSV not found: "
            f"{roi_csv}\n"
            f"Run tumor_roi_overlay.py for this slide first."
        )

    mpp = getattr(cfg, "MPP", 0.25)

    focal_threshold = getattr(
        cfg,
        "NECROSIS_FOCAL_THRESHOLD",
        FOCAL_THRESHOLD,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 55}")
    print("  Necrosis features")
    print(f"  Slide      : {slide_name}")
    print(f"  Segmentation: {segmentation_geojson.name}")
    print(f"  Cluster ROI : {roi_csv.name}")
    print(f"  MPP         : {mpp}")
    print(f"  Focal cutoff: {focal_threshold:.3f}")
    print(f"  Output      : {out_dir}")
    print(f"{'=' * 55}")

    df = extract_necrosis_features_from_csv(
        geojson_path=segmentation_geojson,
        roi_csv_path=roi_csv,
        mpp=mpp,
        out_csv=out_csv,
        focal_threshold=focal_threshold,
    )

    # Simple summary.
    n_clusters = len(df)
    n_absent = int(
        (df["necrosis_phenotype"] == "absent").sum()
    )
    n_focal = int(
        (df["necrosis_phenotype"] == "focal").sum()
    )
    n_present = int(
        (df["necrosis_phenotype"] == "present").sum()
    )

    total_necrosis_um2 = float(
        df["necrosis_area_um2"].sum()
    )

    print("\n  === Necrosis summary ===")
    print(f"  Clusters:              {n_clusters}")
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
        "roi_csv": str(roi_csv),
        "cluster_csv": str(out_csv),
    }


# ═════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═════════════════════════════════════════════════════════════════════════
#
# Reuses config.py's full CLI exactly like immune_proximity_features.py.
#
# Examples:
#
#   python necrosis_features.py \
#       --wsi-path slides/TCGA-A1-A0SP.svs \
#       --out-dir vipsegd_output
#
#   python necrosis_proximity_features.py \
#       --from-json run_config.json
#
# If NECROSIS_FOCAL_THRESHOLD is added to PipelineConfig/config_from_args,
# it will automatically be available through the shared configuration CLI.

def main(argv=None) -> None:
    from .config import config_from_args

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
    seg_geojson = (
        Path(cfg.OUT_DIR) / slide_name
        / "segmentation" / "segmentation_all_classes.geojson"
    )
    if not roi_csv.exists():
        raise SystemExit(
            f"Tumour cluster ROI CSV not found: {roi_csv}. "
            f"Run tumor_roi_overlay.py for this slide (with the same --out-dir) first."
        )
    if not seg_geojson.exists():
        raise SystemExit(
            f"Segmentation GeoJSON not found: {seg_geojson}. "
            f"Run stitch.py for this slide (with the same --out-dir) first."
        )

    run_necrosis_features(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()