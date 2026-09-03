#!/usr/bin/env python3
"""
cluster_tils_tsr_scoring.py
===========================

Clean report-compatible edition: internal calculations stay complete, while saved
CSVs contain core scientific fields plus columns consumed by the current QMD report.
Master-ROI construction, inter-tumour corridor generation, TSR, stromal TIL,
intratumoral TIL, and inter-tumour stromal TIL scoring.

This version is designed to sit directly on top of the cleaned
``tumor_roi_overlay.py`` stage. Tumour foci are treated as biological objects;
representative square tumour ROIs are retained only as image samples. The
scorer does not recluster tumour tiles. Corridor-QC-passing focus edges create
inter-tumour relationships; focus pairs closer than ``INTER_TUMOR_MIN_GAP_UM``
are directly grouped for Master-ROI connectivity without creating a corridor.

Performance architecture
------------------------
The default ``fast`` mode uses the manifest tile fractions as a spatial raster:
Master-ROI scoring and corridor QC are masked NumPy sums, and routing uses a
local SciPy sparse-graph Dijkstra implementation. Vector geometry is retained
for tumour foci, Master-ROI/corridor exports, morphology, and interoperability.
Optional ``refine`` mode recomputes final accepted Master-ROI compartments with
exact Shapely polygon intersections for A/B validation or publication endpoints.

Spatial model
-------------
* tumour focus = node
* valid focus-to-focus relationship = graph edge
* inter-tumour corridor = tissue-aware path along an edge
* Master ROI = one graph-connected local tumour ecosystem
* tumour ROI = square image sample retained from tumor_roi_overlay.py

The public entry point remains ``run_cluster_tils_tsr_score`` for package
compatibility. Legacy filenames ``cluster_scoring_polygons.geojson`` and
``tils_tsr_by_cluster.csv`` are written alongside the newer Master-ROI names so
existing morphology, immune-proximity, pipeline skip/resume, and QMD code continue
to work unchanged.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle, Patch
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay, cKDTree
from shapely import intersects_xy
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from .config import cfg as default_cfg, PipelineConfig


CLASS_ORDER = ["Tumour", "Stroma", "Inflammatory", "Necrosis", "Others"]
CLASS_COLORS = {
    "Tumour": "#dc1e1e",
    "Stroma": "#1eb41e",
    "Inflammatory": "#1e64ff",
    "Necrosis": "#ffa500",
    "Others": "#c800c8",
}

# Descriptive report bands used only for overlay presentation.
# They mirror the current QMD display bands and are not clinical cut-offs.
TILS_LEVEL_COLORS = {
    "very low": "#2c7bb6",
    "low": "#abd9e9",
    "intermediate": "#fdae61",
    "high": "#d7191c",
    "unreliable": "#888888",
}


def _stil_display_level(value, reliable=True) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unreliable"
    if not bool(reliable) or not np.isfinite(v):
        return "unreliable"
    if v < 5.0:
        return "very low"
    if v < 10.0:
        return "low"
    if v < 30.0:
        return "intermediate"
    return "high"


def _fmt_overlay_pct(value) -> str:
    try:
        v = float(value)
        return f"{v:.1f}%" if np.isfinite(v) else "—"
    except (TypeError, ValueError):
        return "—"


def _hex_to_rgb01(value: str) -> np.ndarray:
    h = value.lstrip("#")
    return np.asarray([int(h[i:i+2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


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


def fill_polygon_holes(geom: BaseGeometry) -> BaseGeometry:
    parts = []
    for p in iter_polygon_parts(geom):
        parts.append(Polygon(p.exterior.coords))
    return make_valid(unary_union(parts)) if parts else geom


def infer_tile_step(df: pd.DataFrame) -> Tuple[int, int]:
    def _step(vals):
        u = np.sort(pd.Series(vals).dropna().unique())
        d = np.diff(u)
        d = d[d > 0]
        return int(round(np.median(d))) if len(d) else 256
    return _step(df["wx"]), _step(df["wy"])


def normalize_class_name(feat: dict) -> Optional[str]:
    props = feat.get("properties") or {}
    cls = (props.get("classification") or {}).get("name")
    if cls in CLASS_ORDER:
        return cls
    cls = props.get("class") or props.get("name") or props.get("label")
    if cls in CLASS_ORDER:
        return cls
    idx = props.get("class_index")
    if isinstance(idx, int) and 0 <= idx < len(CLASS_ORDER):
        return CLASS_ORDER[idx]
    return None


def load_regions(path: Path, min_area_px2: float = 1.0) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for feat in data.get("features", []):
        cls = normalize_class_name(feat)
        if cls is None:
            continue
        try:
            geom = make_valid(shape(feat.get("geometry")))
        except Exception:
            continue
        if geom is None or geom.is_empty or geom.area < min_area_px2:
            continue
        rows.append({"class": cls, "geometry": geom, "geometry_area_px2": float(geom.area)})
    return pd.DataFrame(rows)


def load_foci(path: Path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        geom = make_valid(shape(feat["geometry"]))
        if geom is None or geom.is_empty:
            continue
        fid = props.get("focus_id", props.get("cluster_id"))
        if fid is None:
            raise ValueError("tumor_foci.geojson requires focus_id or cluster_id")
        rp = geom.representative_point()
        rows.append({
            "focus_id": int(fid),
            "geometry": geom,
            "centroid_x": float(props.get("centroid_x", rp.x)),
            "centroid_y": float(props.get("centroid_y", rp.y)),
            "focus_area_px2": float(props.get("focus_area_px2", geom.area)),
            "focus_area_um2": props.get("focus_area_um2", np.nan),
        })
    return pd.DataFrame(rows).sort_values("focus_id").reset_index(drop=True) if rows else pd.DataFrame()


def load_roi_boxes(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rois = pd.read_csv(path)
    if "focus_id" not in rois.columns and "cluster_id" in rois.columns:
        rois["focus_id"] = rois["cluster_id"]
    return rois


def _query_indices(tree: STRtree, geoms: List[BaseGeometry], geom: BaseGeometry) -> List[int]:
    hits = tree.query(geom)
    if len(hits) == 0:
        return []
    if isinstance(hits[0], (int, np.integer)):
        return [int(i) for i in hits]
    id_map = {id(g): i for i, g in enumerate(geoms)}
    return [id_map[id(g)] for g in hits]


class RegionIndex:
    """One reusable spatial index for semantic segmentation polygons.

    Building the STRtree once avoids repeating the index construction for
    corridor QC and Master-ROI scoring.
    """

    def __init__(self, regions: pd.DataFrame):
        self.regions = regions
        self.geoms = regions["geometry"].tolist()
        self.classes = regions["class"].to_numpy()
        self.tree = STRtree(self.geoms)

    def areas(self, geom: BaseGeometry) -> Dict[str, float]:
        out = {c: 0.0 for c in CLASS_ORDER}
        if geom is None or geom.is_empty:
            return out
        for i in _query_indices(self.tree, self.geoms, geom):
            g = self.geoms[i]
            if not geom.intersects(g):
                continue
            inter = make_valid(geom.intersection(g))
            if inter is None or inter.is_empty:
                continue
            out[self.classes[i]] += float(inter.area)
        return out



class TileSpatialGrid:
    """Dense tile-lattice representation of the manifest fractions.

    The grid stores the original fractional semantic signal rather than a
    winner-take-all class mask. Missing lattice cells are represented by
    ``valid == False`` and zero class fractions.

    Notes
    -----
    * ``wx``/``wy`` are treated as tile-centre coordinates, matching the
      existing ViSpace manifest and ROI logic.
    * The tile area used for masked sums is ``step_x * step_y``. This is exact
      for a non-overlapping lattice and an approximation if upstream tiles
      overlap. ``refine`` mode is available when exact vector areas are needed.
    """

    REQUIRED_FRACTION_COLUMNS = {
        "Tumour": "frac_Tumour",
        "Stroma": "frac_Stroma",
        "Inflammatory": "frac_Inflammatory",
        "Necrosis": "frac_Necrosis",
        "Others": "frac_Others",
    }

    def __init__(self, manifest: pd.DataFrame, mpp: float, max_cells: int = 25_000_000):
        missing = [c for c in ["wx", "wy", *self.REQUIRED_FRACTION_COLUMNS.values()] if c not in manifest.columns]
        if missing:
            raise ValueError(f"Manifest missing required tile-raster columns: {missing}")

        self.mpp = float(mpp)
        self.step_x, self.step_y = infer_tile_step(manifest)
        self.x0 = float(manifest["wx"].min())
        self.y0 = float(manifest["wy"].min())

        gx = np.rint((manifest["wx"].to_numpy(float) - self.x0) / self.step_x).astype(np.int32)
        gy = np.rint((manifest["wy"].to_numpy(float) - self.y0) / self.step_y).astype(np.int32)
        self.nx = int(gx.max()) + 1 if len(gx) else 0
        self.ny = int(gy.max()) + 1 if len(gy) else 0
        n_cells = int(self.nx) * int(self.ny)
        if n_cells <= 0:
            raise RuntimeError("Manifest produced an empty tile lattice")
        if n_cells > int(max_cells):
            raise MemoryError(
                f"Tile lattice would contain {n_cells:,} cells ({self.nx}x{self.ny}), "
                f"above SPATIAL_GRID_MAX_CELLS={int(max_cells):,}. "
                "Raise the safety limit only if memory permits, or use a coarser manifest."
            )

        self.shape = (self.ny, self.nx)
        self.tile_area_px2 = float(self.step_x * self.step_y)
        self.valid = np.zeros(self.shape, dtype=bool)
        self.row_index = np.full(self.shape, -1, dtype=np.int32)
        self.valid[gy, gx] = True
        self.row_index[gy, gx] = np.arange(len(manifest), dtype=np.int32)

        self.frac: Dict[str, np.ndarray] = {}
        for cls, col in self.REQUIRED_FRACTION_COLUMNS.items():
            arr = np.zeros(self.shape, dtype=np.float32)
            vals = np.nan_to_num(manifest[col].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
            arr[gy, gx] = np.clip(vals, 0.0, 1.0).astype(np.float32)
            self.frac[cls] = arr

        total = np.zeros(self.shape, dtype=np.float32)
        for cls in CLASS_ORDER:
            total += self.frac[cls]
        self.coverage = np.clip(total, 0.0, 1.0)
        self.manifest = manifest.copy()
        self.manifest["gx"] = gx
        self.manifest["gy"] = gy

    @property
    def n_cells(self) -> int:
        return int(self.nx * self.ny)

    def tissue_cost(self, tumor_weight: float, necrosis_weight: float, other_weight: float) -> np.ndarray:
        return (
            1.0
            + float(tumor_weight) * self.frac["Tumour"]
            + float(necrosis_weight) * self.frac["Necrosis"]
            + float(other_weight) * self.frac["Others"]
        ).astype(np.float32, copy=False)

    def focus_owner_array(self, focus_tiles: pd.DataFrame) -> np.ndarray:
        owner = np.zeros(self.shape, dtype=np.int32)
        if focus_tiles.empty:
            return owner
        work = focus_tiles.copy()
        if "focus_id" not in work.columns and "cluster_id" in work.columns:
            work["focus_id"] = work["cluster_id"]
        if "focus_id" not in work.columns:
            raise ValueError("tumor_focus_tiles.csv requires focus_id or cluster_id")
        gx = np.rint((work["wx"].to_numpy(float) - self.x0) / self.step_x).astype(int)
        gy = np.rint((work["wy"].to_numpy(float) - self.y0) / self.step_y).astype(int)
        fids = work["focus_id"].to_numpy(int)
        ok = (gx >= 0) & (gx < self.nx) & (gy >= 0) & (gy < self.ny)
        owner[gy[ok], gx[ok]] = fids[ok]
        return owner

    def coords_from_xy(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(round((float(wx) - self.x0) / self.step_x))
        gy = int(round((float(wy) - self.y0) / self.step_y))
        return gx, gy

    def xy_from_coords(self, gx: int, gy: int) -> Tuple[float, float]:
        return self.x0 + int(gx) * self.step_x, self.y0 + int(gy) * self.step_y

    def _index_bounds_for_geometry(self, geom: BaseGeometry, pad_cells: int = 1) -> Tuple[int, int, int, int]:
        minx, miny, maxx, maxy = geom.bounds
        gx0 = max(0, int(math.floor((minx - self.x0) / self.step_x)) - pad_cells)
        gx1 = min(self.nx - 1, int(math.ceil((maxx - self.x0) / self.step_x)) + pad_cells)
        gy0 = max(0, int(math.floor((miny - self.y0) / self.step_y)) - pad_cells)
        gy1 = min(self.ny - 1, int(math.ceil((maxy - self.y0) / self.step_y)) + pad_cells)
        return gx0, gx1, gy0, gy1

    def geometry_mask(self, geom: BaseGeometry) -> np.ndarray:
        """Rasterize geometry by tile-centre inclusion.

        A lattice cell is selected when its tile centre intersects the geometry.
        This is intentionally a fast, centre-sampled rasterization; partial
        boundary-cell coverage is not represented in ``fast`` mode. Exact vector
        boundary areas can be recovered with ``SPATIAL_SCORING_MODE='refine'``.
        """
        out = np.zeros(self.shape, dtype=bool)
        if geom is None or geom.is_empty:
            return out
        geom = make_valid(geom)
        if geom is None or geom.is_empty:
            return out
        gx0, gx1, gy0, gy1 = self._index_bounds_for_geometry(geom)
        if gx1 < gx0 or gy1 < gy0:
            return out
        xs = self.x0 + np.arange(gx0, gx1 + 1, dtype=float) * self.step_x
        ys = self.y0 + np.arange(gy0, gy1 + 1, dtype=float) * self.step_y
        xx, yy = np.meshgrid(xs, ys)
        local = intersects_xy(geom, xx, yy)
        out[gy0:gy1 + 1, gx0:gx1 + 1] = np.asarray(local, dtype=bool)
        return out

    def path_corridor_mask(self, path: Sequence[Tuple[int, int]], corridor_width_px: float) -> np.ndarray:
        """Rasterize a routed path to a corridor mask.

        The effective corridor width is never allowed to be narrower than one
        tile pitch. This prevents a valid vector/path corridor from disappearing
        solely because it falls between tile centres.
        """
        out = np.zeros(self.shape, dtype=bool)
        if not path:
            return out
        min_pitch_px = min(self.step_x, self.step_y)
        radius_px = max(float(corridor_width_px) / 2.0, min_pitch_px / 2.0)
        xs = np.asarray([p[0] for p in path], dtype=int)
        ys = np.asarray([p[1] for p in path], dtype=int)
        pad_x = int(math.ceil(radius_px / self.step_x)) + 2
        pad_y = int(math.ceil(radius_px / self.step_y)) + 2
        gx0 = max(0, int(xs.min()) - pad_x)
        gx1 = min(self.nx - 1, int(xs.max()) + pad_x)
        gy0 = max(0, int(ys.min()) - pad_y)
        gy1 = min(self.ny - 1, int(ys.max()) + pad_y)
        seed = np.zeros((gy1 - gy0 + 1, gx1 - gx0 + 1), dtype=bool)
        seed[ys - gy0, xs - gx0] = True
        dist = distance_transform_edt(~seed, sampling=(float(self.step_y), float(self.step_x)))
        local = dist <= radius_px
        out[gy0:gy1 + 1, gx0:gx1 + 1] = local
        return out

    def class_areas(self, mask: np.ndarray) -> Dict[str, float]:
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} != tile grid shape {self.shape}")
        return {
            cls: float(self.frac[cls][mask].sum(dtype=np.float64) * self.tile_area_px2)
            for cls in CLASS_ORDER
        }

    def region_area(self, mask: np.ndarray) -> float:
        if mask.shape != self.shape:
            raise ValueError(f"Mask shape {mask.shape} != tile grid shape {self.shape}")
        return float(np.count_nonzero(mask) * self.tile_area_px2)

    def window_composition(self, box: Tuple[float, float, float, float]) -> Dict[str, float]:
        x1, y1, x2, y2 = map(float, box)
        gx0 = max(0, int(math.floor((x1 - self.x0) / self.step_x)) - 1)
        gx1 = min(self.nx - 1, int(math.ceil((x2 - self.x0) / self.step_x)) + 1)
        gy0 = max(0, int(math.floor((y1 - self.y0) / self.step_y)) - 1)
        gy1 = min(self.ny - 1, int(math.ceil((y2 - self.y0) / self.step_y)) + 1)
        if gx1 < gx0 or gy1 < gy0:
            return {"tumor": np.nan, "stroma": np.nan, "inflammatory": np.nan,
                    "necrosis": np.nan, "others": np.nan, "n_tiles": 0}
        xs = self.x0 + np.arange(gx0, gx1 + 1, dtype=float) * self.step_x
        ys = self.y0 + np.arange(gy0, gy1 + 1, dtype=float) * self.step_y
        xsel = (xs >= x1) & (xs < x2)
        ysel = (ys >= y1) & (ys < y2)
        local_select = np.outer(ysel, xsel)
        valid = self.valid[gy0:gy1 + 1, gx0:gx1 + 1] & local_select
        n = int(valid.sum())
        if n == 0:
            return {"tumor": np.nan, "stroma": np.nan, "inflammatory": np.nan,
                    "necrosis": np.nan, "others": np.nan, "n_tiles": 0}
        keymap = {
            "tumor": "Tumour",
            "stroma": "Stroma",
            "inflammatory": "Inflammatory",
            "necrosis": "Necrosis",
            "others": "Others",
        }
        out = {k: float(self.frac[cls][gy0:gy1 + 1, gx0:gx1 + 1][valid].mean()) for k, cls in keymap.items()}
        out["n_tiles"] = n
        return out

def candidate_focus_pairs(foci: pd.DataFrame, knn_k: int = 3) -> List[Tuple[int, int]]:
    """Delaunay + kNN candidate graph; exact boundary distance is checked later."""
    n = len(foci)
    if n < 2:
        return []
    pts = foci[["centroid_x", "centroid_y"]].to_numpy(float)
    ids = foci["focus_id"].to_numpy(int)
    pairs = set()

    if n >= 3:
        try:
            tri = Delaunay(pts)
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        a, b = int(ids[simplex[i]]), int(ids[simplex[j]])
                        pairs.add(tuple(sorted((a, b))))
        except Exception:
            pass

    k = min(max(2, knn_k + 1), n)
    tree = cKDTree(pts)
    _, nn = tree.query(pts, k=k)
    if nn.ndim == 1:
        nn = nn[:, None]
    for i in range(n):
        for j in np.atleast_1d(nn[i])[1:]:
            a, b = int(ids[i]), int(ids[int(j)])
            if a != b:
                pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


def build_close_focus_merges(
    foci: pd.DataFrame,
    mpp: float,
    min_gap_um: float,
    knn_k: int = 3,
) -> pd.DataFrame:
    """Return very-close focus pairs that should share a Master ROI.

    ``INTER_TUMOR_MIN_GAP_UM`` separates two concepts:

    * gap < min_gap: too close to interpret as an inter-tumour stromal corridor;
      directly merge the foci for Master-ROI connectivity.
    * min_gap <= gap <= max_gap: eligible for routed inter-tumour corridor QC.

    Close merges therefore affect Master-ROI connectivity but do not create an
    inter-tumour ROI or an inter-tumour TIL denominator.
    """
    if foci.empty or len(foci) < 2 or min_gap_um <= 0:
        return pd.DataFrame(columns=[
            "focus_id_1", "focus_id_2", "boundary_gap_um", "relationship_type"
        ])
    flookup = foci.set_index("focus_id")
    rows = []
    for a, b in candidate_focus_pairs(foci, knn_k):
        gap_um = float(flookup.loc[a, "geometry"].distance(flookup.loc[b, "geometry"]) * mpp)
        if gap_um < min_gap_um:
            rows.append({
                "focus_id_1": int(a),
                "focus_id_2": int(b),
                "boundary_gap_um": gap_um,
                "relationship_type": "close_focus_merge",
            })
    return pd.DataFrame(rows)


def _nearest_member_pair(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    if a.empty or b.empty:
        raise ValueError("Focus tile membership missing for one or both foci")
    bpts = b[["wx", "wy"]].to_numpy(float)
    tree = cKDTree(bpts)
    dist, idx = tree.query(a[["wx", "wy"]].to_numpy(float), k=1)
    ia = int(np.argmin(dist))
    ib = int(idx[ia])
    return a.iloc[ia], b.iloc[ib]


def prepare_manifest_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], int], int, int, float, float]:
    step_x, step_y = infer_tile_step(df)
    x0, y0 = float(df["wx"].min()), float(df["wy"].min())
    out = df.copy()
    out["gx"] = np.rint((out["wx"] - x0) / step_x).astype(int)
    out["gy"] = np.rint((out["wy"] - y0) / step_y).astype(int)
    coord_to_row = {(int(r.gx), int(r.gy)): i for i, r in enumerate(out.itertuples())}
    return out, coord_to_row, step_x, step_y, x0, y0



def dijkstra_tissue_path(
    grid: TileSpatialGrid,
    tissue_costs: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    search_margin_cells: int,
    focus_owner: np.ndarray,
    focus_a: int,
    focus_b: int,
) -> List[Tuple[int, int]]:
    """Shortest tissue-aware path on a local sparse tile graph.

    Only existing manifest tiles are nodes. Tiles belonging to unrelated tumour
    foci are removed from the graph. Edge weights are physical step length
    multiplied by the destination tile's semantic tissue cost. The local graph
    is solved by SciPy's compiled ``csgraph.dijkstra`` implementation.
    """
    if start == goal:
        return [start]

    sx, sy = map(int, start)
    gx, gy = map(int, goal)
    pad = int(max(1, search_margin_cells))
    x0 = max(0, min(sx, gx) - pad)
    x1 = min(grid.nx - 1, max(sx, gx) + pad)
    y0 = max(0, min(sy, gy) - pad)
    y1 = min(grid.ny - 1, max(sy, gy) + pad)

    allowed = grid.valid[y0:y1 + 1, x0:x1 + 1].copy()
    owners = focus_owner[y0:y1 + 1, x0:x1 + 1]
    blocked = (owners != 0) & (owners != int(focus_a)) & (owners != int(focus_b))
    allowed &= ~blocked

    lsx, lsy = sx - x0, sy - y0
    lgx, lgy = gx - x0, gy - y0
    if not (0 <= lsx < allowed.shape[1] and 0 <= lsy < allowed.shape[0]):
        return []
    if not (0 <= lgx < allowed.shape[1] and 0 <= lgy < allowed.shape[0]):
        return []
    if not allowed[lsy, lsx] or not allowed[lgy, lgx]:
        return []

    node_ids = np.full(allowed.shape, -1, dtype=np.int32)
    local_y, local_x = np.nonzero(allowed)
    n_nodes = len(local_y)
    if n_nodes == 0:
        return []
    node_ids[local_y, local_x] = np.arange(n_nodes, dtype=np.int32)

    local_cost = tissue_costs[y0:y1 + 1, x0:x1 + 1]
    src_chunks = []
    dst_chunks = []
    weight_chunks = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]

    h, w = allowed.shape
    for dy, dx in directions:
        sy0 = max(0, -dy)
        sy1 = h - max(0, dy)
        sx0 = max(0, -dx)
        sx1 = w - max(0, dx)
        dy0 = max(0, dy)
        dy1 = h - max(0, -dy)
        dx0 = max(0, dx)
        dx1 = w - max(0, -dx)
        if sy1 <= sy0 or sx1 <= sx0:
            continue
        src_ok = allowed[sy0:sy1, sx0:sx1]
        dst_ok = allowed[dy0:dy1, dx0:dx1]
        ok = src_ok & dst_ok
        if not np.any(ok):
            continue
        src = node_ids[sy0:sy1, sx0:sx1][ok]
        dst = node_ids[dy0:dy1, dx0:dx1][ok]
        step_px = math.hypot(dx * grid.step_x, dy * grid.step_y)
        weights = step_px * local_cost[dy0:dy1, dx0:dx1][ok].astype(np.float64)
        src_chunks.append(src)
        dst_chunks.append(dst)
        weight_chunks.append(weights)

    if not src_chunks:
        return []

    src = np.concatenate(src_chunks)
    dst = np.concatenate(dst_chunks)
    weights = np.concatenate(weight_chunks)
    graph = csr_matrix((weights, (src, dst)), shape=(n_nodes, n_nodes))

    start_node = int(node_ids[lsy, lsx])
    goal_node = int(node_ids[lgy, lgx])
    dist, predecessors = dijkstra(
        graph,
        directed=True,
        indices=start_node,
        return_predecessors=True,
    )
    if not np.isfinite(dist[goal_node]):
        return []

    rev = [goal_node]
    cur = goal_node
    sentinel = -9999
    while cur != start_node:
        cur = int(predecessors[cur])
        if cur == sentinel or cur < 0:
            return []
        rev.append(cur)
        if len(rev) > n_nodes + 1:
            return []
    rev.reverse()

    path = []
    for nid in rev:
        yy = int(local_y[nid]) + y0
        xx = int(local_x[nid]) + x0
        path.append((xx, yy))
    return path

def path_to_geometry(
    path: Sequence[Tuple[int, int]],
    step_x: int,
    step_y: int,
    x0: float,
    y0: float,
    corridor_width_px: float,
) -> BaseGeometry:
    if not path:
        return Polygon()
    points = [(x0 + gx * step_x, y0 + gy * step_y) for gx, gy in path]
    if len(points) == 1:
        return Point(points[0]).buffer(max(step_x, step_y) / 2)
    line = LineString(points)
    return make_valid(line.buffer(max(corridor_width_px / 2.0, min(step_x, step_y) / 2.0), cap_style=2, join_style=1))


def _path_quantile_point(path: Sequence[Tuple[int, int]], q: float, step_x: int, step_y: int, x0: float, y0: float) -> Tuple[float, float]:
    idx = int(round(np.clip(q, 0.0, 1.0) * max(0, len(path) - 1)))
    gx, gy = path[idx]
    return x0 + gx * step_x, y0 + gy * step_y


def _window_composition(df: pd.DataFrame, box: Tuple[float, float, float, float]) -> Dict[str, float]:
    x1, y1, x2, y2 = box
    inside = df[(df["wx"] >= x1) & (df["wx"] < x2) & (df["wy"] >= y1) & (df["wy"] < y2)]
    if inside.empty:
        return {"tumor": np.nan, "stroma": np.nan, "inflammatory": np.nan,
                "necrosis": np.nan, "others": np.nan, "n_tiles": 0}
    return {
        "tumor": float(inside["frac_Tumour"].mean()),
        "stroma": float(inside["frac_Stroma"].mean()),
        "inflammatory": float(inside["frac_Inflammatory"].mean()),
        "necrosis": float(inside["frac_Necrosis"].mean()),
        "others": float(inside["frac_Others"].mean()),
        "n_tiles": int(len(inside)),
    }



def build_focus_edges_and_corridors(
    foci: pd.DataFrame,
    focus_tiles: pd.DataFrame,
    grid: TileSpatialGrid,
    mpp: float,
    min_gap_um: float,
    max_gap_um: float,
    knn_k: int,
    corridor_width_um: float,
    search_margin_um: float,
    tumor_weight: float,
    necrosis_weight: float,
    other_weight: float,
) -> Tuple[pd.DataFrame, Dict[int, BaseGeometry], Dict[int, List[Tuple[int, int]]]]:
    if foci.empty or len(foci) < 2:
        return pd.DataFrame(), {}, {}

    flookup = foci.set_index("focus_id")
    work_tiles = focus_tiles.copy()
    if "focus_id" not in work_tiles.columns and "cluster_id" in work_tiles.columns:
        work_tiles["focus_id"] = work_tiles["cluster_id"]
    if "focus_id" not in work_tiles.columns:
        raise ValueError("tumor_focus_tiles.csv requires focus_id or cluster_id")
    work_tiles["gx"] = np.rint((work_tiles["wx"] - grid.x0) / grid.step_x).astype(int)
    work_tiles["gy"] = np.rint((work_tiles["wy"] - grid.y0) / grid.step_y).astype(int)
    focus_owner = grid.focus_owner_array(work_tiles)

    tissue_costs = grid.tissue_cost(
        tumor_weight=tumor_weight,
        necrosis_weight=necrosis_weight,
        other_weight=other_weight,
    )
    margin_cells = max(2, int(math.ceil((search_margin_um / mpp) / min(grid.step_x, grid.step_y))))
    corridor_width_px = corridor_width_um / mpp

    rows: List[Dict] = []
    corridors: Dict[int, BaseGeometry] = {}
    paths: Dict[int, List[Tuple[int, int]]] = {}
    edge_id = 1

    for a, b in tqdm(candidate_focus_pairs(foci, knn_k), desc="Building focus graph (Dijkstra)", unit="pair"):
        ga = flookup.loc[a, "geometry"]
        gb = flookup.loc[b, "geometry"]
        gap_px = float(ga.distance(gb))
        gap_um = gap_px * mpp
        # Very-close pairs (gap < min_gap_um) are handled separately as
        # direct Master-ROI connectivity, not as inter-tumour corridors.
        if gap_um < min_gap_um or gap_um > max_gap_um:
            continue

        ta = work_tiles[work_tiles["focus_id"] == a]
        tb = work_tiles[work_tiles["focus_id"] == b]
        if ta.empty or tb.empty:
            continue
        ra, rb = _nearest_member_pair(ta, tb)
        start = (int(ra.gx), int(ra.gy))
        goal = (int(rb.gx), int(rb.gy))

        path = dijkstra_tissue_path(
            grid=grid,
            tissue_costs=tissue_costs,
            start=start,
            goal=goal,
            search_margin_cells=margin_cells,
            focus_owner=focus_owner,
            focus_a=a,
            focus_b=b,
        )
        if not path:
            continue

        geom = path_to_geometry(
            path,
            grid.step_x,
            grid.step_y,
            grid.x0,
            grid.y0,
            corridor_width_px,
        )
        path_len_px = sum(
            math.hypot(
                (path[i][0] - path[i - 1][0]) * grid.step_x,
                (path[i][1] - path[i - 1][1]) * grid.step_y,
            )
            for i in range(1, len(path))
        )
        rows.append({
            "edge_id": edge_id,
            "focus_id_1": int(a),
            "focus_id_2": int(b),
            "boundary_gap_um": gap_um,
            "path_length_um": float(path_len_px * mpp),
            "n_path_tiles": int(len(path)),
            "routing_method": "scipy_csgraph_dijkstra",
        })
        corridors[edge_id] = geom
        paths[edge_id] = path
        edge_id += 1

    return pd.DataFrame(rows), corridors, paths

def score_corridor_qc(
    edges: pd.DataFrame,
    paths: Dict[int, List[Tuple[int, int]]],
    grid: TileSpatialGrid,
    focus_owner: np.ndarray,
    corridor_width_px: float,
    max_tumor_frac: float = 0.20,
    max_necrosis_frac: float = 0.30,
    min_stromal_like_frac: float = 0.30,
    min_tissue_fraction: float = 0.50,
    min_path_efficiency: float = 0.50,
) -> pd.DataFrame:
    """Fast corridor QC entirely on the tile-raster substrate.

    The corridor mask is generated from the Dijkstra path. Endpoint focus-owned
    tiles are removed using ``focus_owner`` before composition is measured.
    Class areas, total corridor area, tissue fraction and background therefore
    all use the same raster-cell denominator; no Shapely difference/intersection
    is performed in corridor QC.
    """
    if edges.empty:
        return edges.copy()
    if focus_owner.shape != grid.shape:
        raise ValueError("focus_owner shape must match TileSpatialGrid shape")

    rows = []
    for e in tqdm(edges.itertuples(), total=len(edges), desc="Corridor QC (raster)", unit="edge"):
        eid = int(e.edge_id)
        fa, fb = int(e.focus_id_1), int(e.focus_id_2)
        corridor_mask = grid.path_corridor_mask(paths.get(eid, []), corridor_width_px)
        gap_mask = corridor_mask & (focus_owner != fa) & (focus_owner != fb)

        raster_cell_area = grid.region_area(gap_mask)
        areas = grid.class_areas(gap_mask) if raster_cell_area > 0 else {c: 0.0 for c in CLASS_ORDER}
        segmented = float(sum(areas.values()))
        tissue_fraction = segmented / raster_cell_area if raster_cell_area > 0 else 0.0
        background = raster_cell_area - segmented

        def frac(cls: str) -> float:
            return float(areas[cls] / segmented) if segmented > 0 else np.nan

        tumor_frac = frac("Tumour")
        stroma_frac = frac("Stroma")
        inflam_frac = frac("Inflammatory")
        necrosis_frac = frac("Necrosis")
        others_frac = frac("Others")
        stromal_like_frac = (
            float((areas["Stroma"] + areas["Inflammatory"]) / segmented)
            if segmented > 0 else np.nan
        )
        path_efficiency = (
            float(e.boundary_gap_um / e.path_length_um)
            if float(e.path_length_um) > 0 else np.nan
        )

        reasons = []
        if raster_cell_area <= 0:
            reasons.append("empty_gap_corridor")
        if tissue_fraction < min_tissue_fraction:
            reasons.append("low_tissue_fraction")
        if np.isfinite(tumor_frac) and tumor_frac > max_tumor_frac:
            reasons.append("high_tumor_contamination")
        if np.isfinite(necrosis_frac) and necrosis_frac > max_necrosis_frac:
            reasons.append("high_necrosis")
        if (not np.isfinite(stromal_like_frac)) or stromal_like_frac < min_stromal_like_frac:
            reasons.append("low_stromal_like_fraction")
        if (not np.isfinite(path_efficiency)) or path_efficiency < min_path_efficiency:
            reasons.append("low_path_efficiency")

        rec = e._asdict()
        rec.update({
            # Backward-compatible name now uses the fast-mode raster area basis.
            "corridor_gap_area_px2": raster_cell_area,
            "corridor_raster_cell_area_px2": raster_cell_area,
            "corridor_area_basis": "raster_cell_area",
            "corridor_segmented_area_px2": segmented,
            "corridor_background_area_px2": background,
            "corridor_tissue_fraction": tissue_fraction,
            "corridor_tumor_frac": tumor_frac,
            "corridor_stroma_frac": stroma_frac,
            "corridor_inflammatory_frac": inflam_frac,
            "corridor_necrosis_frac": necrosis_frac,
            "corridor_others_frac": others_frac,
            "corridor_stromal_like_frac": stromal_like_frac,
            "path_efficiency": path_efficiency,
            "corridor_qc_valid": len(reasons) == 0,
            "corridor_qc_reason": ";".join(reasons),
            "corridor_scoring_method": "tile_raster_masked_sum",
        })
        rows.append(rec)

    return pd.DataFrame(rows)

def _filter_edge_geometry_maps(
    edges: pd.DataFrame,
    corridors: Dict[int, BaseGeometry],
    paths: Dict[int, List[Tuple[int, int]]],
) -> Tuple[Dict[int, BaseGeometry], Dict[int, List[Tuple[int, int]]]]:
    valid_ids = set(edges["edge_id"].astype(int).tolist()) if not edges.empty else set()
    return (
        {eid: g for eid, g in corridors.items() if int(eid) in valid_ids},
        {eid: p for eid, p in paths.items() if int(eid) in valid_ids},
    )


class _UnionFind:
    def __init__(self, ids: Iterable[int]):
        self.parent = {int(i): int(i) for i in ids}
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def assign_master_rois(foci: pd.DataFrame, edges: pd.DataFrame) -> Dict[int, int]:
    uf = _UnionFind(foci["focus_id"].astype(int).tolist())
    if not edges.empty:
        for r in edges.itertuples():
            uf.union(int(r.focus_id_1), int(r.focus_id_2))
    groups = defaultdict(list)
    for fid in foci["focus_id"].astype(int):
        groups[uf.find(int(fid))].append(int(fid))
    ordered = sorted(groups.values(), key=lambda ids: min(ids))
    mapping_out = {}
    for mid, ids in enumerate(ordered, start=1):
        for fid in ids:
            mapping_out[fid] = mid
    return mapping_out


def build_master_geometries(
    foci: pd.DataFrame,
    edges: pd.DataFrame,
    corridors: Dict[int, BaseGeometry],
    master_map: Dict[int, int],
    stroma_margin_px: float,
    tissue_union: Optional[BaseGeometry],
) -> pd.DataFrame:
    focus_geoms = foci.set_index("focus_id")["geometry"].to_dict()
    mids = sorted(set(master_map.values()))
    rows = []
    for mid in mids:
        fids = sorted([fid for fid, m in master_map.items() if m == mid])
        parts = [focus_geoms[fid] for fid in fids]
        eids = []
        if not edges.empty:
            for e in edges.itertuples():
                if master_map.get(int(e.focus_id_1)) == mid and master_map.get(int(e.focus_id_2)) == mid:
                    parts.append(corridors[int(e.edge_id)])
                    eids.append(int(e.edge_id))
        core = make_valid(unary_union(parts))
        geom = make_valid(core.buffer(stroma_margin_px, join_style=1)) if stroma_margin_px > 0 else core
        if tissue_union is not None and not tissue_union.is_empty:
            geom = make_valid(geom.intersection(tissue_union))
        if geom is None or geom.is_empty:
            continue
        rows.append({
            "master_roi_id": int(mid),
            "cluster_id": int(mid),
            "focus_ids": ";".join(map(str, fids)),
            "edge_ids": ";".join(map(str, eids)),
            "n_foci": len(fids),
            "n_edges": len(eids),
            "geometry": geom,
        })
    return pd.DataFrame(rows)


def make_nonoverlapping_master_rois(masters: pd.DataFrame) -> pd.DataFrame:
    if masters.empty:
        return masters
    work = masters.copy()
    work["area_sort"] = work["geometry"].map(lambda g: g.area)
    work = work.sort_values(["area_sort", "master_roi_id"], ascending=[False, True])
    assigned: List[BaseGeometry] = []
    rows = []
    for _, r in work.iterrows():
        geom = r["geometry"]
        if assigned:
            geom = make_valid(geom.difference(unary_union(assigned)))
        if geom is None or geom.is_empty or geom.area <= 0:
            continue
        assigned.append(geom)
        row = r.drop(labels=["area_sort"]).to_dict()
        row["geometry"] = geom
        rows.append(row)
    return pd.DataFrame(rows).sort_values("master_roi_id").reset_index(drop=True)



def build_inter_tumor_rois(
    edges: pd.DataFrame,
    paths: Dict[int, List[Tuple[int, int]]],
    grid: TileSpatialGrid,
    mpp: float,
    roi_size_um: float,
    sample_fractions: Sequence[float],
    max_tumor_frac: float,
    max_necrosis: float,
    master_map: Dict[int, int],
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame()
    half = (roi_size_um / mpp) / 2.0
    rows = []
    rid = 1
    for e in edges.itertuples():
        path = paths.get(int(e.edge_id), [])
        if not path:
            continue
        for q in sample_fractions:
            cx, cy = _path_quantile_point(path, float(q), grid.step_x, grid.step_y, grid.x0, grid.y0)
            box = (cx - half, cy - half, cx + half, cy + half)
            comp = grid.window_composition(box)
            if comp["n_tiles"] == 0:
                continue
            if np.isfinite(comp["tumor"]) and comp["tumor"] > max_tumor_frac:
                continue
            if np.isfinite(comp["necrosis"]) and comp["necrosis"] > max_necrosis:
                continue
            rows.append({
                "inter_roi_id": rid,
                "edge_id": int(e.edge_id),
                "master_roi_id": int(master_map[int(e.focus_id_1)]),
                "focus_id_1": int(e.focus_id_1),
                "focus_id_2": int(e.focus_id_2),
                "path_fraction": float(q),
                "center_x": cx,
                "center_y": cy,
                "x_min": box[0],
                "y_min": box[1],
                "x_max": box[2],
                "y_max": box[3],
                **comp,
            })
            rid += 1
    return pd.DataFrame(rows)

def _areas_in_geometry(
    geom: BaseGeometry,
    regions: pd.DataFrame,
    geoms: List[BaseGeometry],
    tree: STRtree,
) -> Dict[str, float]:
    """Backward-compatible helper retained for external callers."""
    out = {c: 0.0 for c in CLASS_ORDER}
    classes = regions["class"].to_numpy()
    for i in _query_indices(tree, geoms, geom):
        g = geoms[i]
        if not geom.intersects(g):
            continue
        inter = make_valid(geom.intersection(g))
        if inter is None or inter.is_empty:
            continue
        out[classes[i]] += float(inter.area)
    return out



def score_master_rois_fast(
    masters: pd.DataFrame,
    foci: pd.DataFrame,
    edges: pd.DataFrame,
    paths: Dict[int, List[Tuple[int, int]]],
    grid: TileSpatialGrid,
    corridor_width_px: float,
    mpp: float,
    min_tissue_fraction: float,
    min_tsr_denom_px2: float,
    min_tils_denom_px2: float,
) -> pd.DataFrame:
    """Score Master-ROI compartments with fractional tile-raster masked sums."""
    if masters.empty:
        return pd.DataFrame()

    focus_geom = foci.set_index("focus_id")["geometry"].to_dict()
    rows = []

    for m in tqdm(masters.itertuples(), total=len(masters), desc="Scoring Master ROIs (raster)", unit="master"):
        master_geom = m.geometry
        master_mask = grid.geometry_mask(master_geom)
        fids = [int(x) for x in str(m.focus_ids).split(";") if x]
        eids = [int(x) for x in str(m.edge_ids).split(";") if x]

        tumor_bed_geom = fill_polygon_holes(unary_union([focus_geom[fid] for fid in fids]))
        tumor_bed_mask = grid.geometry_mask(tumor_bed_geom) & master_mask
        stromal_mask = master_mask & ~tumor_bed_mask

        # Each accepted edge belongs to exactly one connected Master ROI, so
        # retaining full-lattice corridor masks in a global cache only inflates
        # memory. Build and release them within the current Master ROI instead.
        inter_mask = np.zeros(grid.shape, dtype=bool)
        for eid in eids:
            inter_mask |= grid.path_corridor_mask(paths.get(eid, []), corridor_width_px)
        inter_mask &= master_mask
        inter_mask &= ~tumor_bed_mask

        whole = grid.class_areas(master_mask)
        intra = grid.class_areas(tumor_bed_mask)
        stromal = grid.class_areas(stromal_mask)
        inter = grid.class_areas(inter_mask)

        tumor = whole["Tumour"]
        stroma = whole["Stroma"]
        inflam = whole["Inflammatory"]
        necrosis = whole["Necrosis"]
        others = whole["Others"]
        segmented = tumor + stroma + inflam + necrosis + others
        raster_cell_area = grid.region_area(master_mask)
        vector_area = float(master_geom.area)

        # Fast-mode class areas are raster masked sums, therefore reliability
        # and background must use the same raster-cell area basis. Mixing the
        # exact vector polygon area here would bias tissue_fraction, especially
        # around buffered/partial boundary tiles.
        area = raster_cell_area
        tissue_fraction = segmented / area if area > 0 else 0.0
        background = area - segmented

        raw_denom = tumor + stroma
        tsr_raw = stroma / raw_denom if raw_denom > 0 else np.nan
        tumor_pct = tumor / raw_denom * 100 if raw_denom > 0 else np.nan
        stroma_pct = stroma / raw_denom * 100 if raw_denom > 0 else np.nan

        intra_i = intra["Inflammatory"]
        intra_t = intra["Tumour"]
        stromal_i = stromal["Inflammatory"]
        stromal_s = stromal["Stroma"]
        inter_i = inter["Inflammatory"]
        inter_s = inter["Stroma"]

        itil_denom = intra_t + intra_i
        stil_denom = stromal_s + stromal_i
        intertil_denom = inter_s + inter_i
        itil = intra_i / itil_denom * 100 if itil_denom > 0 else np.nan
        stil_occ = stromal_i / stil_denom * 100 if stil_denom > 0 else np.nan
        intertil = inter_i / intertil_denom * 100 if intertil_denom > 0 else np.nan
        inflammatory_to_stroma = inflam / stroma * 100 if stroma > 0 else np.nan

        effective_t = tumor + intra_i
        effective_s = stroma + stromal_i
        eff_denom = effective_t + effective_s
        tsr_comp = effective_s / eff_denom if eff_denom > 0 else np.nan

        tsr_reliable = bool(raw_denom >= min_tsr_denom_px2 and tissue_fraction >= min_tissue_fraction)
        stil_reliable = bool(stil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)
        itil_reliable = bool(itil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)
        intertil_reliable = bool(intertil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)

        gap_sub = edges[edges["edge_id"].isin(eids)] if (not edges.empty and eids) else pd.DataFrame()

        rows.append({
            "master_roi_id": int(m.master_roi_id),
            "cluster_id": int(m.master_roi_id),
            "focus_ids": m.focus_ids,
            "n_foci": int(m.n_foci),
            "n_intertumor_edges": int(m.n_edges),
            "scoring_method": "tile_raster_masked_sum",
            "master_area_px2": area,
            "master_area_basis": "raster_cell_area",
            "master_vector_area_px2": vector_area,
            "master_raster_cell_area_px2": raster_cell_area,
            "master_area_um2": area * mpp * mpp,
            "master_area_mm2": area * mpp * mpp / 1e6,
            "tumor_area_px2": tumor,
            "stroma_area_px2": stroma,
            "inflammatory_area_px2": inflam,
            "necrosis_area_px2": necrosis,
            "others_area_px2": others,
            "background_area_px2": background,
            "tissue_fraction": tissue_fraction,
            "TSR_tumor_pct": tumor_pct,
            "TSR_stroma_pct": stroma_pct,
            "TSR_stroma_fraction": tsr_raw,
            "TSR_compartment_stroma_fraction": tsr_comp,
            "TSR_reliable": tsr_reliable,
            "sTILs_pct_salgado": inflammatory_to_stroma,
            "inflammatory_to_stroma_ratio_pct": inflammatory_to_stroma,
            "sTILs_pct_stromal_occupancy": stil_occ,
            "sTILs_pct": stil_occ,
            "sTILs_reliable": stil_reliable,
            "iTILs_pct_computational": itil,
            "iTILs_reliable": itil_reliable,
            "inter_tumor_sTILs_pct": intertil,
            "inter_tumor_sTILs_reliable": intertil_reliable,
            "intratumoral_inflammatory_area_px2": intra_i,
            "intratumoral_tumor_area_px2": intra_t,
            "stromal_inflammatory_area_px2": stromal_i,
            "stromal_stroma_area_px2": stromal_s,
            "intertumor_inflammatory_area_px2": inter_i,
            "intertumor_stroma_area_px2": inter_s,
            "mean_focus_gap_um": float(gap_sub["boundary_gap_um"].mean()) if not gap_sub.empty else np.nan,
            "median_focus_gap_um": float(gap_sub["boundary_gap_um"].median()) if not gap_sub.empty else np.nan,
            "max_focus_gap_um": float(gap_sub["boundary_gap_um"].max()) if not gap_sub.empty else np.nan,
        })

    return pd.DataFrame(rows)


def score_master_rois_refine(
    masters: pd.DataFrame,
    foci: pd.DataFrame,
    edges: pd.DataFrame,
    corridors: Dict[int, BaseGeometry],
    region_index: RegionIndex,
    mpp: float,
    min_tissue_fraction: float,
    min_tsr_denom_px2: float,
    min_tils_denom_px2: float,
) -> pd.DataFrame:
    if masters.empty:
        return pd.DataFrame()
    focus_geom = foci.set_index("focus_id")["geometry"].to_dict()
    rows = []

    for m in tqdm(masters.itertuples(), total=len(masters), desc="Scoring Master ROIs", unit="master"):
        master_geom = m.geometry
        fids = [int(x) for x in str(m.focus_ids).split(";") if x]
        eids = [int(x) for x in str(m.edge_ids).split(";") if x]
        tumor_bed = fill_polygon_holes(unary_union([focus_geom[fid] for fid in fids]))
        tumor_bed = make_valid(tumor_bed.intersection(master_geom))
        stromal_comp = make_valid(master_geom.difference(tumor_bed))
        corridor_geom = make_valid(unary_union([corridors[eid] for eid in eids])) if eids else Polygon()
        if corridor_geom is not None and not corridor_geom.is_empty:
            # Inter-tumour immune scoring must exclude the endpoint tumour bed.
            corridor_geom = make_valid(
                corridor_geom.intersection(master_geom).difference(tumor_bed)
            )

        whole = region_index.areas(master_geom)
        intra = region_index.areas(tumor_bed) if tumor_bed and not tumor_bed.is_empty else {c: 0.0 for c in CLASS_ORDER}
        stromal = region_index.areas(stromal_comp) if stromal_comp and not stromal_comp.is_empty else {c: 0.0 for c in CLASS_ORDER}
        inter = region_index.areas(corridor_geom) if corridor_geom and not corridor_geom.is_empty else {c: 0.0 for c in CLASS_ORDER}

        tumor = whole["Tumour"]
        stroma = whole["Stroma"]
        inflam = whole["Inflammatory"]
        necrosis = whole["Necrosis"]
        others = whole["Others"]
        segmented = tumor + stroma + inflam + necrosis + others
        area = float(master_geom.area)
        tissue_fraction = min(1.0, segmented / area) if area > 0 else 0.0
        background = max(0.0, area - segmented)

        raw_denom = tumor + stroma
        tsr_raw = stroma / raw_denom if raw_denom > 0 else np.nan
        tumor_pct = tumor / raw_denom * 100 if raw_denom > 0 else np.nan
        stroma_pct = stroma / raw_denom * 100 if raw_denom > 0 else np.nan

        intra_i = intra["Inflammatory"]
        intra_t = intra["Tumour"]
        stromal_i = stromal["Inflammatory"]
        stromal_s = stromal["Stroma"]
        inter_i = inter["Inflammatory"]
        inter_s = inter["Stroma"]

        itil_denom = intra_t + intra_i
        stil_denom = stromal_s + stromal_i
        intertil_denom = inter_s + inter_i
        itil = intra_i / itil_denom * 100 if itil_denom > 0 else np.nan
        stil_occ = stromal_i / stil_denom * 100 if stil_denom > 0 else np.nan
        intertil = inter_i / intertil_denom * 100 if intertil_denom > 0 else np.nan
        salgado = inflam / stroma * 100 if stroma > 0 else np.nan

        effective_t = tumor + intra_i
        effective_s = stroma + stromal_i
        eff_denom = effective_t + effective_s
        tsr_comp = effective_s / eff_denom if eff_denom > 0 else np.nan

        tsr_reliable = bool(raw_denom >= min_tsr_denom_px2 and tissue_fraction >= min_tissue_fraction)
        stil_reliable = bool(stil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)
        itil_reliable = bool(itil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)
        intertil_reliable = bool(intertil_denom >= min_tils_denom_px2 and tissue_fraction >= min_tissue_fraction)

        gap_sub = edges[
            edges["edge_id"].isin(eids)
        ] if (not edges.empty and eids) else pd.DataFrame()

        rows.append({
            "master_roi_id": int(m.master_roi_id),
            "cluster_id": int(m.master_roi_id),  # backward compatibility
            "focus_ids": m.focus_ids,
            "n_foci": int(m.n_foci),
            "n_intertumor_edges": int(m.n_edges),
            "scoring_method": "exact_vector_refine",
            "master_area_basis": "exact_vector_area",
            "master_vector_area_px2": area,
            "master_area_px2": area,
            "master_area_um2": area * mpp * mpp,
            "master_area_mm2": area * mpp * mpp / 1e6,
            "tumor_area_px2": tumor,
            "stroma_area_px2": stroma,
            "inflammatory_area_px2": inflam,
            "necrosis_area_px2": necrosis,
            "others_area_px2": others,
            "background_area_px2": background,
            "tissue_fraction": tissue_fraction,
            "TSR_tumor_pct": tumor_pct,
            "TSR_stroma_pct": stroma_pct,
            "TSR_stroma_fraction": tsr_raw,
            "TSR_compartment_stroma_fraction": tsr_comp,
            "TSR_reliable": tsr_reliable,
            "sTILs_pct_salgado": salgado,
            "inflammatory_to_stroma_ratio_pct": salgado,
            "sTILs_pct_stromal_occupancy": stil_occ,
            "sTILs_pct": stil_occ,
            "sTILs_reliable": stil_reliable,
            "iTILs_pct_computational": itil,
            "iTILs_reliable": itil_reliable,
            "inter_tumor_sTILs_pct": intertil,
            "inter_tumor_sTILs_reliable": intertil_reliable,
            "intratumoral_inflammatory_area_px2": intra_i,
            "intratumoral_tumor_area_px2": intra_t,
            "stromal_inflammatory_area_px2": stromal_i,
            "stromal_stroma_area_px2": stromal_s,
            "intertumor_inflammatory_area_px2": inter_i,
            "intertumor_stroma_area_px2": inter_s,
            "mean_focus_gap_um": float(gap_sub["boundary_gap_um"].mean()) if not gap_sub.empty else np.nan,
            "median_focus_gap_um": float(gap_sub["boundary_gap_um"].median()) if not gap_sub.empty else np.nan,
            "max_focus_gap_um": float(gap_sub["boundary_gap_um"].max()) if not gap_sub.empty else np.nan,
        })
    return pd.DataFrame(rows)



def compare_fast_vs_refine(fast_scores: pd.DataFrame, refine_scores: pd.DataFrame) -> pd.DataFrame:
    """Create a compact A/B comparison for validation of raster vs exact areas."""
    if fast_scores.empty or refine_scores.empty:
        return pd.DataFrame()
    metrics = [
        "master_area_px2",
        "tumor_area_px2",
        "stroma_area_px2",
        "inflammatory_area_px2",
        "necrosis_area_px2",
        "others_area_px2",
        "TSR_stroma_fraction",
        "TSR_compartment_stroma_fraction",
        "sTILs_pct_stromal_occupancy",
        "iTILs_pct_computational",
        "inter_tumor_sTILs_pct",
    ]
    left = fast_scores[["master_roi_id"] + [m for m in metrics if m in fast_scores.columns]].copy()
    right = refine_scores[["master_roi_id"] + [m for m in metrics if m in refine_scores.columns]].copy()
    merged = left.merge(right, on="master_roi_id", suffixes=("_fast", "_refine"))
    rows = []
    for r in merged.itertuples(index=False):
        mid = int(getattr(r, "master_roi_id"))
        for metric in metrics:
            fk = f"{metric}_fast"
            rk = f"{metric}_refine"
            if not hasattr(r, fk) or not hasattr(r, rk):
                continue
            fv = float(getattr(r, fk))
            rv = float(getattr(r, rk))
            delta = fv - rv if np.isfinite(fv) and np.isfinite(rv) else np.nan
            delta_pct = (100.0 * delta / rv) if np.isfinite(delta) and rv != 0 else np.nan
            rows.append({
                "master_roi_id": mid,
                "metric": metric,
                "fast_value": fv,
                "refine_value": rv,
                "fast_minus_refine": delta,
                "delta_pct_of_refine": delta_pct,
            })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------------
# Lean/report-oriented exports
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den not in (0, 0.0) and np.isfinite(den) else np.nan


def classify_tsr_category(stroma_pct: float, threshold_pct: float = 50.0) -> str:
    """Descriptive TSR category used by the report.

    This is a computational display category, not a validated clinical cut-off.
    ``stroma-high`` is assigned when stroma percentage in the Tumour+Stroma
    compartment is >= ``threshold_pct``; otherwise ``stroma-low``. Missing
    values return ``indeterminate``.
    """
    try:
        value = float(stroma_pct)
    except (TypeError, ValueError):
        return "indeterminate"
    if not np.isfinite(value):
        return "indeterminate"
    return "stroma-high" if value >= float(threshold_pct) else "stroma-low"


def select_core_master_features(
    scores: pd.DataFrame,
    mpp: float,
    tsr_stroma_high_threshold_pct: float = 50.0,
) -> pd.DataFrame:
    """Return only scientifically important + QMD-consumed per-Master-ROI fields.

    ``cluster_id`` is preserved as a compatibility alias for ``master_roi_id``.
    QMD compatibility aliases are added without changing the underlying metrics.
    """
    if scores is None or scores.empty:
        return pd.DataFrame()
    d = scores.copy()
    d["tumor_stroma_ratio"] = np.where(
        d["stroma_area_px2"].to_numpy(float) > 0,
        d["tumor_area_px2"].to_numpy(float) / d["stroma_area_px2"].to_numpy(float),
        np.nan,
    )
    d["tumor_pct_TS_compartment"] = d["TSR_tumor_pct"]
    d["stroma_pct_TS_compartment"] = d["TSR_stroma_pct"]
    d["TSR_category"] = d["stroma_pct_TS_compartment"].apply(
        lambda v: classify_tsr_category(v, tsr_stroma_high_threshold_pct)
    )
    d["sTIL_pct"] = d["sTILs_pct_stromal_occupancy"]
    d["intratumoral_TIL_pct"] = d["iTILs_pct_computational"]

    wanted = [
        # identity/topology
        "master_roi_id", "cluster_id", "focus_ids", "n_foci", "n_intertumor_edges",
        # size/QC
        "master_area_mm2", "tissue_fraction",
        # QMD + core TSR
        "tumor_stroma_ratio", "tumor_pct_TS_compartment", "stroma_pct_TS_compartment",
        "TSR_category", "TSR_stroma_fraction", "TSR_compartment_stroma_fraction", "TSR_reliable",
        # QMD + core immune
        "sTIL_pct", "sTILs_reliable",
        "intratumoral_TIL_pct", "iTILs_reliable",
        "inter_tumor_sTILs_pct", "inter_tumor_sTILs_reliable",
        # spatial context
        "mean_focus_gap_um", "median_focus_gap_um", "max_focus_gap_um",
    ]
    return d[[c for c in wanted if c in d.columns]].copy()


def make_qmd_combined_clump_table(scores: pd.DataFrame, mpp: float) -> pd.DataFrame:
    """Compatibility table for the current QMD's combined-clump section.

    In the new architecture, a Master ROI is the morphology-aware combined tumour
    ecosystem, so each Master ROI becomes one report clump. ``cluster_ids`` is kept
    as the legacy column name and contains the member focus IDs.
    """
    if scores is None or scores.empty:
        return pd.DataFrame()
    d = scores.copy()
    total_i = d["inflammatory_area_px2"].to_numpy(float)
    intra_i = d["intratumoral_inflammatory_area_px2"].to_numpy(float)
    intra_frac = np.divide(intra_i, total_i, out=np.full_like(intra_i, np.nan), where=total_i > 0)
    return pd.DataFrame({
        "cluster_id": d["cluster_id"].astype(int),
        "cluster_ids": d["focus_ids"].astype(str),
        "n_clusters": d["n_foci"].astype(int),
        "combined_clump_sTIL_pct": d["sTILs_pct_stromal_occupancy"],
        "sTIL_pct": d["sTILs_pct_stromal_occupancy"],
        "intratumoral_TIL_pct": d["iTILs_pct_computational"],
        "intratumoral_fraction_of_inflammatory": intra_frac,
        "inflammatory_area_stromal_um2": d["stromal_inflammatory_area_px2"] * mpp * mpp,
        "inflammatory_area_intratumoral_um2": d["intratumoral_inflammatory_area_px2"] * mpp * mpp,
        "inter_tumor_sTIL_pct": d["inter_tumor_sTILs_pct"],
        "tissue_fraction": d["tissue_fraction"],
    })


def select_core_wsi_features(
    summary: dict,
    scores: pd.DataFrame,
    tsr_stroma_high_threshold_pct: float = 50.0,
) -> dict:
    """Keep core science metrics and provide the exact aliases consumed by the QMD."""
    out = {}
    stroma_pct = summary.get("WSI_TSR_stroma_pct", np.nan)
    tumor_pct = summary.get("WSI_TSR_tumor_pct", np.nan)
    ratio = (float(tumor_pct) / float(stroma_pct)) if np.isfinite(tumor_pct) and np.isfinite(stroma_pct) and float(stroma_pct) > 0 else np.nan
    n_reliable = int(scores["sTILs_reliable"].fillna(False).astype(bool).sum()) if (scores is not None and not scores.empty and "sTILs_reliable" in scores.columns) else 0

    # Exact names used by the current QMD.
    out.update({
        "WSI_tumor_stroma_ratio": ratio,
        "WSI_TSR_category": classify_tsr_category(stroma_pct, tsr_stroma_high_threshold_pct),
        "WSI_sTIL_pct": summary.get("WSI_sTILs_pct_stromal_occupancy", np.nan),
        "WSI_neighborhood_sTIL_pct": summary.get("WSI_sTILs_pct_stromal_occupancy", np.nan),
        "WSI_neighborhood_intratumoral_TIL_pct": summary.get("WSI_iTILs_pct_computational", np.nan),
        "WSI_tumor_pct_TS_compartment": tumor_pct,
        "WSI_stroma_pct_TS_compartment": stroma_pct,
        "n_clusters_input": int(summary.get("n_foci", 0)),
        "n_clusters_scored": int(summary.get("n_master_rois", 0)),
        "n_clusters_sTIL_reliable": n_reliable,
        "n_til_neighborhoods": int(summary.get("n_master_rois", 0)),
    })

    # Important new-architecture fields worth retaining even if the current QMD
    # does not yet display all of them.
    for key in [
        "n_master_rois", "n_foci", "n_intertumor_edges",
        "WSI_TSR_stroma_fraction", "WSI_TSR_compartment_stroma_fraction",
        "WSI_sTILs_pct_stromal_occupancy", "WSI_iTILs_pct_computational",
        "WSI_inter_tumor_sTILs_pct", "mean_master_tissue_fraction",
        "total_master_area_mm2",
    ]:
        if key in summary:
            out[key] = summary[key]
    return out


def select_core_edge_features(edges: pd.DataFrame) -> pd.DataFrame:
    if edges is None or edges.empty:
        return pd.DataFrame(columns=["edge_id", "focus_id_1", "focus_id_2", "boundary_gap_um", "path_length_um", "path_efficiency"])
    wanted = [
        "edge_id", "master_roi_id", "focus_id_1", "focus_id_2",
        "boundary_gap_um", "path_length_um", "path_efficiency",
        "corridor_tumor_frac", "corridor_stromal_like_frac",
        "corridor_necrosis_frac", "corridor_tissue_fraction",
    ]
    return edges[[c for c in wanted if c in edges.columns]].copy()


def select_core_inter_roi_features(inter_rois: pd.DataFrame) -> pd.DataFrame:
    if inter_rois is None or inter_rois.empty:
        return pd.DataFrame()
    wanted = [
        "inter_roi_id", "edge_id", "master_roi_id", "focus_id_1", "focus_id_2",
        "path_fraction", "center_x", "center_y",
        "x_min", "y_min", "x_max", "y_max",
        "tumor", "stroma", "inflammatory", "necrosis", "others",
    ]
    return inter_rois[[c for c in wanted if c in inter_rois.columns]].copy()

def area_weighted_wsi(scores: pd.DataFrame) -> dict:
    if scores.empty:
        return {"n_master_rois": 0}
    tumor = float(scores["tumor_area_px2"].sum())
    stroma = float(scores["stroma_area_px2"].sum())
    inflam = float(scores["inflammatory_area_px2"].sum())
    intra_i = float(scores["intratumoral_inflammatory_area_px2"].sum())
    intra_t = float(scores["intratumoral_tumor_area_px2"].sum())
    stromal_i = float(scores["stromal_inflammatory_area_px2"].sum())
    stromal_s = float(scores["stromal_stroma_area_px2"].sum())
    inter_i = float(scores["intertumor_inflammatory_area_px2"].sum())
    inter_s = float(scores["intertumor_stroma_area_px2"].sum())
    raw_denom = tumor + stroma
    eff_t, eff_s = tumor + intra_i, stroma + stromal_i
    return {
        "n_master_rois": int(len(scores)),
        "n_foci": int(scores["n_foci"].sum()),
        "n_intertumor_edges": int(scores["n_intertumor_edges"].sum()),
        "WSI_TSR_stroma_fraction": stroma / raw_denom if raw_denom > 0 else np.nan,
        "WSI_TSR_tumor_pct": tumor / raw_denom * 100 if raw_denom > 0 else np.nan,
        "WSI_TSR_stroma_pct": stroma / raw_denom * 100 if raw_denom > 0 else np.nan,
        "WSI_TSR_compartment_stroma_fraction": eff_s / (eff_s + eff_t) if (eff_s + eff_t) > 0 else np.nan,
        "WSI_sTILs_pct_salgado": inflam / stroma * 100 if stroma > 0 else np.nan,
        "WSI_sTILs_pct_stromal_occupancy": stromal_i / (stromal_s + stromal_i) * 100 if (stromal_s + stromal_i) > 0 else np.nan,
        "WSI_iTILs_pct_computational": intra_i / (intra_t + intra_i) * 100 if (intra_t + intra_i) > 0 else np.nan,
        "WSI_inter_tumor_sTILs_pct": inter_i / (inter_s + inter_i) * 100 if (inter_s + inter_i) > 0 else np.nan,
        "mean_master_tissue_fraction": float(scores["tissue_fraction"].mean()),
        "total_master_area_mm2": float(scores["master_area_mm2"].sum()),
    }


def export_master_geojson(masters: pd.DataFrame, out_path: Path) -> None:
    features = []
    for r in masters.itertuples():
        features.append({
            "type": "Feature",
            "properties": {
                "master_roi_id": int(r.master_roi_id),
                "cluster_id": int(r.master_roi_id),
                "focus_ids": str(r.focus_ids),
                "edge_ids": str(r.edge_ids),
                "n_foci": int(r.n_foci),
                "n_edges": int(r.n_edges),
            },
            "geometry": mapping(r.geometry),
        })
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def export_corridors_geojson(edges: pd.DataFrame, corridors: Dict[int, BaseGeometry], out_path: Path) -> None:
    features = []
    for e in edges.itertuples():
        props = {
            "edge_id": int(e.edge_id),
            "focus_id_1": int(e.focus_id_1),
            "focus_id_2": int(e.focus_id_2),
            "boundary_gap_um": float(e.boundary_gap_um),
            "path_length_um": float(e.path_length_um),
        }
        for name in [
            "path_efficiency", "corridor_tissue_fraction", "corridor_tumor_frac",
            "corridor_stroma_frac", "corridor_inflammatory_frac",
            "corridor_necrosis_frac", "corridor_others_frac",
            "corridor_stromal_like_frac", "corridor_qc_valid", "corridor_qc_reason",
        ]:
            if hasattr(e, name):
                value = getattr(e, name)
                if isinstance(value, (np.bool_, bool)):
                    value = bool(value)
                elif isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating, float)):
                    value = float(value) if np.isfinite(value) else None
                props[name] = value
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(corridors[int(e.edge_id)]),
        })
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))


def _mpl_polygons(geom: BaseGeometry, **kwargs):
    patches = []
    for p in iter_polygon_parts(geom):
        patches.append(MplPolygon(np.asarray(p.exterior.coords), closed=True, **kwargs))
    return patches


def _path_xy(
    path: Sequence[Tuple[int, int]],
    manifest: pd.DataFrame,
) -> List[Tuple[float, float]]:
    if not path:
        return []
    step_x, step_y = infer_tile_step(manifest)
    x0, y0 = float(manifest["wx"].min()), float(manifest["wy"].min())
    return [(x0 + gx * step_x, y0 + gy * step_y) for gx, gy in path]


def plot_cluster_overlay(
    grid: TileSpatialGrid,
    clusters: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    out_png: Path,
    max_panel_rows: int = 14,
) -> None:
    """Create the report-style ``cluster_tils_tsr_overlay.png``.

    The layout intentionally mirrors the earlier ViSpace TSR/sTIL overlay:
    a tissue map on the left, sTIL-coloured scoring-region outlines and C# labels,
    plus a compact per-region score panel on the right.

    Performance note
    ----------------
    The background is rendered from the already-built manifest tile raster using
    the dominant semantic class per tile. Quantitative scoring remains fractional
    (sum of ``frac_*`` values); the winner-take-all colouring is visualization only.
    This avoids loading the large semantic GeoJSON merely to draw the report PNG.
    """
    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.2, 1.6], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    ax_panel = fig.add_subplot(gs[0, 1])
    ax_panel.axis("off")

    # ------------------------------------------------------------------
    # Tissue map: dominant semantic class for display only.
    # Scoring itself continues to use fractional class arrays.
    # ------------------------------------------------------------------
    frac_stack = np.stack([grid.frac[c] for c in CLASS_ORDER], axis=-1)
    dominant = np.argmax(frac_stack, axis=-1)
    rgba = np.ones((grid.ny, grid.nx, 4), dtype=np.float32)
    rgba[..., :3] = 1.0
    rgba[..., 3] = 0.0
    for idx, cls in enumerate(CLASS_ORDER):
        m = grid.valid & (dominant == idx)
        rgba[m, :3] = _hex_to_rgb01(CLASS_COLORS[cls])
        rgba[m, 3] = 0.85

    x_left = grid.x0 - grid.step_x / 2.0
    x_right = grid.x0 + (grid.nx - 0.5) * grid.step_x
    y_top = grid.y0 - grid.step_y / 2.0
    y_bottom = grid.y0 + (grid.ny - 0.5) * grid.step_y
    ax.imshow(
        rgba,
        origin="upper",
        interpolation="nearest",
        extent=[x_left, x_right, y_bottom, y_top],
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Master-ROI outlines and score lookup.
    # ``cluster_id`` is intentionally displayed because it is the downstream
    # compatibility alias of ``master_roi_id``.
    # ------------------------------------------------------------------
    score_map = pd.DataFrame()
    if cluster_scores is not None and not cluster_scores.empty and "cluster_id" in cluster_scores.columns:
        score_map = (
            cluster_scores
            .drop_duplicates(subset="cluster_id", keep="first")
            .set_index("cluster_id")
        )

    panel_rows = []
    for c in clusters.itertuples():
        cid_raw = getattr(c, "cluster_id", None)
        if cid_raw is None:
            cid_raw = getattr(c, "master_roi_id")
        cid = int(cid_raw)
        if score_map.empty or cid not in score_map.index:
            continue
        s = score_map.loc[cid]
        stil_level = _stil_display_level(
            s.get("sTIL_pct", np.nan),
            s.get("sTILs_reliable", False),
        )
        edge_color = TILS_LEVEL_COLORS.get(stil_level, "#888888")

        for patch in _mpl_polygons(c.geometry, fill=False, edgecolor=edge_color, linewidth=3.5):
            patch.set_zorder(10)
            ax.add_patch(patch)

        cx, cy = c.geometry.representative_point().coords[0]
        ax.text(
            cx, cy, f"C{cid}",
            fontsize=9, weight="bold", color="black", ha="center", va="center",
            bbox=dict(
                facecolor="white", alpha=0.80, edgecolor=edge_color,
                linewidth=1.8, boxstyle="round,pad=0.20",
            ),
            zorder=12,
        )

        tumor_pct = s.get("tumor_pct_TS_compartment", np.nan)
        stroma_pct = s.get("stroma_pct_TS_compartment", np.nan)
        try:
            tsr_display = (
                f"{float(tumor_pct):.0f}/{float(stroma_pct):.0f}"
                if np.isfinite(float(tumor_pct)) and np.isfinite(float(stroma_pct))
                else "—"
            )
        except (TypeError, ValueError):
            tsr_display = "—"

        panel_rows.append({
            "cluster_id": cid,
            "edge": edge_color,
            "tsr_display": tsr_display,
            "tsr_category": str(s.get("TSR_category", "indeterminate")),
            "tsr_reliable": bool(s.get("TSR_reliable", False)),
            "stil": s.get("sTIL_pct", np.nan),
            "stil_level": stil_level,
            "stil_reliable": bool(s.get("sTILs_reliable", False)),
            "itil": s.get("intratumoral_TIL_pct", np.nan),
            "itil_reliable": bool(s.get("iTILs_reliable", False)),
            "intertil": s.get("inter_tumor_sTILs_pct", np.nan),
            "intertil_reliable": bool(s.get("inter_tumor_sTILs_reliable", False)),
            "area_mm2": s.get("master_area_mm2", np.nan),
            "tissue_fraction": s.get("tissue_fraction", np.nan),
            "n_foci": s.get("n_foci", np.nan),
        })

    # Whole raster extent keeps the report spatially comparable to the old view.
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)
    ax.set_aspect("equal")
    ax.set_xlabel("WSI x (px)", fontsize=10)
    ax.set_ylabel("WSI y (px)", fontsize=10)
    ax.set_title(
        "Master ROI scoring polygons — TSR & spatial TILs\n"
        "TSR = Tumour%/Stroma%; sTIL = stromal inflammatory occupancy; "
        "iTIL = computational intratumoral occupancy\n"
        "Tissue colours use dominant manifest class for display; quantitative scoring uses fractional tile sums",
        fontsize=9,
    )

    # Two legends, matching the older overlay structure.
    class_handles = [
        Patch(facecolor=CLASS_COLORS[c], edgecolor="none", label=c)
        for c in CLASS_ORDER
    ]
    level_handles = [
        Patch(facecolor="none", edgecolor=col, linewidth=3.0, label=f"sTIL {lvl}")
        for lvl, col in TILS_LEVEL_COLORS.items()
    ]
    leg1 = ax.legend(
        handles=class_handles, loc="upper right", fontsize=8,
        title="Tissue class", framealpha=0.92,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=level_handles, loc="lower right", fontsize=8,
        title="Master ROI outline", framealpha=0.92,
    )

    # ------------------------------------------------------------------
    # Right-side score panel.
    # ------------------------------------------------------------------
    ax_panel.text(0.02, 0.990, "Master ROI scores", fontsize=13, weight="bold", va="top")
    ax_panel.text(
        0.02, 0.955,
        "TSR = Tumour% / Stroma%\n"
        "sTIL = stromal inflammatory occupancy\n"
        "iTIL = computational intratumoral occupancy\n"
        "Inter-TIL = accepted inter-tumour corridor occupancy",
        fontsize=7.5, va="top", color="#444444",
    )

    # If there are many Master ROIs, keep the report legible and show the largest.
    rows_to_show = sorted(
        panel_rows,
        key=lambda r: (
            -(float(r["area_mm2"]) if pd.notna(r["area_mm2"]) else -1.0),
            r["cluster_id"],
        ),
    )
    hidden = max(0, len(rows_to_show) - int(max_panel_rows))
    rows_to_show = rows_to_show[: int(max_panel_rows)]
    rows_to_show = sorted(rows_to_show, key=lambda r: r["cluster_id"])

    y = 0.845
    n = max(1, len(rows_to_show))
    r_gap = min(0.125, max(0.052, 0.77 / n))

    for row in rows_to_show:
        ax_panel.text(
            0.02, y, f"C{row['cluster_id']}",
            fontsize=9.5, weight="bold", va="top",
            bbox=dict(
                facecolor="white", edgecolor=row["edge"],
                linewidth=2.1, boxstyle="round,pad=0.23",
            ),
        )

        tsr_flag = "" if row["tsr_reliable"] else " !"
        stil_flag = "" if row["stil_reliable"] else " !"
        itil_flag = "" if row["itil_reliable"] else " !"
        inter_flag = "" if row["intertil_reliable"] else " !"

        try:
            area_txt = f"{float(row['area_mm2']):.3f} mm²" if np.isfinite(float(row['area_mm2'])) else "—"
        except (TypeError, ValueError):
            area_txt = "—"
        try:
            tissue_txt = f"{100.0 * float(row['tissue_fraction']):.1f}%" if np.isfinite(float(row['tissue_fraction'])) else "—"
        except (TypeError, ValueError):
            tissue_txt = "—"
        try:
            foci_txt = str(int(row["n_foci"])) if np.isfinite(float(row["n_foci"])) else "—"
        except (TypeError, ValueError):
            foci_txt = "—"

        ax_panel.text(
            0.19, y,
            f"TSR {row['tsr_display']} (T%/S%){tsr_flag}  [{row['tsr_category']}]\n"
            f"sTIL {_fmt_overlay_pct(row['stil'])}{stil_flag}  [{row['stil_level']}]\n"
            f"iTIL {_fmt_overlay_pct(row['itil'])}{itil_flag}   "
            f"Inter-TIL {_fmt_overlay_pct(row['intertil'])}{inter_flag}\n"
            f"Foci {foci_txt}   Area {area_txt}   Tissue {tissue_txt}",
            fontsize=8.0, va="top",
        )
        y -= r_gap

    if hidden:
        ax_panel.text(
            0.02, max(0.015, y - 0.01),
            f"Showing {len(rows_to_show)} largest Master ROIs; {hidden} additional region(s) omitted from this panel.",
            fontsize=7.5, color="#555555", va="top",
        )

    ax_panel.set_xlim(0, 1)
    ax_panel.set_ylim(0, 1)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.065, wspace=0.02)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _parse_sample_fractions(value) -> Tuple[float, ...]:
    if value is None:
        return (0.5,)
    if isinstance(value, str):
        vals = [float(x.strip()) for x in value.split(",") if x.strip()]
    else:
        vals = [float(x) for x in value]
    vals = [x for x in vals if 0 <= x <= 1]
    return tuple(vals) if vals else (0.5,)



def run_cluster_tils_tsr_score(wsi_path: str, cfg: PipelineConfig = None) -> dict:
    """Run focus-graph / Master-ROI TIL-TSR scoring for one WSI.

    ``SPATIAL_SCORING_MODE='fast'`` (default)
        * manifest fractional tile raster
        * masked NumPy class-area sums
        * raster corridor QC
        * local sparse SciPy Dijkstra routing
        * no semantic GeoJSON loading/intersection for scoring

    ``SPATIAL_SCORING_MODE='refine'``
        Runs the same fast graph/QC pipeline first, then recomputes final
        accepted Master-ROI compartments with exact semantic-vector
        intersections. Vector exports and the ``cluster_id`` alias are
        identical in both modes.
    """
    if cfg is None:
        cfg = default_cfg

    started = time.perf_counter()
    stage_times: Dict[str, float] = {}

    slide = Path(wsi_path).stem
    base = Path(cfg.OUT_DIR) / slide
    roi_dir = base / "spatial_feature_results" / "tumor_roi_overlay"
    out_dir = base / "spatial_feature_results" / "cluster_tils_tsr_score"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_csv = roi_dir / "tumor_roi_boxes.csv"
    foci_geojson = roi_dir / "tumor_foci.geojson"
    focus_tiles_csv = roi_dir / "tumor_focus_tiles.csv"
    geojson = base / "segmentation" / "segmentation_all_classes.geojson"
    manifest_csv = base / "segmentation" / "manifest.csv"

    scoring_mode = str(getattr(cfg, "SPATIAL_SCORING_MODE", "fast")).strip().lower()
    if scoring_mode not in {"fast", "refine"}:
        raise ValueError("SPATIAL_SCORING_MODE must be 'fast' or 'refine'")

    required = [foci_geojson, focus_tiles_csv, manifest_csv]
    if scoring_mode == "refine":
        required.append(geojson)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    # ---------------------------------------------------------------
    # Important algorithmic configuration only.
    # ---------------------------------------------------------------
    mpp = float(getattr(cfg, "MPP", 0.25))
    max_grid_cells = int(getattr(cfg, "SPATIAL_GRID_MAX_CELLS", 25_000_000))

    # Focus graph
    min_gap_um = float(getattr(cfg, "INTER_TUMOR_MIN_GAP_UM", 50.0))
    max_gap_um = float(getattr(cfg, "INTER_TUMOR_MAX_GAP_UM", 1000.0))
    knn_k = int(getattr(cfg, "INTER_TUMOR_KNN_K", 3))

    # Tissue-aware routing
    corridor_width_um = float(getattr(cfg, "INTER_TUMOR_CORRIDOR_WIDTH_UM", 100.0))
    search_margin_um = float(getattr(cfg, "INTER_TUMOR_SEARCH_MARGIN_UM", 300.0))
    tumor_weight = float(getattr(cfg, "INTER_PATH_TUMOR_WEIGHT", 5.0))
    necrosis_weight = float(getattr(cfg, "INTER_PATH_NECROSIS_WEIGHT", 6.0))
    other_weight = float(getattr(cfg, "INTER_PATH_OTHER_WEIGHT", 1.0))

    # Corridor validation
    corridor_max_tumor = float(getattr(cfg, "INTER_CORRIDOR_MAX_TUMOR_FRAC", 0.20))
    corridor_max_necrosis = float(getattr(cfg, "INTER_CORRIDOR_MAX_NECROSIS_FRAC", 0.30))
    corridor_min_stromal_like = float(getattr(cfg, "INTER_CORRIDOR_MIN_STROMAL_LIKE_FRAC", 0.30))
    corridor_min_tissue = float(getattr(cfg, "INTER_CORRIDOR_MIN_TISSUE_FRACTION", 0.50))
    corridor_min_efficiency = float(getattr(cfg, "INTER_CORRIDOR_MIN_PATH_EFFICIENCY", 0.50))

    # Representative inter-tumour ROI sampling
    inter_roi_size_um = float(getattr(cfg, "INTER_TUMOR_ROI_SIZE_UM", getattr(cfg, "ROI_SIZE_UM", 200.0)))
    sample_fractions = _parse_sample_fractions(getattr(cfg, "INTER_TUMOR_SAMPLE_FRACTIONS", (0.5,)))
    inter_max_tumor = float(getattr(cfg, "INTER_TUMOR_MAX_TUMOR_FRAC", 0.20))
    inter_max_necrosis = float(getattr(cfg, "INTER_TUMOR_MAX_NECROSIS", 0.50))

    # Master ROI + scoring QC
    stroma_margin_um = float(getattr(cfg, "MASTER_STROMA_MARGIN_UM", getattr(cfg, "CLUSTER_BUFFER_UM", 100.0)))
    min_tissue_fraction = float(getattr(cfg, "CLUSTER_MIN_TISSUE_FRACTION", 0.10))
    min_tsr_denom = float(getattr(cfg, "CLUSTER_MIN_TSR_DENOM_PX2", 5000.0))
    min_tils_denom = float(getattr(cfg, "CLUSTER_MIN_TILS_DENOM_PX2", 5000.0))
    min_region_area = float(getattr(cfg, "CLUSTER_MIN_POLYGON_AREA_PX2", 1.0))
    tsr_stroma_high_threshold_pct = float(getattr(cfg, "TSR_STROMA_HIGH_THRESHOLD_PCT", 50.0))

    # ---------------------------------------------------------------
    # Inputs + tile spatial grid
    # ---------------------------------------------------------------
    t = time.perf_counter()
    foci = load_foci(foci_geojson)
    focus_tiles = pd.read_csv(focus_tiles_csv)
    manifest = pd.read_csv(manifest_csv)
    tumor_rois = load_roi_boxes(roi_csv)
    stage_times["load_core_inputs_s"] = time.perf_counter() - t

    if foci.empty:
        raise RuntimeError("No tumour foci available for scoring")

    t = time.perf_counter()
    grid = TileSpatialGrid(manifest, mpp=mpp, max_cells=max_grid_cells)
    stage_times["build_tile_raster_s"] = time.perf_counter() - t
    corridor_width_px = corridor_width_um / mpp
    # Authoritative focus ownership on the same tile lattice used for fast QC.
    focus_owner = grid.focus_owner_array(focus_tiles)

    # ---------------------------------------------------------------
    # Candidate focus pairs -> exact gap -> sparse Dijkstra route
    # ---------------------------------------------------------------
    t = time.perf_counter()
    candidate_edges, candidate_corridors, candidate_paths = build_focus_edges_and_corridors(
        foci,
        focus_tiles,
        grid,
        mpp,
        min_gap_um=min_gap_um,
        max_gap_um=max_gap_um,
        knn_k=knn_k,
        corridor_width_um=corridor_width_um,
        search_margin_um=search_margin_um,
        tumor_weight=tumor_weight,
        necrosis_weight=necrosis_weight,
        other_weight=other_weight,
    )
    stage_times["focus_graph_dijkstra_s"] = time.perf_counter() - t

    # Foci separated by less than the minimum inter-tumour gap are too close
    # to define a meaningful stromal corridor, but they should still belong to
    # the same Master ROI. Keep those direct merges separate from true
    # inter-tumour corridor edges.
    close_merges = build_close_focus_merges(
        foci, mpp=mpp, min_gap_um=min_gap_um, knn_k=knn_k
    )

    if candidate_edges.empty and close_merges.empty:
        warnings.warn(
            "No focus relationships survived distance/path construction; "
            "each focus will remain its own Master ROI.",
            RuntimeWarning,
        )

    # ---------------------------------------------------------------
    # Corridor QC from tile masks
    # ---------------------------------------------------------------
    t = time.perf_counter()
    all_edges = score_corridor_qc(
        candidate_edges,
        candidate_paths,
        grid,
        focus_owner,
        corridor_width_px=corridor_width_px,
        max_tumor_frac=corridor_max_tumor,
        max_necrosis_frac=corridor_max_necrosis,
        min_stromal_like_frac=corridor_min_stromal_like,
        min_tissue_fraction=corridor_min_tissue,
        min_path_efficiency=corridor_min_efficiency,
    )
    edges = (
        all_edges[all_edges["corridor_qc_valid"]].copy().reset_index(drop=True)
        if not all_edges.empty else all_edges.copy()
    )
    corridors, paths = _filter_edge_geometry_maps(edges, candidate_corridors, candidate_paths)
    stage_times["corridor_qc_raster_s"] = time.perf_counter() - t

    if not all_edges.empty and edges.empty:
        suffix = (
            " Very-close foci may still share a Master ROI via close-focus merges."
            if not close_merges.empty else
            " Foci will otherwise be scored as separate Master ROIs."
        )
        warnings.warn(
            "All routed inter-tumour edges were rejected by corridor QC." + suffix,
            RuntimeWarning,
        )

    # ---------------------------------------------------------------
    # Accepted graph -> Master ROI vectors (unchanged external schema)
    # ---------------------------------------------------------------
    t = time.perf_counter()
    connectivity_parts = []
    if not edges.empty:
        connectivity_parts.append(edges[["focus_id_1", "focus_id_2"]].copy())
    if not close_merges.empty:
        connectivity_parts.append(close_merges[["focus_id_1", "focus_id_2"]].copy())
    connectivity_edges = (
        pd.concat(connectivity_parts, ignore_index=True)
        if connectivity_parts else pd.DataFrame(columns=["focus_id_1", "focus_id_2"])
    )

    master_map = assign_master_rois(foci, connectivity_edges)
    if not edges.empty:
        edges["master_roi_id"] = edges["focus_id_1"].map(master_map).astype(int)
    if not all_edges.empty:
        all_edges["master_roi_id"] = all_edges["focus_id_1"].map(master_map).astype(int)
    if not close_merges.empty:
        close_merges["master_roi_id"] = close_merges["focus_id_1"].map(master_map).astype(int)

    masters = build_master_geometries(
        foci,
        edges,
        corridors,
        master_map,
        stroma_margin_px=stroma_margin_um / mpp,
        tissue_union=None,
    )
    masters = make_nonoverlapping_master_rois(masters)
    stage_times["master_roi_vector_build_s"] = time.perf_counter() - t

    # Representative inter-tumour image samples use the same tile raster.
    t = time.perf_counter()
    inter_rois = build_inter_tumor_rois(
        edges,
        paths,
        grid,
        mpp,
        inter_roi_size_um,
        sample_fractions,
        max_tumor_frac=inter_max_tumor,
        max_necrosis=inter_max_necrosis,
        master_map=master_map,
    )
    stage_times["inter_roi_sampling_s"] = time.perf_counter() - t

    if not edges.empty and inter_rois.empty:
        warnings.warn(
            "Valid focus edges exist, but no representative inter-tumour ROI passed "
            "the tumour/necrosis window filters. Master-ROI scoring still proceeds.",
            RuntimeWarning,
        )

    # ---------------------------------------------------------------
    # Fast Master scoring: masked fractional tile sums
    # ---------------------------------------------------------------
    t = time.perf_counter()
    fast_scores = score_master_rois_fast(
        masters,
        foci,
        edges,
        paths,
        grid,
        corridor_width_px=corridor_width_px,
        mpp=mpp,
        min_tissue_fraction=min_tissue_fraction,
        min_tsr_denom_px2=min_tsr_denom,
        min_tils_denom_px2=min_tils_denom,
    )
    stage_times["master_scoring_raster_s"] = time.perf_counter() - t

    scores = fast_scores
    refine_scores = pd.DataFrame()
    comparison = pd.DataFrame()

    # ---------------------------------------------------------------
    # Optional exact vector refinement for final accepted compartments
    # ---------------------------------------------------------------
    if scoring_mode == "refine":
        t = time.perf_counter()
        regions = load_regions(geojson, min_region_area)
        if regions.empty:
            raise RuntimeError("No segmentation regions loaded for refine mode")
        region_index = RegionIndex(regions)
        stage_times["load_refine_vectors_index_s"] = time.perf_counter() - t

        t = time.perf_counter()
        refine_scores = score_master_rois_refine(
            masters,
            foci,
            edges,
            corridors,
            region_index,
            mpp,
            min_tissue_fraction=min_tissue_fraction,
            min_tsr_denom_px2=min_tsr_denom,
            min_tils_denom_px2=min_tils_denom,
        )
        comparison = compare_fast_vs_refine(fast_scores, refine_scores)
        scores = refine_scores
        stage_times["master_scoring_vector_refine_s"] = time.perf_counter() - t

    wsi_summary = area_weighted_wsi(scores)
    wsi_summary.update({
        "spatial_scoring_mode": scoring_mode,
        "mpp": mpp,
        "tile_grid_nx": grid.nx,
        "tile_grid_ny": grid.ny,
        "tile_grid_cells": grid.n_cells,
        "tile_step_x_px": grid.step_x,
        "tile_step_y_px": grid.step_y,
        "tile_area_px2_used_fast": grid.tile_area_px2,
        "inter_tumor_min_gap_um": min_gap_um,
        "inter_tumor_max_gap_um": max_gap_um,
        "master_stroma_margin_um": stroma_margin_um,
        "n_inter_tumor_rois": int(len(inter_rois)),
        "n_intertumor_edges_candidate": int(len(all_edges)),
        "n_intertumor_edges_rejected_qc": int(len(all_edges) - len(edges)),
        "n_close_focus_merges": int(len(close_merges)),
        "inter_corridor_max_tumor_frac": corridor_max_tumor,
        "inter_corridor_max_necrosis_frac": corridor_max_necrosis,
        "inter_corridor_min_stromal_like_frac": corridor_min_stromal_like,
        "inter_corridor_min_tissue_fraction": corridor_min_tissue,
        "inter_corridor_min_path_efficiency": corridor_min_efficiency,
    })

    # ---------------------------------------------------------------
    # Minimal output contract.
    #
    # Three logical outputs are retained:
    #   1) Master ROI geometry
    #   2) Per-Master ROI TSR/TIL table
    #   3) WSI summary
    #
    # The first two are also written under legacy filenames required by
    # downstream ViSpace stages and run_vispace.py. Therefore five physical
    # files are written, but only three distinct tabular/vector datasets are
    # produced; the standard report/QC overlay PNG is retained separately.
    # ---------------------------------------------------------------
    master_geojson = out_dir / "master_roi_polygons.geojson"
    legacy_cluster_geojson = out_dir / "cluster_scoring_polygons.geojson"
    master_csv = out_dir / "tils_tsr_by_master_roi.csv"
    legacy_cluster_csv = out_dir / "tils_tsr_by_cluster.csv"
    wsi_csv = out_dir / "tils_tsr_wsi_summary.csv"
    overlay_png = out_dir / "cluster_tils_tsr_overlay.png"

    t = time.perf_counter()

    # Same Master-ROI geometry, two filenames.
    export_master_geojson(masters, master_geojson)
    export_master_geojson(masters, legacy_cluster_geojson)

    core_scores = select_core_master_features(
        scores,
        mpp,
        tsr_stroma_high_threshold_pct=tsr_stroma_high_threshold_pct,
    )
    core_wsi_summary = select_core_wsi_features(
        wsi_summary,
        scores,
        tsr_stroma_high_threshold_pct=tsr_stroma_high_threshold_pct,
    )

    # Same per-Master table, two filenames.
    core_scores.to_csv(master_csv, index=False)
    core_scores.to_csv(legacy_cluster_csv, index=False)

    # Slide-level summary.
    pd.DataFrame([core_wsi_summary]).to_csv(wsi_csv, index=False)

    # Standard report/QC overlay retained for the QMD.
    # The visual style mirrors the older cluster overlay while using the fast
    # manifest raster as its tissue background.
    plot_cluster_overlay(
        grid=grid,
        clusters=masters,
        cluster_scores=core_scores,
        out_png=overlay_png,
    )

    stage_times["write_outputs_s"] = time.perf_counter() - t
    stage_times["total_s"] = time.perf_counter() - started

    print(f"  Spatial scoring mode      : {scoring_mode}")
    print(f"  Tile grid                 : {grid.nx} x {grid.ny} ({grid.n_cells:,} cells)")
    print(f"  Tumour foci               : {len(foci)}")
    print(f"  Candidate inter-edges     : {len(all_edges)}")
    print(f"  Valid inter-tumour edges  : {len(edges)}")
    print(f"  Close-focus merges        : {len(close_merges)}")
    print(f"  Master ROIs               : {len(masters)}")
    print(f"  Inter-tumour ROIs         : {len(inter_rois)}")
    print("  Wrote minimal output contract:")
    print(f"    {master_geojson.name}")
    print(f"    {legacy_cluster_geojson.name}  [legacy alias]")
    print(f"    {master_csv.name}")
    print(f"    {legacy_cluster_csv.name}  [legacy alias]")
    print(f"    {wsi_csv.name}")
    print(f"    {overlay_png.name}")

    return {
        "slide_name": slide,
        "scoring_mode": scoring_mode,
        "master_geojson": str(master_geojson),
        "cluster_geojson": str(legacy_cluster_geojson),
        "master_csv": str(master_csv),
        "cluster_csv": str(legacy_cluster_csv),
        "wsi_csv": str(wsi_csv),
        "overlay_png": str(overlay_png),
        "wsi_summary": core_wsi_summary,
    }



def main(argv=None) -> None:
    from .config import config_from_args
    cfg, _ = config_from_args(argv)
    if not cfg.WSI_PATH or cfg.WSI_PATH == "your data path":
        raise SystemExit("--wsi-path is required")
    run_cluster_tils_tsr_score(wsi_path=cfg.WSI_PATH, cfg=cfg)


if __name__ == "__main__":
    main()
