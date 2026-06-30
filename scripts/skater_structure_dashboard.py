"""Neighbourhood + MST structure map for SKATER.

Generates results/skater/dashboard_structure.html — a single interactive
Plotly map with four toggle-able layers:

  1. Sector boundaries  — polygon outlines for all 5166 census sectors
  2. Adjacency edges    — Queen contiguity graph (sectors share any boundary point)
  3. MST edges          — Minimum Spanning Tree built from egg_corr_dist weights
  4. Sector centroids   — centroid dot per sector (hover shows CD_SETOR code)

Use the buttons (top-left) to isolate any combination of layers.
The MST skeleton reveals which spatial connections SKATER will cut
during the pruning phase.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
RESULTS_BASE = Path("results/skater")
CENTROIDS_PATH = Path(
    "data/dvc/add_population_info/sector_centroids_with_idw.csv"
)
GEOJSON_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)
BH_CENTER = {"lat": -19.917, "lon": -43.934}


# ── Data loaders ──────────────────────────────────────────────────────

def _load_centroids() -> pd.DataFrame:
    """Load unique (sector_id, centroid_latitude, centroid_longitude).

    Reads from the IDW eggs CSV which has one row per sector per biweek.
    Deduplication gives exactly one centroid per sector.

    Returns:
        DataFrame indexed by sector_id with columns
        [centroid_latitude, centroid_longitude].
    """
    df = pd.read_csv(
        CENTROIDS_PATH,
        usecols=["CD_SETOR", "centroid_latitude", "centroid_longitude"],
        dtype={"CD_SETOR": str},
    ).rename(columns={"CD_SETOR": "sector_id"})
    return df.drop_duplicates("sector_id").set_index("sector_id")


def _load_geojson() -> dict:
    """Load the BH census sector GeoJSON FeatureCollection from disk."""
    with open(GEOJSON_PATH) as fh:
        return json.load(fh)


def _load_edges(
    results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the adjacency and MST edge lists saved by src/skater/run.py.

    Args:
        results_dir: Run-specific output directory (results/skater/<run_label>/).

    Returns:
        adj: DataFrame with columns [src, dst] — all Queen contiguity edges.
        mst: DataFrame with columns [src, dst, weight] — MST edges with
             egg_corr_dist weights (0 = identical egg dynamics, 2 = max diff).
    """
    adj = pd.read_csv(
        results_dir / "adjacency_edges.csv",
        dtype={"src": str, "dst": str},
    )
    mst = pd.read_csv(
        results_dir / "mst_edges.csv",
        dtype={"src": str, "dst": str},
    )
    return adj, mst


# ── Geometry helpers ──────────────────────────────────────────────────

def _sector_boundary_latlons(
    geojson: dict,
) -> tuple[list[float | None], list[float | None]]:
    """Extract all sector polygon exterior rings as None-separated coordinate lists.

    Plotly Scattermap draws a continuous line through all coordinates.
    Inserting None between polygons breaks the line so each sector is
    drawn as a closed ring instead of connecting to its neighbours.

    Handles both Polygon and MultiPolygon geometries.

    Args:
        geojson: GeoJSON FeatureCollection with sector polygon geometries.

    Returns:
        (lats, lons): Two flat lists, each with None separators between sectors.
    """
    lats: list[float | None] = []
    lons: list[float | None] = []
    for feat in geojson["features"]:
        geom = feat["geometry"]
        # GeoJSON coordinates are [lon, lat] — note reversed order vs Plotly
        if geom["type"] == "Polygon":
            rings = [geom["coordinates"][0]]      # outer ring only
        else:
            rings = [poly[0] for poly in geom["coordinates"]]
        for ring in rings:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
            # Sentinel to break the line between polygons
            lons.append(None)
            lats.append(None)
    return lats, lons


def _edge_latlons(
    edges: pd.DataFrame,
    cent_lat: pd.Series,
    cent_lon: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Build lat/lon coordinate arrays for a set of graph edges.

    Each edge (src → dst) becomes three entries: [src_coord, dst_coord, None].
    The None acts as a line-break so Plotly draws separate segments instead
    of one continuous path through all nodes.

    This vectorised approach avoids iterrows() and handles 16k+ edges fast.

    Args:
        edges:    DataFrame with columns [src, dst] of sector IDs.
        cent_lat: Series mapping sector_id → centroid latitude.
        cent_lon: Series mapping sector_id → centroid longitude.

    Returns:
        (lats, lons): Flat numpy arrays with None separators.
    """
    src_lat = edges["src"].map(cent_lat).to_numpy(dtype=object)
    src_lon = edges["src"].map(cent_lon).to_numpy(dtype=object)
    dst_lat = edges["dst"].map(cent_lat).to_numpy(dtype=object)
    dst_lon = edges["dst"].map(cent_lon).to_numpy(dtype=object)
    none_col = np.full(len(edges), None)
    lats = np.stack([src_lat, dst_lat, none_col], axis=1).ravel()
    lons = np.stack([src_lon, dst_lon, none_col], axis=1).ravel()
    return lats, lons


# ── Figure builder ────────────────────────────────────────────────────

def build_structure_figure(
    adj: pd.DataFrame,
    mst: pd.DataFrame,
    cent: pd.DataFrame,
    geojson: dict,
    mst_cost: str = "",
) -> go.Figure:
    """Build the four-layer structure map as a Plotly Figure.

    Layer order (traces listed in render order, bottom to top):
      0. Sector boundaries — thin dark polygon outlines
      1. Adjacency edges   — grey lines connecting neighbour centroids
      2. MST edges         — blue lines (the backbone SKATER will prune)
      3. Sector centroids  — red dots (hover shows sector ID)

    Toggle buttons in the layout control which layers are visible.

    Args:
        adj:      Queen contiguity edges DataFrame [src, dst].
        mst:      MST edges DataFrame [src, dst, weight].
        cent:     Centroid DataFrame indexed by sector_id.
        geojson:  GeoJSON FeatureCollection for boundary extraction.
        mst_cost: Cost function name to show in the figure title, e.g.
                  'egg_corr_dist'.  Empty → no suffix.

    Returns:
        A go.Figure ready to be written to HTML.
    """
    cent_lat = cent["centroid_latitude"]
    cent_lon = cent["centroid_longitude"]

    # ── Layer 0: sector polygon boundaries ────────────────────────────
    sec_lats, sec_lons = _sector_boundary_latlons(geojson)
    trace_sectors = go.Scattermap(
        lat=sec_lats,
        lon=sec_lons,
        mode="lines",
        line={"width": 0.8, "color": "rgba(80,80,80,0.9)"},
        name="Sector boundaries",
        hoverinfo="none",
        visible=True,
    )

    # ── Layer 1: Queen adjacency edges ────────────────────────────────
    adj_lats, adj_lons = _edge_latlons(adj, cent_lat, cent_lon)
    trace_adj = go.Scattermap(
        lat=list(adj_lats),
        lon=list(adj_lons),
        mode="lines",
        line={"width": 0.8, "color": "rgba(100,100,100,0.9)"},
        name="Adjacency (Queen)",
        hoverinfo="none",
        visible=True,
    )

    # ── Layer 2: MST edges ────────────────────────────────────────────
    # Single colour — weight encoding kept simple for readability.
    # Lower weight = more similar egg dynamics = preferred by Prim.
    mst_lats, mst_lons = _edge_latlons(mst, cent_lat, cent_lon)
    trace_mst = go.Scattermap(
        lat=list(mst_lats),
        lon=list(mst_lons),
        mode="lines",
        line={"width": 1.2, "color": "rgba(31,119,180,0.7)"},
        name="MST edges",
        hoverinfo="none",
        visible=True,
    )

    # ── Layer 3: sector centroids ─────────────────────────────────────
    # Only sectors that appear in the adjacency graph (i.e. have data)
    valid = cent.index.isin(
        pd.concat([adj["src"], adj["dst"]]).unique()
    )
    c_sub = cent[valid]
    trace_pts = go.Scattermap(
        lat=c_sub["centroid_latitude"].tolist(),
        lon=c_sub["centroid_longitude"].tolist(),
        mode="markers",
        marker={"size": 6, "color": "rgba(200,50,50,0.75)"},
        name="Sector centroids",
        hovertemplate="%{text}<extra></extra>",
        text=c_sub.index.tolist(),
        visible=True,
    )

    # ── Assemble figure ───────────────────────────────────────────────
    # Trace order must match the visibility lists in the buttons below
    fig = go.Figure(data=[trace_sectors, trace_adj, trace_mst, trace_pts])
    fig.update_layout(
        map={
            "style": "open-street-map",
            "zoom": 10.5,
            "center": BH_CENTER,
        },
        title=(
            f"Neighbourhood structure [{mst_cost or 'egg_corr_dist'}]"
            f" — {adj.shape[0]:,} adjacency edges, "
            f"{mst.shape[0]:,} MST edges, "
            f"{len(geojson['features']):,} sectors"
        ),
        height=900,
        margin={"t": 60, "b": 20, "l": 20, "r": 20},
        # Toggle buttons: visibility list = [sectors, adj, mst, centroids]
        updatemenus=[{
            "type": "buttons",
            "direction": "right",
            "showactive": True,
            "x": 0.01,
            "xanchor": "left",
            "y": 1.04,
            "yanchor": "top",
            "buttons": [
                {
                    "label": "All layers",
                    "method": "update",
                    "args": [{"visible": [True, True, True, True]}],
                },
                {
                    "label": "Sectors + Adjacency",
                    "method": "update",
                    "args": [{"visible": [True, True, False, True]}],
                },
                {
                    "label": "Sectors + MST",
                    "method": "update",
                    "args": [{"visible": [True, False, True, True]}],
                },
                {
                    "label": "Sectors only",
                    "method": "update",
                    "args": [{"visible": [True, False, False, False]}],
                },
            ],
        }],
        legend={"x": 0.01, "y": 0.97, "bgcolor": "rgba(255,255,255,0.8)"},
    )
    return fig


# ── Entry point ───────────────────────────────────────────────────────

def main() -> None:
    """Load data, build figure, write HTML dashboard."""
    # ── Read mst_cost and run_label from params ───────────────────────
    with open("params.yaml") as fh:
        _params = yaml.safe_load(fh).get("skater", {})
    mst_cost = _params.get("mst_cost", "egg_corr_dist")
    run_label = _params.get("run_label", "default")
    results_dir = RESULTS_BASE / run_label

    logger.info("Loading data… (run=%s)", run_label)
    cent = _load_centroids()
    adj, mst = _load_edges(results_dir)
    geojson = _load_geojson()
    logger.info(
        "adj=%d edges, mst=%d edges, %d centroids, %d sectors",
        len(adj), len(mst), len(cent), len(geojson["features"]),
    )
    fig = build_structure_figure(adj, mst, cent, geojson, mst_cost=mst_cost)
    out = results_dir / "dashboard_structure.html"
    fig.write_html(str(out))
    logger.info("Saved %s (%.1f MB)", out, out.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
