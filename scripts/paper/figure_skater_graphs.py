"""
SKATER graph figure - Belo Horizonte

Builds Figure 3 of the paper (sections/methodology.tex,
\\label{fig:skater_graphs}): the weighted spatial adjacency graph
(left) and its minimum spanning tree (right), both drawn over sector
centroids on top of the census-sector outlines. Static matplotlib
rendering -- with 5,166 nodes, a real street basemap adds visual
noise without adding information for this abstract graph-structure
diagram (unlike Figure 2, which follows a real-basemap/folium
convention -- see project memory).

Inputs:
  data/processed/bh_sectors_2022_with_populations.geojson
  data/dvc/add_population_info/bh_sectors_2022_centroids.csv
  results/skater/spearman_100/adjacency_edges.csv
  results/skater/spearman_100/mst_edges.csv
Output:
  6a441e20c1f1a66c183b3c38/Figures/figure_skater_graphs.pdf
"""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

SECTORS_GEOJSON = Path("data/processed/bh_sectors_2022_with_populations.geojson")
CENTROIDS_CSV = Path("data/dvc/add_population_info/bh_sectors_2022_centroids.csv")
ADJACENCY_CSV = Path("results/skater/spearman_100/adjacency_edges.csv")
MST_CSV = Path("results/skater/spearman_100/mst_edges.csv")

OUTPUT_PATH = Path("6a441e20c1f1a66c183b3c38/Figures/figure_skater_graphs.pdf")

NODE_COLOR = "#08519c"
ADJACENCY_EDGE_COLOR = "#999999"
MST_EDGE_COLOR = "#B0302A"
FIGSIZE = (10, 5.3)
MAP_PADDING_FRAC = 0.02

# =============================================================================
# DATA LOADING
# =============================================================================


def load_sectors() -> gpd.GeoDataFrame:
    """Load BH census sector polygons.

    Returns:
        GeoDataFrame with CD_SETOR (str) and geometry columns.
    """
    gdf = gpd.read_file(SECTORS_GEOJSON)
    gdf["CD_SETOR"] = gdf["CD_SETOR"].astype(str)
    return gdf


def load_centroids() -> pd.DataFrame:
    """Load per-sector centroid coordinates.

    Returns:
        DataFrame with columns [centroid_latitude,
        centroid_longitude], indexed by CD_SETOR (str).
    """
    centroids = pd.read_csv(
        CENTROIDS_CSV,
        usecols=["CD_SETOR", "centroid_latitude", "centroid_longitude"],
        dtype={"CD_SETOR": str},
    )
    return centroids.set_index("CD_SETOR")


def load_edges(path: Path) -> pd.DataFrame:
    """Load a SKATER edge list (adjacency or MST).

    Args:
        path: Path to an edges CSV with at least [src, dst] columns.

    Returns:
        DataFrame with [src, dst] as strings.
    """
    edges = pd.read_csv(path, dtype={"src": str, "dst": str})
    return edges[["src", "dst"]]


# =============================================================================
# PLOTTING
# =============================================================================


def style_map_panel(
    ax: Axes, bounds: np.ndarray, mean_lat: float, title: str, caption: str
) -> None:
    """Apply shared framing, aspect, title and caption to a map panel.

    Args:
        ax: Target axes.
        bounds: (minx, miny, maxx, maxy) shared across all panels.
        mean_lat: Mean latitude, used for a geographic aspect ratio.
        title: Bold title drawn above the map.
        caption: Caption drawn below the map.
    """
    minx, miny, maxx, maxy = bounds
    pad_x = (maxx - minx) * MAP_PADDING_FRAC
    pad_y = (maxy - miny) * MAP_PADDING_FRAC
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)
    ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.text(
        0.5, -0.03, caption,
        transform=ax.transAxes,
        ha="center", va="top", fontsize=9,
    )


def plot_graph_panel(
    ax: Axes,
    sectors: gpd.GeoDataFrame,
    centroids: pd.DataFrame,
    edges: pd.DataFrame,
    edge_color: str,
    edge_lw: float,
    edge_alpha: float,
    node_size: float,
) -> None:
    """Draw sector outlines with a graph (nodes + edges) overlaid.

    Args:
        ax: Target axes.
        sectors: Sector polygons, from load_sectors().
        centroids: Per-sector centroid coordinates, from
            load_centroids().
        edges: Edge list (src, dst sector ids), from load_edges().
        edge_color: Line color for edges.
        edge_lw: Line width for edges.
        edge_alpha: Line opacity for edges.
        node_size: Marker size for centroid nodes.
    """
    sectors.plot(ax=ax, color="#F2F2F2", edgecolor="#CCCCCC", linewidth=0.15)

    src_xy = centroids.loc[edges["src"], ["centroid_longitude", "centroid_latitude"]].to_numpy()
    dst_xy = centroids.loc[edges["dst"], ["centroid_longitude", "centroid_latitude"]].to_numpy()
    segments = np.stack([src_xy, dst_xy], axis=1)
    ax.add_collection(
        LineCollection(segments, colors=edge_color, linewidths=edge_lw, alpha=edge_alpha)
    )

    ax.scatter(
        centroids["centroid_longitude"], centroids["centroid_latitude"],
        s=node_size, color=NODE_COLOR, linewidths=0, zorder=3,
    )


def build_figure() -> None:
    """Assemble the two-panel figure and save it to OUTPUT_PATH."""
    sectors = load_sectors()
    centroids = load_centroids()
    adjacency_edges = load_edges(ADJACENCY_CSV)
    mst_edges = load_edges(MST_CSV)

    bounds = sectors.total_bounds
    mean_lat = float((bounds[1] + bounds[3]) / 2.0)

    mpl.rcParams.update({"font.size": 9})
    fig, (ax_adj, ax_mst) = plt.subplots(1, 2, figsize=FIGSIZE)

    plot_graph_panel(
        ax_adj, sectors, centroids, adjacency_edges,
        edge_color=ADJACENCY_EDGE_COLOR, edge_lw=0.25, edge_alpha=0.6, node_size=1.5,
    )
    style_map_panel(
        ax_adj, bounds, mean_lat,
        "Weighted spatial adjacency graph",
        f"{len(adjacency_edges)} edges",
    )

    plot_graph_panel(
        ax_mst, sectors, centroids, mst_edges,
        edge_color=MST_EDGE_COLOR, edge_lw=0.4, edge_alpha=0.9, node_size=1.5,
    )
    style_map_panel(
        ax_mst, bounds, mean_lat,
        "Minimum spanning tree",
        f"{len(mst_edges)} edges",
    )

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)

    print(f"Sectors (nodes): {len(centroids)}")
    print(f"Adjacency edges: {len(adjacency_edges)}")
    print(f"MST edges: {len(mst_edges)}")
    print(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_figure()
