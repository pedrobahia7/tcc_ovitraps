"""Queen contiguity adjacency graph via shapely 2.x STRtree.

Two census sectors are considered neighbours (Queen criterion) if their
polygon boundaries share any point — an edge, a corner, or anything in
between.  This matches the standard spatial-weights definition used in
PySAL/GeoDa.

Implementation note: shapely 2.x STRtree.query(geom, predicate=…) uses
a GEOS-native predicate test, which is ~10× faster than testing each
pair of polygons individually.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)


def _to_shape(geom_dict: dict) -> Polygon | MultiPolygon:
    """Convert a GeoJSON geometry dict to a shapely geometry object."""
    return shape(geom_dict)


def build_adjacency(
    geojson: dict,
    sector_filter: set[str] | None = None,
) -> nx.Graph:
    """Build a Queen contiguity graph from a GeoJSON FeatureCollection.

    Each node is a census sector (CD_SETOR string).  An edge is added
    between two sectors whenever their polygon boundaries intersect —
    i.e., they share at least one point (edge or corner contact).

    If the resulting graph has disconnected components (can happen for
    sectors that are topological islands — e.g. enclaves), they are
    bridged to the main component via the nearest centroid.  This is
    logged as a WARNING so the operator is aware.

    Args:
        geojson:       GeoJSON FeatureCollection with polygon geometries.
                       Expected property key: 'CD_SETOR'.
        sector_filter: If provided, only sectors in this set are included.
                       Sectors absent from the data matrices should be
                       excluded to keep the MST consistent.

    Returns:
        An undirected networkx Graph with sector IDs as nodes.
    """
    features = geojson["features"]
    logger.info("Building adjacency for %d GeoJSON features…", len(features))

    # ── Extract shapes, respecting the optional filter ────────────────
    sector_ids: list[str] = []
    geometries: list[Polygon | MultiPolygon] = []
    for feat in features:
        sid = str(feat["properties"]["CD_SETOR"])
        if sector_filter is not None and sid not in sector_filter:
            continue
        sector_ids.append(sid)
        geometries.append(_to_shape(feat["geometry"]))

    # ── Build STRtree on boundary geometries ──────────────────────────
    # Boundary = 1-D ring (edge) of the polygon; two sectors sharing a
    # boundary point will have intersecting boundaries.
    boundaries = [g.boundary for g in geometries]
    tree = STRtree(boundaries)

    # ── Populate adjacency graph ──────────────────────────────────────
    G: nx.Graph = nx.Graph()
    for sid in sector_ids:
        G.add_node(sid)

    for i, bnd in enumerate(boundaries):
        # query returns indices of all boundaries that intersect bnd
        candidates = tree.query(bnd, predicate="intersects")
        for j in candidates:
            if j <= i:
                # Skip self-intersection and duplicate pairs (i,j) == (j,i)
                continue
            G.add_edge(sector_ids[i], sector_ids[j])

    logger.info(
        "Adjacency: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    # ── Handle disconnected components ────────────────────────────────
    n_comp = nx.number_connected_components(G)
    if n_comp > 1:
        logger.warning(
            "%d disconnected components — bridging via nearest centroid",
            n_comp,
        )
        G = _connect_components(G, geometries, sector_ids)  # noqa: N806

    return G


def _connect_components(
    graph: nx.Graph,
    geometries: list[Polygon | MultiPolygon],
    sector_ids: list[str],
) -> nx.Graph:
    """Bridge disconnected components by adding nearest-centroid edges.

    For each component that is not the largest (main) component, finds
    the pair of sectors (one from the small component, one from the main)
    with the smallest centroid-to-centroid distance and adds that edge.

    This ensures the final adjacency graph — and therefore the MST — is
    always connected, which is a prerequisite for SKATER.

    Args:
        graph:      Adjacency graph that may have multiple components.
        geometries: Shapely geometries aligned with sector_ids.
        sector_ids: List of sector ID strings, same order as geometries.

    Returns:
        The same graph with bridge edges added (mutated in-place).
    """
    idx_map = {sid: i for i, sid in enumerate(sector_ids)}
    components = list(nx.connected_components(graph))

    # Precompute centroid array for vectorised distance lookup
    centroids = np.array(
        [[g.centroid.x, g.centroid.y] for g in geometries]
    )
    main_comp = max(components, key=len)
    main_idxs = np.array([idx_map[s] for s in main_comp])

    for comp in components:
        if comp is main_comp:
            continue
        comp_idxs = np.array([idx_map[s] for s in comp])

        # (n_comp × n_main) Euclidean distance matrix
        dists = np.linalg.norm(
            centroids[comp_idxs][:, None]
            - centroids[main_idxs][None],
            axis=2,
        )
        ci, mi = np.unravel_index(dists.argmin(), dists.shape)
        s1 = sector_ids[comp_idxs[ci]]
        s2 = sector_ids[main_idxs[mi]]
        graph.add_edge(s1, s2)
        logger.warning("  Bridged %s ↔ %s (nearest centroid)", s1, s2)

    return graph
