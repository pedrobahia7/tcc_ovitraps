"""Build the weighted Minimum Spanning Tree (MST) for SKATER.

The MST defines the spatial backbone that SKATER prunes.  Edge weights
represent *dissimilarity* between adjacent sectors — lower weight means
more similar egg dynamics, so Prim's algorithm will prefer those edges,
keeping correlated neighbours together.

Cost function is selected via cfg.mst_cost and looked up in the
MST_COST registry (metrics.py), making it swappable without editing
this module.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np

from .config import SkaterConfig
from .metrics import MST_COST

logger = logging.getLogger(__name__)


def build_mst(
    adjacency: nx.Graph,
    data_matrix: np.ndarray,
    sector_list: list[str],
    cfg: SkaterConfig,
) -> nx.Graph:
    """Build a weighted MST over the adjacency graph using cfg.mst_cost.

    Steps:
      1. Remove any adjacency nodes not present in sector_list (sectors
         that were excluded from data loading due to missing eggs/dengue).
      2. Assign an edge weight = cfg.mst_cost(row_i, row_j) to every
         pair of adjacent sectors.
      3. Run Prim's algorithm to extract the MST.

    The returned MST carries 'weight' edge attributes so the pruning step
    can inspect edge costs if needed.

    Args:
        adjacency:   Queen contiguity graph from build_adjacency().
        data_matrix: (n_sectors, n_biweeks) matrix whose rows feed the MST
                     cost function.  Caller selects the appropriate matrix:
                       'egg_corr_dist'  → pass data.eggs_all (all biweeks)
                       'case_corr_dist' → pass data.dengue_epic (epidemic only)
        sector_list: Ordered list of sector IDs.  Determines row indices
                     into data_matrix.
        cfg:         Pipeline config; cfg.mst_cost selects the cost function.

    Returns:
        A networkx Graph — the MST — with 'weight' on every edge.
    """
    cost_fn = MST_COST[cfg.mst_cost]

    # Build a fast index: sector_id → row in data_matrix
    sector_idx = {s: i for i, s in enumerate(sector_list)}
    sector_set = set(sector_list)

    # ── Remove adjacency nodes absent from our data matrices ──────────
    extra = set(adjacency.nodes()) - sector_set
    if extra:
        logger.warning(
            "Dropping %d adjacency nodes absent from sector data", len(extra)
        )
        adjacency = adjacency.copy()
        adjacency.remove_nodes_from(extra)

    logger.info(
        "Computing MST edge costs (%s) for %d edges…",
        cfg.mst_cost,
        adjacency.number_of_edges(),
    )

    # ── Assign a dissimilarity weight to every adjacency edge ─────────
    weighted: nx.Graph = nx.Graph()
    weighted.add_nodes_from(adjacency.nodes())

    for u, v in adjacency.edges():
        i, j = sector_idx.get(u), sector_idx.get(v)
        if i is None or j is None:
            # Defensive fallback — should not happen after the drop above
            w = 2.0
        else:
            w = cost_fn(data_matrix[i], data_matrix[j], cfg.mst_min_overlap)
        weighted.add_edge(u, v, weight=w)

    # ── Run Prim's MST on the weighted graph ──────────────────────────
    logger.info("Running Prim's MST…")
    mst = nx.minimum_spanning_tree(weighted, algorithm="prim")

    logger.info(
        "MST: %d nodes, %d edges (max_weight=%.4f)",
        mst.number_of_nodes(),
        mst.number_of_edges(),
        max(d["weight"] for _, _, d in mst.edges(data=True)),
    )
    return mst
