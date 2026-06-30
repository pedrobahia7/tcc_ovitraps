"""Greedy-global MST pruning for SKATER.

Algorithm (Assunção et al. 2006, adapted):
  1. Start with the full MST as one cluster (C=1).
  2. At each step, try every edge in every current subtree.
  3. Cut the edge that maximises the global objective Q:
       Q = Σ_c  (pop_c / POP_total) · q_c
     where q_c = best signed Pearson(lagged_eggs_c, dengue_c).
  4. A cut is rejected if either child would have fewer than S_min
     sectors or N_min valid (eggs, dengue) observation pairs.
  5. Repeat until C_max clusters or no valid cut remains.

One Snapshot is stored per step, capturing the full assignment map
and per-cluster diagnostics for later export and visualisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .config import SkaterConfig
from .metrics import PRUNE_OBJ

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────


@dataclass
class ClusterInfo:
    """All diagnostic information about a single cluster at one cut step.

    Attributes:
        sectors:       Frozen set of sector IDs belonging to this cluster.
        q_c:           Cluster's contribution score.  Meaning depends on
                       cfg.prune_obj:
                         'corr_q'   → best signed Pearson r (lagged eggs→dengue)
                         'egg_ssd'  → negative within-cluster egg SSD
                         'case_ssd' → negative within-cluster dengue SSD
        best_k:        The lag (biweeks) that produced q_c for correlation
                       objectives.  Always 0 for SSD objectives.
        pop:           Total population of the cluster (sum of sector medians).
        n_valid_pairs: Data-density count used to enforce the N_min guard.
                       For correlation: number of valid (eggs[t-k], dengue[t]) pairs.
                       For SSD: number of non-NaN entries in the relevant submatrix.
    """

    sectors: frozenset[str]
    q_c: float
    best_k: int
    pop: float
    n_valid_pairs: int


@dataclass
class Snapshot:
    """State of the full partition at a particular number of clusters C.

    One Snapshot is created after each successful cut and saved to CSV.

    Attributes:
        C:           Number of clusters at this step.
        Q:           Global objective value Q = Σ_c (pop_c/POP) · q_c.
        clusters:    Ordered list of ClusterInfo objects (one per cluster).
        assignments: Dict mapping each sector_id → cluster index (0-based).
    """

    C: int
    Q: float
    clusters: list[ClusterInfo] = field(default_factory=list)
    assignments: dict[str, int] = field(default_factory=dict)


# ── Internal helpers ──────────────────────────────────────────────────


def _split_tree(
    tree: nx.Graph, u: str, v: str
) -> tuple[frozenset[str], frozenset[str]]:
    """Partition a tree into two components by removing edge (u, v).

    Uses an iterative BFS from u, skipping the (u,v) edge, to collect
    one component; the other is everything else in the tree.

    Args:
        tree: A connected networkx tree (no cycles).
        u, v: The edge to conceptually remove.

    Returns:
        (comp_u, comp_v): Two disjoint frozensets of node IDs.
    """
    visited: set[str] = {u}
    stack = [u]
    while stack:
        node = stack.pop()
        for nbr in tree.neighbors(node):
            if nbr not in visited and not (
                (node == u and nbr == v) or (node == v and nbr == u)
            ):
                visited.add(nbr)
                stack.append(nbr)
    comp_u = frozenset(visited)
    comp_v = frozenset(tree.nodes()) - comp_u
    # TODO assert v in comp_v
    # TODO assert comp_u.isdisjoint(comp_v)
    # TODO assert comp_v is fully connected

    return comp_u, comp_v


def _aggregate(
    sector_set: frozenset[str],
    sector_idx: dict[str, int],
    eggs_epic: np.ndarray,
    dengue_epic: np.ndarray,
    pop_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Aggregate per-sector matrices into cluster time series and submatrices.

    Eggs: simple nanmean across sectors (each ovitrap has equal weight).
    Dengue: population-weighted mean to avoid small sectors dominating
            the rate signal.

    Also returns the raw sector submatrices so SSD-based pruning objectives
    can compute within-cluster variance directly.

    Args:
        sector_set:   Sector IDs to aggregate.
        sector_idx:   Mapping sector_id → row index in the data matrices.
        eggs_epic:    (n_sectors, n_epic_biweeks) IDW egg values.
        dengue_epic:  (n_sectors, n_epic_biweeks) EB dengue rates per 1000.
        pop_vector:   (n_sectors,) median population per sector.

    Returns:
        (eggs_agg, dengue_agg, pop_total, eggs_sub, dengue_sub):
          eggs_agg   — (n_epic_biweeks,) cluster-mean egg counts.
          dengue_agg — (n_epic_biweeks,) pop-weighted dengue rate.
          pop_total  — sum of all sector populations in this cluster.
          eggs_sub   — (n_sub, n_epic_biweeks) raw sector egg values.
          dengue_sub — (n_sub, n_epic_biweeks) raw sector dengue values.
    """
    idxs = np.array([sector_idx[s] for s in sector_set], dtype=int)
    eggs_sub = eggs_epic[idxs, :]    # (n_sub, n_epic_bw)
    dengue_sub = dengue_epic[idxs, :]  # (n_sub, n_epic_bw)
    pop_sub = pop_vector[idxs]       # (n_sub,)

    with np.errstate(all="ignore"):
        eggs_agg = np.nanmean(eggs_sub, axis=0)  # (n_epic_bw,)

    pop_total = float(pop_sub.sum())
    if pop_total > 0:
        # Weight each sector's dengue rate by its population share
        dengue_agg = (dengue_sub * pop_sub[:, None]).sum(
            axis=0
        ) / pop_total
    else:
        # Degenerate case: zero population, fall back to simple mean
        with np.errstate(all="ignore"):
            dengue_agg = np.nanmean(dengue_sub, axis=0)

    return eggs_agg, dengue_agg, pop_total, eggs_sub, dengue_sub


def _cluster_info(
    sector_set: frozenset[str],
    sector_idx: dict[str, int],
    eggs_epic: np.ndarray,
    dengue_epic: np.ndarray,
    pop_vector: np.ndarray,
    year_ids: np.ndarray,
    cfg: SkaterConfig,
) -> ClusterInfo:
    """Compute and return the full ClusterInfo for a candidate sector set.

    Delegates to the prune_fn selected by cfg.prune_obj.  The function
    returns n_valid_pairs alongside q_c — validity counting is now the
    responsibility of each pruning objective (correlation objectives count
    lag pairs; SSD objectives count non-NaN entries in the submatrix).

    If n_valid_pairs < N_min, q_c is zeroed so the cluster does not
    contribute positively to Q (the cut guard in greedy_prune also checks
    this and will reject the cut entirely).
    """
    prune_fn = PRUNE_OBJ[cfg.prune_obj]

    # ── Aggregate sector-level data into cluster series + submatrices ──
    eggs_agg, dengue_agg, pop, eggs_sub, dengue_sub = _aggregate(
        sector_set, sector_idx, eggs_epic, dengue_epic, pop_vector
    )

    # ── Compute objective, lag, and data-density in one call ──────────
    q_c, best_k, n_valid = prune_fn(
        eggs_agg, dengue_agg,
        eggs_sub, dengue_sub,
        year_ids, cfg.k_min, cfg.k_max, cfg.N_min,
    )

    return ClusterInfo(
        sectors=sector_set,
        # Zero q_c for data-sparse clusters so they don't reward cuts
        q_c=0.0 if n_valid < cfg.N_min else q_c,
        best_k=best_k,
        pop=pop,
        n_valid_pairs=n_valid,
    )


def _assignments(clusters: list[ClusterInfo]) -> dict[str, int]:
    """Build a flat sector_id → cluster_index mapping from a cluster list."""
    return {s: cid for cid, ci in enumerate(clusters) for s in ci.sectors}


# ── Main pruning loop ─────────────────────────────────────────────────


def greedy_prune(
    mst: nx.Graph,
    sector_idx: dict[str, int],
    eggs_epic: np.ndarray,
    dengue_epic: np.ndarray,
    pop_vector: np.ndarray,
    year_ids: np.ndarray,
    cfg: SkaterConfig,
) -> list[Snapshot]:
    """Run greedy global SKATER pruning from C=1 to C=cfg.C_max.

    At each step, every edge in every current subtree is a candidate cut.
    The edge producing the largest ΔQ = new_contrib − old_contrib is
    selected.  Ties are broken deterministically by (tree_idx, u, v).

    Guard conditions reject any cut where either child would have:
      - fewer than cfg.S_min sectors, OR
      - fewer than cfg.N_min valid (eggs, dengue) observation pairs.

    Args:
        mst:         Full minimum spanning tree (networkx Graph).
        sector_idx:  Mapping sector_id → row index in the data matrices.
        eggs_epic:   (n_sectors, n_epic_biweeks) IDW egg values.
        dengue_epic: (n_sectors, n_epic_biweeks) EB dengue rate per 1000.
        pop_vector:  (n_sectors,) median population per sector.
        year_ids:    (n_epic_biweeks,) epidemic-year label per biweek.
        cfg:         Pipeline configuration.

    Returns:
        List of Snapshot objects, one per C value from 1 to C_max
        (or until no valid cut remains).
    """
    POP_TOTAL = float(pop_vector.sum())

    # ── Initialise with the full MST as a single cluster ──────────────
    all_sectors = frozenset(mst.nodes())
    init_ci = _cluster_info(
        all_sectors,
        sector_idx,
        eggs_epic,
        dengue_epic,
        pop_vector,
        year_ids,
        cfg,
    )
    # One tree per current cluster; starts as just the full MST
    trees: list[nx.Graph] = [mst.copy()]
    cluster_infos: list[ClusterInfo] = [init_ci]
    Q_cur = (init_ci.pop / POP_TOTAL) * init_ci.q_c

    snapshots: list[Snapshot] = [
        Snapshot(
            C=1,
            Q=Q_cur,
            clusters=[init_ci],
            assignments=_assignments([init_ci]),
        )
    ]
    # ── Greedy cut loop: add one cluster per iteration ────────────────
    for step in range(cfg.C_max - 1):
        best_dQ = float("-inf")
        best_key: tuple[int, str, str] | None = None
        best_cis: tuple[ClusterInfo, ClusterInfo] | None = None

        # Search every edge in every current subtree
        for t_idx, (tree, t_ci) in enumerate(zip(trees, cluster_infos)):
            q_t, pop_t = t_ci.q_c, t_ci.pop
            # What Q this tree contributes before the cut
            old_contrib = (pop_t / POP_TOTAL) * q_t

            # Sort edges for deterministic tie-breaking
            for u, v in sorted(tree.edges()):
                set_a, set_b = _split_tree(tree, u, v)

                # Guard: too few sectors in either child
                if len(set_a) < cfg.S_min or len(set_b) < cfg.S_min:
                    continue

                ci_a = _cluster_info(
                    set_a,
                    sector_idx,
                    eggs_epic,
                    dengue_epic,
                    pop_vector,
                    year_ids,
                    cfg,
                )
                # Guard: too few valid observation pairs in child A
                if ci_a.n_valid_pairs < cfg.N_min:
                    continue

                ci_b = _cluster_info(
                    set_b,
                    sector_idx,
                    eggs_epic,
                    dengue_epic,
                    pop_vector,
                    year_ids,
                    cfg,
                )
                # Guard: too few valid observation pairs in child B
                if ci_b.n_valid_pairs < cfg.N_min:
                    continue

                # How much Q would change if we make this cut
                new_contrib = (ci_a.pop / POP_TOTAL) * ci_a.q_c + (
                    ci_b.pop / POP_TOTAL
                ) * ci_b.q_c
                dQ = new_contrib - old_contrib

                # Track best cut; secondary sort on (t_idx, u, v) for ties
                key = (t_idx, min(u, v), max(u, v))
                if dQ > best_dQ or (
                    dQ == best_dQ
                    and best_key is not None
                    and key < best_key
                ):
                    best_dQ = dQ
                    best_key = key
                    best_cis = (ci_a, ci_b)

        # No valid cut found — stop early
        if best_key is None:
            logger.warning(
                "No valid cut at C=%d — stopping at C=%d",
                step + 2,
                step + 1,
            )
            break

        # ── Apply the best cut ─────────────────────────────────────────
        t_idx, ua, ub = best_key
        ci_a, ci_b = best_cis  # type: ignore[misc]
        old_tree = trees[t_idx]

        # Replace the cut tree with its two children
        sub_a = old_tree.subgraph(ci_a.sectors).copy()
        sub_b = old_tree.subgraph(ci_b.sectors).copy()
        trees[t_idx] = sub_a
        cluster_infos[t_idx] = ci_a
        trees.append(sub_b)
        cluster_infos.append(ci_b)

        # Recompute global Q from scratch (correlations don't decompose)
        Q_cur = sum((ci.pop / POP_TOTAL) * ci.q_c for ci in cluster_infos)
        # TODO make sure the correlations are calculated again
        logger.info(
            "Cut %d: tree %d edge (%s,%s) → Q=%.4f (ΔQ=%.4f, C=%d)",
            step + 1,
            t_idx,
            ua,
            ub,
            Q_cur,
            best_dQ,
            step + 2,
        )
        snapshots.append(
            Snapshot(
                C=step + 2,
                Q=Q_cur,
                clusters=list(cluster_infos),
                assignments=_assignments(cluster_infos),
            )
        )

    return snapshots
