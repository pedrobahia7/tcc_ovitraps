"""Orchestrate the full SKATER pipeline and persist results.

Entry point: `python -m src.skater.run`

Execution order:
  1. Load config from params.yaml[skater:]
  2. Load and align data matrices (eggs, dengue, geojson)
  3. Build Queen contiguity adjacency graph
  4. Build weighted MST (Prim, cost = egg_corr_dist)
  5. Run greedy pruning → list of Snapshots (C=1..C_max)
  6. Save CSVs and graph edge lists to results/skater/

Output files:
  cluster_assignments.csv  — sector → cluster_id for every C value
  q_trajectory.csv         — Q score at each C
  cluster_diagnostics.csv  — per-cluster stats (q_c, best_k, pop, …)
  adjacency_edges.csv      — all Queen contiguity edges (src, dst)
  mst_edges.csv            — MST edges with dissimilarity weights
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from .adjacency import build_adjacency
from .config import load_config
from .data import SkaterData, load_data
from .mst import build_mst
from .prune import Snapshot, StopInfo, greedy_prune

logger = logging.getLogger(__name__)
RESULTS_BASE = Path("results/skater")


# ── Reproducibility ───────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    """Fix all random seeds for reproducible MST and pruning results."""
    random.seed(seed)
    np.random.seed(seed)


# ── CSV serialisers ───────────────────────────────────────────────────

def _save_assignments(snapshots: list[Snapshot], out: Path) -> None:
    """Write cluster assignments for every C value to CSV.

    Columns: C (number of clusters), sector_id, cluster_id (0-based).
    One row per sector per C value → shape (n_sectors × C_max, 3).
    """
    records = [
        {"C": snap.C, "sector_id": sid, "cluster_id": cid}
        for snap in snapshots
        for sid, cid in snap.assignments.items()
    ]
    pd.DataFrame(records).to_csv(out / "cluster_assignments.csv", index=False)
    logger.info("Saved cluster_assignments.csv")


def _save_trajectory(snapshots: list[Snapshot], out: Path) -> None:
    """Write the Q-vs-C trajectory to CSV.

    Columns: C, Q.  Used by the analysis dashboard to plot the
    objective function as it evolves with each additional cluster.
    """
    pd.DataFrame(
        [{"C": s.C, "Q": s.Q} for s in snapshots]
    ).to_csv(out / "q_trajectory.csv", index=False)
    logger.info("Saved q_trajectory.csv")


def _save_diagnostics(snapshots: list[Snapshot], out: Path) -> None:
    """Write per-cluster diagnostics for every C value to CSV.

    Columns: C, cluster_id, q_c, best_k, n_sectors, pop, n_valid_pairs.
    Used by the analysis dashboard to annotate time-series plots.
    """
    records = [
        {
            "C": snap.C,
            "cluster_id": cid,
            "q_c": ci.q_c,
            "best_k": ci.best_k,
            "n_sectors": len(ci.sectors),
            "pop": ci.pop,
            "n_valid_pairs": ci.n_valid_pairs,
        }
        for snap in snapshots
        for cid, ci in enumerate(snap.clusters)
    ]
    pd.DataFrame(records).to_csv(
        out / "cluster_diagnostics.csv", index=False
    )
    logger.info("Saved cluster_diagnostics.csv")


def _save_graph_structures(
    adjacency: nx.Graph, mst: nx.Graph, out: Path
) -> None:
    """Save the adjacency graph and MST edge lists to CSV.

    adjacency_edges.csv — columns: src, dst.
      All Queen contiguity edges (undirected, one row per edge).

    mst_edges.csv — columns: src, dst, weight.
      MST edges with their egg_corr_dist weights (0=similar, 2=max diff).
      Used by the structure dashboard to visualise the spatial skeleton.
    """
    adj_records = [{"src": u, "dst": v} for u, v in adjacency.edges()]
    pd.DataFrame(adj_records).to_csv(
        out / "adjacency_edges.csv", index=False
    )

    mst_records = [
        {"src": u, "dst": v, "weight": d["weight"]}
        for u, v, d in mst.edges(data=True)
    ]
    pd.DataFrame(mst_records).to_csv(
        out / "mst_edges.csv", index=False
    )
    logger.info(
        "Saved adjacency_edges.csv (%d) and mst_edges.csv (%d)",
        len(adj_records),
        len(mst_records),
    )


# ── Reusable pipeline ─────────────────────────────────────────────────

def _default_mst_matrix(cfg: SkaterConfig, data: SkaterData) -> np.ndarray:
    """Select the MST cost matrix exactly as the original pipeline does.

    egg_corr_dist (and every egg/spearman/euclidean cost) uses all
    biweeks; only case_corr_dist falls back to the epidemic dengue
    matrix.  Preserved verbatim so callers that pass no override get
    identical behaviour.
    """
    if cfg.mst_cost == "case_corr_dist":
        return data.dengue_epic
    return data.eggs_all


def run_pipeline(
    cfg: SkaterConfig,
    data: SkaterData,
    adjacency: nx.Graph,
    out_dir: Path,
    mst_matrix: np.ndarray | None = None,
) -> tuple[list[Snapshot], StopInfo]:
    """Build the MST, prune greedily and persist the full output set.

    This is the single source of truth for the SKATER algorithm run.
    Both the all-years `main()` and the cross-validated per-fold runs
    call it, so the CV folds behave exactly like the original algorithm.

    Args:
        cfg:        Resolved SKATER configuration for this run.
        data:       Aligned data matrices (already restricted to whatever
                    epidemic years cfg specifies).
        adjacency:  Queen contiguity graph (geometry only, reusable).
        out_dir:    Directory to write all CSV/JSON outputs into.
        mst_matrix: Optional MST cost matrix override.  None → the exact
                    matrix the original pipeline would select.  CV passes
                    a copy with the held-out year's columns removed.

    Returns:
        (snapshots, stop_info) from greedy_prune.
    """
    _set_seeds(cfg.seed)
    if mst_matrix is None:
        mst_matrix = _default_mst_matrix(cfg, data)

    mst = build_mst(adjacency, mst_matrix, data.sector_list, cfg)
    sector_idx = {s: i for i, s in enumerate(data.sector_list)}

    snapshots, stop_info = greedy_prune(
        mst=mst,
        sector_idx=sector_idx,
        eggs_epic=data.eggs_epic,
        dengue_epic=data.dengue_epic,
        pop_vector=data.pop_vector,
        year_ids=data.year_ids,
        cfg=cfg,
    )

    # ── Persist all outputs ───────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "run_params.json", "w") as fh:
        json.dump(cfg.model_dump(), fh, indent=2)

    with open(out_dir / "stop_info.json", "w") as fh:
        json.dump(
            {
                "reason": stop_info.reason,
                "C_final": stop_info.C_final,
                "best_dQ": stop_info.best_dQ,
                "threshold": stop_info.threshold,
            },
            fh,
            indent=2,
        )
    logger.info(
        "Saved stop_info.json (reason=%s, C_final=%d) → %s",
        stop_info.reason, stop_info.C_final, out_dir,
    )

    _save_assignments(snapshots, out_dir)
    _save_trajectory(snapshots, out_dir)
    _save_diagnostics(snapshots, out_dir)
    _save_graph_structures(adjacency, mst, out_dir)
    logger.info("SKATER done — %d snapshots in %s", len(snapshots), out_dir)
    return snapshots, stop_info


# ── Pipeline entry point ──────────────────────────────────────────────

def main() -> None:
    """Run the full all-years SKATER pipeline end-to-end."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    cfg = load_config()
    logger.info("SkaterConfig: %s", cfg.model_dump())

    data = load_data(cfg)
    adjacency = build_adjacency(
        data.geojson, sector_filter=set(data.sector_list)
    )
    run_pipeline(cfg, data, adjacency, RESULTS_BASE / cfg.run_label)


if __name__ == "__main__":
    main()
