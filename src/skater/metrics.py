"""MST cost and pruning objective functions for SKATER.

Two registries — MST_COST and PRUNE_OBJ — act as strategy maps:
new metrics can be added here (entries P2-P4) without touching any
other module.  The registry key is what goes in params.yaml.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

# ── Type aliases ──────────────────────────────────────────────────────
# These make function signatures self-documenting when used as values
# in the registry dicts below.
MstCostFn = Callable[[np.ndarray, np.ndarray, int], float]
PruneObjFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int, int, int],
    tuple[float, int],
]


# ── MST cost function ─────────────────────────────────────────────────

def egg_corr_dist(
    eggs_i: np.ndarray,
    eggs_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Dissimilarity between two sector egg time series.

    Defined as  cost = 1 − Pearson(eggs_i, eggs_j)  computed only on
    biweeks where *both* series have a non-NaN value.

    Range: [0, 2]
      - 0.0 → perfectly correlated (identical dynamics)
      - 1.0 → no correlation
      - 2.0 → perfectly anti-correlated  OR  fallback (see below)

    Returns 2.0 (max cost) in any degenerate case:
      - fewer than `min_overlap` shared non-NaN biweeks
      - either series has zero variance (constant value)
      - Pearson formula returns NaN for any other reason

    Args:
        eggs_i: IDW egg counts for sector i, shape (n_biweeks,).
        eggs_j: IDW egg counts for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Float in [0, 2].
    """
    # Keep only biweeks where both sectors reported eggs
    mask = ~(np.isnan(eggs_i) | np.isnan(eggs_j))
    n = int(mask.sum())
    if n < min_overlap:
        return 2.0

    xi, xj = eggs_i[mask], eggs_j[mask]

    # A constant series has zero variance — correlation is undefined
    if xi.std() < 1e-12 or xj.std() < 1e-12:
        return 2.0

    r = float(np.corrcoef(xi, xj)[0, 1])
    return 2.0 if np.isnan(r) else float(1.0 - r)


# ── Pruning objective function ────────────────────────────────────────

def best_lag_corr(
    eggs: np.ndarray,
    dengue: np.ndarray,
    year_ids: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    n_min: int = 20,
) -> tuple[float, int]:
    """Best signed Pearson r between lagged cluster eggs and dengue rate.

    For each candidate lag k in [k_min, k_max], computes:
        r_k = Pearson( eggs[t−k],  dengue[t] )
    across all epidemic biweeks, then returns the (r, k) pair with the
    highest r.

    **Year-boundary protection**: lag is applied *within* each epidemic
    year.  e.g. for year 2012_13 with biweeks [b0..b25], lag-3 produces
    pairs (b0,b3), (b1,b4), …, (b22,b25).  The last 3 biweeks of one
    year are never paired with the first 3 of the next, which would mix
    inter-season noise into the signal.

    Args:
        eggs:     Cluster-aggregated IDW eggs, shape (n_epic_biweeks,).
        dengue:   Cluster pop-weighted EB dengue rate, shape (n_epic_biweeks,).
        year_ids: String array of epidemic-year labels per biweek,
                  shape (n_epic_biweeks,).  e.g. ['2012_13', '2012_13', …]
        k_min:    Smallest lag to test (biweeks).
        k_max:    Largest lag to test (biweeks).
        n_min:    Minimum valid (eggs, dengue) pairs required across all
                  years before a lag is accepted.

    Returns:
        (best_r, best_k): best signed Pearson r and the lag that achieved it.
        Returns (0.0, k_min) when no lag has enough valid data.
    """
    best_r = float("-inf")
    best_k = k_min
    unique_years = np.unique(year_ids)

    for k in range(k_min, k_max + 1):
        # Collect (eggs[t-k], dengue[t]) pairs within each epidemic year
        lagged_e: list[float] = []
        target_d: list[float] = []

        for yr in unique_years:
            mask = year_ids == yr
            ye = eggs[mask]
            yd = dengue[mask]
            n = len(ye)

            # Skip years too short to form any lag-k pair
            if n <= k:
                continue

            # Shift eggs back by k steps; align with future dengue
            lagged_e.extend(ye[:-k].tolist())
            target_d.extend(yd[k:].tolist())

        # Require minimum pool size before computing correlation
        if len(lagged_e) < n_min:
            continue

        ae = np.asarray(lagged_e, dtype=float)
        ad = np.asarray(target_d, dtype=float)

        # Drop biweeks where either value is missing
        valid = ~(np.isnan(ae) | np.isnan(ad))
        if int(valid.sum()) < n_min:
            continue

        # Skip degenerate (constant) series
        if ae[valid].std() < 1e-12 or ad[valid].std() < 1e-12:
            continue

        r = float(np.corrcoef(ae[valid], ad[valid])[0, 1])
        if np.isnan(r):
            continue

        if r > best_r:
            best_r = r
            best_k = k

    # No valid lag found → neutral score
    if best_r == float("-inf"):
        return 0.0, k_min

    return float(best_r), best_k


# ── Strategy registries ───────────────────────────────────────────────
# To add a new metric: define the function above, then add it here.
# The key must match the value used in params.yaml.

MST_COST: dict[str, MstCostFn] = {
    "egg_corr_dist": egg_corr_dist,
    # P3 (future): "cross_zone_pred": cross_zone_pred,
}

PRUNE_OBJ: dict[str, PruneObjFn] = {
    "eggs_dengue_corr": best_lag_corr,
    # P2 (future): "pretrained_mlp": pretrained_mlp_score,
    # P4 (future): "trained_mlp":    trained_mlp_score,
}
