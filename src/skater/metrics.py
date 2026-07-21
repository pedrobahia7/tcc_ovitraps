"""MST cost and pruning objective functions for SKATER.

Two registries — MST_COST and PRUNE_OBJ — act as strategy maps:
new metrics can be added here without touching any other module.
The registry key matches the value set in params.yaml.

MST_COST keys:
  'egg_corr_dist'   — 1 − Pearson(eggs_i, eggs_j), all biweeks
  'case_corr_dist'  — 1 − Pearson(dengue_i, dengue_j), epidemic biweeks, lag=0
  'egg_euclidean'   — L2‖eggs_i − eggs_j‖, all biweeks (Assunção 2006 paper-style)
  'case_euclidean'  — L2‖cases_i − cases_j‖, epidemic biweeks (paper-style)
  'egg_spearman'    — 1 − Spearman(eggs_i, eggs_j), all biweeks
  'case_spearman'   — 1 − Spearman(dengue_i, dengue_j), epidemic biweeks

PRUNE_OBJ keys:
  'corr_q'          — best signed Pearson(lagged_eggs, dengue_rate) (original)
  'corr_q_spearman' — best signed Spearman(lagged_eggs, dengue_rate)
  'egg_ssd'         — negative multivariate SSD of egg time series (paper-style)
  'case_ssd'        — negative multivariate SSD of dengue rate time series (paper-style)
  'eggs_dengue_corr' — backwards-compat alias for 'corr_q'
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Literal

import numpy as np
from scipy.stats import rankdata

# ── Type aliases ──────────────────────────────────────────────────────
# MstCostFn:  (series_i, series_j, min_overlap) → dissimilarity ∈ [0, 2]
MstCostFn = Callable[[np.ndarray, np.ndarray, int], float]

# PruneObjFn: (eggs_agg, dengue_agg, eggs_sub, dengue_sub,
#              year_ids, k_min, k_max, n_min)
#             → (q_c, best_k, n_valid_pairs)
# eggs_sub / dengue_sub are raw (n_sectors × n_biweeks) cluster submatrices.
# Correlation objectives use the aggregated series and ignore *_sub.
# SSD objectives use *_sub directly and ignore the aggregated series.
PruneObjFn = Callable[
    [
        np.ndarray,  # eggs_agg:   (n_epic_bw,) cluster-mean egg series
        np.ndarray,  # dengue_agg: (n_epic_bw,) pop-weighted dengue series
        np.ndarray,  # eggs_sub:   (n_sectors, n_epic_bw) raw egg matrix
        np.ndarray,  # dengue_sub: (n_sectors, n_epic_bw) raw dengue matrix
        np.ndarray,  # year_ids:   (n_epic_bw,) epidemic-year label
        int,         # k_min
        int,         # k_max
        int,         # n_min
    ],
    tuple[float, int, int],  # (q_c, best_k, n_valid_pairs)
]


# ── Helpers ───────────────────────────────────────────────────────────

def _fast_spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman r via rankdata + corrcoef — avoids scipy p-value overhead.

    Equivalent to scipy.stats.spearmanr(x, y).statistic but ~10x faster
    for small arrays because it skips p-value computation and per-call
    input validation.

    Args:
        x: First array.
        y: Second array, same shape as x.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


# ── MST cost functions ────────────────────────────────────────────────

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
        eggs_i:      IDW egg counts for sector i, shape (n_biweeks,).
        eggs_j:      IDW egg counts for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Float in [0, 2].
    """
    # ── Keep only biweeks where both sectors reported eggs ────────────
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


def case_corr_dist(
    cases_i: np.ndarray,
    cases_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Dissimilarity between two sector dengue-rate time series.

    Identical formula to egg_corr_dist:
        cost = 1 − Pearson(cases_i, cases_j)
    Applied to EB dengue rate per 1000, epidemic biweeks only, lag=0.

    Range: [0, 2].  Returns 2.0 in all degenerate cases.

    Args:
        cases_i:     EB dengue rate per 1000 for sector i, shape (n_biweeks,).
        cases_j:     EB dengue rate per 1000 for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Float in [0, 2].
    """
    # ── Keep only biweeks where both sectors have dengue data ─────────
    mask = ~(np.isnan(cases_i) | np.isnan(cases_j))
    n = int(mask.sum())
    if n < min_overlap:
        return 2.0

    xi, xj = cases_i[mask], cases_j[mask]

    if xi.std() < 1e-12 or xj.std() < 1e-12:
        return 2.0

    r = float(np.corrcoef(xi, xj)[0, 1])
    return 2.0 if np.isnan(r) else float(1.0 - r)


def egg_spearman(
    eggs_i: np.ndarray,
    eggs_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Rank-based dissimilarity between two sector egg time series.

    Computes  cost = 1 − Spearman(eggs_i, eggs_j)  on biweeks where
    both series are non-NaN.  Spearman is monotone-invariant and robust
    to outlier traps, treating the data as ordinal after ranking.
    Tied ranks handled by scipy.stats.rankdata (average method).

    Range: [0, 2].  Returns 2.0 in degenerate cases:
      - fewer than `min_overlap` shared non-NaN biweeks
      - constant series (zero rank variance)
      - _fast_spearmanr returns NaN for any other reason

    Args:
        eggs_i:      IDW egg counts for sector i, shape (n_biweeks,).
        eggs_j:      IDW egg counts for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Float in [0, 2].
    """
    mask = ~(np.isnan(eggs_i) | np.isnan(eggs_j))
    if int(mask.sum()) < min_overlap:
        return 2.0
    xi, xj = eggs_i[mask], eggs_j[mask]
    if xi.std() < 1e-12 or xj.std() < 1e-12:
        return 2.0
    r = _fast_spearmanr(xi, xj)
    return 2.0 if np.isnan(r) else float(1.0 - r)


def case_spearman(
    cases_i: np.ndarray,
    cases_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Rank-based dissimilarity between two sector dengue-rate time series.

    Identical formula to egg_spearman:
        cost = 1 − Spearman(cases_i, cases_j)
    Applied to EB dengue rate per 1000, epidemic biweeks only, lag=0.

    Range: [0, 2].  Returns 2.0 in all degenerate cases.

    Args:
        cases_i:     EB dengue rate per 1000 for sector i, shape (n_biweeks,).
        cases_j:     EB dengue rate per 1000 for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Float in [0, 2].
    """
    mask = ~(np.isnan(cases_i) | np.isnan(cases_j))
    if int(mask.sum()) < min_overlap:
        return 2.0
    xi, xj = cases_i[mask], cases_j[mask]
    if xi.std() < 1e-12 or xj.std() < 1e-12:
        return 2.0
    r = _fast_spearmanr(xi, xj)
    return 2.0 if np.isnan(r) else float(1.0 - r)


def egg_euclidean(
    eggs_i: np.ndarray,
    eggs_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Euclidean distance between two sector egg time series.

    Computes  d(i,j) = ‖eggs_i − eggs_j‖₂  over biweeks where both
    series are non-NaN.  This is the dissimilarity used by Assunção et
    al. 2006: raw L2 distance in attribute space, scale-sensitive.

    Range: [0, ∞).  Returns inf when fewer than `min_overlap` shared
    biweeks are available (Prim treats the edge as disconnected).

    Args:
        eggs_i:      IDW egg counts for sector i, shape (n_biweeks,).
        eggs_j:      IDW egg counts for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Non-negative float; inf when insufficient data.
    """
    mask = ~(np.isnan(eggs_i) | np.isnan(eggs_j))
    if int(mask.sum()) < min_overlap:
        return float("inf")
    diff = eggs_i[mask] - eggs_j[mask]
    return float(np.sqrt(np.dot(diff, diff)))


def case_euclidean(
    cases_i: np.ndarray,
    cases_j: np.ndarray,
    min_overlap: int = 10,
) -> float:
    """Euclidean distance between two sector dengue-rate time series.

    Same formula as egg_euclidean applied to EB dengue rate per 1000,
    epidemic biweeks only.  Paper-faithful MST cost for a cases-only
    SKATER run (Assunção et al. 2006).

    Range: [0, ∞).  Returns inf when fewer than `min_overlap` shared
    biweeks are available.

    Args:
        cases_i:     EB dengue rate per 1000 for sector i, shape (n_biweeks,).
        cases_j:     EB dengue rate per 1000 for sector j, shape (n_biweeks,).
        min_overlap: Minimum shared non-NaN observations required.

    Returns:
        Non-negative float; inf when insufficient data.
    """
    mask = ~(np.isnan(cases_i) | np.isnan(cases_j))
    if int(mask.sum()) < min_overlap:
        return float("inf")
    diff = cases_i[mask] - cases_j[mask]
    return float(np.sqrt(np.dot(diff, diff)))


# ── Pruning objective functions ───────────────────────────────────────

def best_lag_corr(
    eggs_agg: np.ndarray,
    dengue_agg: np.ndarray,
    eggs_sub: np.ndarray,
    dengue_sub: np.ndarray,
    year_ids: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    n_min: int = 20,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> tuple[float, int, int]:
    """Best signed correlation between lagged cluster eggs and dengue rate.

    For each candidate lag k in [k_min, k_max], computes:
        r_k = corr( eggs[t−k], dengue[t] )
    across all epidemic biweeks using `method` (Pearson or Spearman),
    then returns the (r, k) pair with the highest r.

    **Year-boundary protection**: lag is applied *within* each epidemic
    year.  e.g. for year 2012_13 with biweeks [b0..b25], lag-3 produces
    pairs (b0,b3), (b1,b4), …, (b22,b25).  The last 3 biweeks of one
    year are never paired with the first 3 of the next, which would mix
    inter-season noise into the signal.

    `eggs_sub` and `dengue_sub` are accepted but ignored — this objective
    operates on aggregated cluster series only.

    Args:
        eggs_agg:   Cluster-aggregated IDW eggs, shape (n_epic_biweeks,).
        dengue_agg: Cluster pop-weighted EB dengue rate, shape (n_epic_biweeks,).
        eggs_sub:   Raw sector egg matrix — ignored by this objective.
        dengue_sub: Raw sector dengue matrix — ignored by this objective.
        year_ids:   String array of epidemic-year labels per biweek,
                    shape (n_epic_biweeks,).  e.g. ['2012_13', '2012_13', …]
        k_min:      Smallest lag to test (biweeks).
        k_max:      Largest lag to test (biweeks).
        n_min:      Minimum valid (eggs, dengue) pairs required across all
                    years before a lag is accepted.
        method:     Correlation method — 'pearson' (default) uses np.corrcoef;
                    'spearman' uses _fast_spearmanr (rankdata + corrcoef).

    Returns:
        (best_r, best_k, n_valid_pairs):
          best_r        — best signed correlation r.
          best_k        — the lag (biweeks) that achieved best_r.
          n_valid_pairs — max valid (eggs[t-k], dengue[t]) count across lags.
          Returns (0.0, k_min, 0) when no lag has enough valid data.
    """
    best_r = float("-inf")
    best_k = k_min
    best_count = 0
    unique_years = np.unique(year_ids)

    for k in range(k_min, k_max + 1):
        # ── Collect (eggs[t-k], dengue[t]) pairs within each year ─────
        lagged_e: list[float] = []
        target_d: list[float] = []

        for yr in unique_years:
            mask = year_ids == yr
            ye = eggs_agg[mask]
            yd = dengue_agg[mask]
            n = len(ye)

            if n <= k:
                continue

            lagged_e.extend(ye[:-k].tolist())
            target_d.extend(yd[k:].tolist())

        if len(lagged_e) < n_min:
            continue

        ae = np.asarray(lagged_e, dtype=float)
        ad = np.asarray(target_d, dtype=float)

        # Drop biweeks where either value is missing
        valid = ~(np.isnan(ae) | np.isnan(ad))
        valid_count = int(valid.sum())

        # Track maximum valid-pair count across all lags for the guard
        best_count = max(best_count, valid_count)

        if valid_count < n_min:
            continue

        if ae[valid].std() < 1e-12 or ad[valid].std() < 1e-12:
            continue

        # ── Compute correlation by chosen method ──────────────────────
        if method == "spearman":
            r = _fast_spearmanr(ae[valid], ad[valid])
        else:
            r = float(np.corrcoef(ae[valid], ad[valid])[0, 1])

        if np.isnan(r):
            continue

        if r > best_r:
            best_r = r
            best_k = k

    if best_r == float("-inf"):
        return 0.0, k_min, best_count

    return float(best_r), best_k, best_count


def egg_ssd_obj(
    eggs_agg: np.ndarray,
    dengue_agg: np.ndarray,
    eggs_sub: np.ndarray,
    dengue_sub: np.ndarray,
    year_ids: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    n_min: int = 20,
) -> tuple[float, int, int]:
    """Negative multivariate SSD over cluster egg time series (original paper).

    Computes q_c = −Σ_t Σ_s (eggs[s,t] − mean_t)²
    where mean_t = nanmean of eggs across all sectors in the cluster at biweek t.

    Sign-flipped so maximising q_c = minimising within-cluster egg variance,
    which is the regionalization criterion from Assunção et al. 2006.

    Aggregated series (eggs_agg, dengue_agg) and year/lag params are accepted
    but ignored — SSD is computed from the raw sector submatrix directly.

    Args:
        eggs_agg:   Cluster-mean eggs — ignored by this objective.
        dengue_agg: Cluster dengue — ignored by this objective.
        eggs_sub:   (n_sectors, n_epic_biweeks) raw egg matrix for cluster sectors.
        dengue_sub: Raw dengue matrix — ignored by this objective.
        year_ids:   Epidemic-year labels — ignored by this objective.
        k_min:      Lag lower bound — ignored by this objective.
        k_max:      Lag upper bound — ignored by this objective.
        n_min:      Ignored; guard for SSD relies on S_min, not N_min.

    Returns:
        (q_c, 0, n_valid):
          q_c     — negative SSD (higher = more homogeneous cluster).
          0       — dummy lag (lag concept does not apply to SSD).
          n_valid — number of non-NaN entries in eggs_sub.
    """
    # ── Compute per-biweek mean and squared deviations ────────────────
    with np.errstate(all="ignore"):
        mean_t = np.nanmean(eggs_sub, axis=0)   # (n_biweeks,)

    deviations = eggs_sub - mean_t[None, :]     # (n_sectors, n_biweeks)

    with np.errstate(all="ignore"):
        ssd = float(np.nansum(deviations ** 2))

    n_valid = int(np.sum(~np.isnan(eggs_sub)))
    return -ssd, 0, n_valid


def case_ssd_obj(
    eggs_agg: np.ndarray,
    dengue_agg: np.ndarray,
    eggs_sub: np.ndarray,
    dengue_sub: np.ndarray,
    year_ids: np.ndarray,
    k_min: int = 2,
    k_max: int = 6,
    n_min: int = 20,
) -> tuple[float, int, int]:
    """Negative multivariate SSD over cluster dengue-rate time series (paper-style).

    Computes q_c = −Σ_t Σ_s (dengue[s,t] − mean_t)²
    where mean_t = nanmean of EB dengue rate across cluster sectors at biweek t.

    This applies the same SSD criterion as egg_ssd_obj but to the dengue-rate
    signal, enabling a purely case-driven regionalization.

    Args:
        eggs_agg:   Cluster-mean eggs — ignored by this objective.
        dengue_agg: Cluster dengue — ignored by this objective.
        eggs_sub:   Raw egg matrix — ignored by this objective.
        dengue_sub: (n_sectors, n_epic_biweeks) raw dengue matrix for cluster sectors.
        year_ids:   Epidemic-year labels — ignored by this objective.
        k_min:      Ignored by this objective.
        k_max:      Ignored by this objective.
        n_min:      Ignored; guard relies on S_min.

    Returns:
        (q_c, 0, n_valid):
          q_c     — negative SSD (higher = more homogeneous cluster).
          0       — dummy lag.
          n_valid — number of non-NaN entries in dengue_sub.
    """
    # ── Compute per-biweek mean and squared deviations ────────────────
    with np.errstate(all="ignore"):
        mean_t = np.nanmean(dengue_sub, axis=0)  # (n_biweeks,)

    deviations = dengue_sub - mean_t[None, :]    # (n_sectors, n_biweeks)

    with np.errstate(all="ignore"):
        ssd = float(np.nansum(deviations ** 2))

    n_valid = int(np.sum(~np.isnan(dengue_sub)))
    return -ssd, 0, n_valid


# ── Strategy registries ───────────────────────────────────────────────
# To add a new metric: define the function above, then add it here.
# The key must match the value used in params.yaml.

MST_COST: dict[str, MstCostFn] = {
    # 1 − Pearson over all biweeks; preferred for eggs (more data)
    "egg_corr_dist": egg_corr_dist,
    # 1 − Pearson over epidemic biweeks only, lag=0; for dengue-only runs
    "case_corr_dist": case_corr_dist,
    # 1 − Spearman over all biweeks; rank-based, robust to outliers
    "egg_spearman": egg_spearman,
    # 1 − Spearman over epidemic biweeks only; rank-based for dengue-only runs
    "case_spearman": case_spearman,
    # L2 distance over all biweeks; scale-sensitive, closest to Assunção 2006
    "egg_euclidean": egg_euclidean,
    # L2 distance over epidemic biweeks; paper-faithful for cases-only runs
    "case_euclidean": case_euclidean,
    # P3 (future): "cross_zone_pred": cross_zone_pred,
}

PRUNE_OBJ: dict[str, PruneObjFn] = {
    # Best lagged Pearson(eggs[t-k], dengue[t]) — original custom objective
    "corr_q": best_lag_corr,
    # Best lagged Spearman(eggs[t-k], dengue[t]) — rank-based, outlier-robust
    "corr_q_spearman": partial(best_lag_corr, method="spearman"),
    # Paper-style: minimise within-cluster egg variance (Assunção 2006)
    "egg_ssd": egg_ssd_obj,
    # Paper-style: minimise within-cluster dengue-rate variance
    "case_ssd": case_ssd_obj,
    # Backwards-compat alias kept so old results/configs still load
    "eggs_dengue_corr": best_lag_corr,
    # P2 (future): "pretrained_mlp": pretrained_mlp_score,
    # P4 (future): "trained_mlp":    trained_mlp_score,
}
