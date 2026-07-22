"""SKATER-CV RMSE choropleth — one interactive HTML file.

dashboard_rmse_map.html
  Choropleth map of BH census sectors coloured by held-out MLP RMSE of
  the region (SKATER cluster) each sector belongs to, with ovitrap
  locations shown as fixed black dots.  A dropdown selects which
  leave-one-epidemic-year-out fold to inspect (each fold has its own
  SKATER partition — geometry differs across folds even at the same
  C); a slider then sweeps that fold's available C values.  Colour
  scale (RMSE) is fixed globally across every fold/C combination so
  shades are directly comparable.

Inputs:
  results/multiscale/partitions/fold_<year>/cluster_assignments.csv
  results/multiscale/skater_cv/metrics_skater.csv
  data/dvc/process_population_data/bh_sectors_2022_with_populations.geojson
  data/processed/ovitraps_data.csv
Outputs:
  results/multiscale/skater_cv/rmse_map.html
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── File paths ────────────────────────────────────────────────────────
PARTITIONS_BASE = Path("results/multiscale/partitions")
METRICS_PATH = Path("results/multiscale/skater_cv/metrics_skater.csv")
GEOJSON_PATH = Path(
    "data/dvc/process_population_data/"
    "bh_sectors_2022_with_populations.geojson"
)
OVITRAP_PATH = Path("data/processed/ovitraps_data.csv")
OUT_PATH = Path("results/multiscale/skater_cv/rmse_map.html")
BH_CENTER = {"lat": -19.917, "lon": -43.934}


# ── Data loaders ──────────────────────────────────────────────────────

def _load_geojson() -> dict:
    """Load the BH census sector GeoJSON FeatureCollection."""
    with open(GEOJSON_PATH) as fh:
        return json.load(fh)


def _load_ovitrap_locations() -> pd.DataFrame:
    """Load unique ovitrap deployment coordinates.

    Returns:
        DataFrame [idarmad, latitude, longitude] — one row per trap.
    """
    df = pd.read_csv(
        OVITRAP_PATH,
        usecols=["idarmad", "latitude", "longitude"],
        low_memory=False,
    )
    return df.drop_duplicates("idarmad").reset_index(drop=True)


def _load_metrics() -> pd.DataFrame:
    """Load per-(fold, C, region) test metrics with cluster_id parsed out.

    `unit` is formatted `C{C}__c{cluster_id}` by
    `src.multiscale.skater_cv_models`; cluster_id is not a separate
    column so it is parsed back out here.

    Returns:
        metrics_skater.csv rows plus an added integer `cluster_id` col.
    """
    df = pd.read_csv(METRICS_PATH)
    df["cluster_id"] = (
        df["unit"].str.split("__c").str[1].astype(int)
    )
    return df


def _load_fold_assignments(fold_year: str) -> pd.DataFrame:
    """Load one fold's sector → cluster_id assignments, all C values.

    Args:
        fold_year: e.g. '2015_16'.

    Returns:
        DataFrame [C, sector_id, cluster_id].
    """
    path = PARTITIONS_BASE / f"fold_{fold_year}" / "cluster_assignments.csv"
    return pd.read_csv(path, dtype={"sector_id": str})


# ── Per-(fold, C) frame construction ────────────────────────────────────

def _frame_for_c(
    asgn_c: pd.DataFrame,
    metrics_c: pd.DataFrame,
    sector_order: list[str],
) -> tuple[list[float], list[str]]:
    """Build the RMSE z-array and hover text for one (fold, C) frame.

    Args:
        asgn_c:       Assignments filtered to one C [sector_id, cluster_id].
        metrics_c:    Metrics filtered to the same fold and C
                      [cluster_id, mlp_rmse, mlp_mae, mlp_r2, naive_rmse,
                      n_sectors, pop].
        sector_order: Sector IDs in the order GeoJSON features appear.

    Returns:
        z:    RMSE per sector, aligned to sector_order.
        text: Hover text per sector, aligned to sector_order.
    """
    cluster_of = asgn_c.set_index("sector_id")["cluster_id"]
    stats = metrics_c.set_index("cluster_id")

    z: list[float] = []
    text: list[str] = []
    for sector_id in sector_order:
        cid = int(cluster_of.get(sector_id, -1))
        if cid not in stats.index:
            z.append(float("nan"))
            text.append(f"<b>{sector_id}</b><br>no data")
            continue
        row = stats.loc[cid]
        z.append(float(row["mlp_rmse"]))
        text.append(
            f"<b>{sector_id}</b><br>"
            f"cluster {cid}<br>"
            f"RMSE={row['mlp_rmse']:.3f}<br>"
            f"MAE={row['mlp_mae']:.3f}<br>"
            f"R²={row['mlp_r2']:.3f}<br>"
            f"naive RMSE={row['naive_rmse']:.3f}<br>"
            f"sectors={int(row['n_sectors'])}<br>"
            f"pop={row['pop']:.0f}"
        )
    return z, text


# ── Figure assembly ──────────────────────────────────────────────────

def build_figure(
    metrics: pd.DataFrame,
    geojson: dict,
    ovitrap_locs: pd.DataFrame,
) -> go.Figure:
    """Build the fold-dropdown + C-slider RMSE choropleth figure.

    Each fold gets its own slider (steps = that fold's available C
    values); only the active fold's slider is visible at a time,
    toggled by the dropdown.  The dropdown also resets the choropleth
    to that fold's first C.  The colour scale is fixed globally
    (`metrics['mlp_rmse'].min()/.max()`) so shades stay comparable
    across every fold/C combination.

    Args:
        metrics:      Full metrics_skater.csv + parsed cluster_id.
        geojson:      GeoJSON FeatureCollection for BH sectors.
        ovitrap_locs: Unique ovitrap locations [idarmad, latitude, longitude].

    Returns:
        go.Figure ready to write as standalone HTML.
    """
    folds = sorted(metrics["fold_year"].unique())
    sector_order = [
        str(f["properties"]["CD_SETOR"]) for f in geojson["features"]
    ]
    rmse_min = float(metrics["mlp_rmse"].min())
    rmse_max = float(metrics["mlp_rmse"].max())

    # ── Precompute every (fold, C) frame ───────────────────────────────
    fold_c_values: dict[str, list[int]] = {}
    frames: dict[tuple[str, int], tuple[list[float], list[str]]] = {}
    for fold_year in folds:
        asgn = _load_fold_assignments(fold_year)
        fold_metrics = metrics[metrics["fold_year"] == fold_year]
        c_values = sorted(fold_metrics["C"].unique().tolist())
        fold_c_values[fold_year] = c_values
        for c_value in c_values:
            asgn_c = asgn[asgn["C"] == c_value]
            metrics_c = fold_metrics[fold_metrics["C"] == c_value]
            frames[(fold_year, c_value)] = _frame_for_c(
                asgn_c, metrics_c, sector_order
            )

    fold0 = folds[0]
    c0 = fold_c_values[fold0][0]
    z0, text0 = frames[(fold0, c0)]

    # ── Fixed map traces ──────────────────────────────────────────────
    choro = go.Choroplethmap(
        geojson=geojson,
        locations=sector_order,
        z=z0,
        text=text0,
        hovertemplate="%{text}<extra></extra>",
        featureidkey="properties.CD_SETOR",
        colorscale="RdYlGn_r",
        zmin=rmse_min,
        zmax=rmse_max,
        marker_opacity=0.75,
        marker_line_width=0.1,
        showscale=True,
        colorbar={"title": {"text": "RMSE"}},
        name="RMSE",
    )

    scatter_traps = go.Scattermap(
        lat=ovitrap_locs["latitude"],
        lon=ovitrap_locs["longitude"],
        mode="markers",
        marker={"size": 4, "color": "black", "opacity": 0.7},
        name="Ovitraps",
        customdata=ovitrap_locs["idarmad"],
        hovertemplate=(
            "Ovitrap %{customdata}<br>"
            "lat=%{lat:.4f}, lon=%{lon:.4f}"
            "<extra></extra>"
        ),
        showlegend=True,
    )

    # ── One slider per fold, only the active fold's slider visible ────
    def _slider_for_fold(fold_year: str, visible: bool) -> dict:
        steps = []
        for c_value in fold_c_values[fold_year]:
            z, text = frames[(fold_year, c_value)]
            steps.append({
                "label": str(c_value),
                "method": "restyle",
                "args": [{"z": [z], "text": [text]}, [0]],
            })
        return {
            "active": 0,
            "steps": steps,
            "visible": visible,
            "x": 0.05,
            "len": 0.9,
            "y": 0.02,
            "yanchor": "top",
            "currentvalue": {
                "prefix": f"{fold_year} — C = ",
                "visible": True,
                "xanchor": "center",
            },
        }

    sliders_all = [
        _slider_for_fold(fold_year, visible=(fold_year == fold0))
        for fold_year in folds
    ]

    # ── Dropdown: switch fold → reset map to that fold's first C and
    # swap which slider is visible ─────────────────────────────────────
    # Bracket-path relayout keys ("sliders[i].visible") toggle visibility
    # without re-serialising every slider's full step data per button —
    # re-sending the whole `sliders` array 3x (once per button) would
    # quadruple the embedded frame data and bloat the HTML.
    buttons = []
    for i, fold_year in enumerate(folds):
        c_first = fold_c_values[fold_year][0]
        z, text = frames[(fold_year, c_first)]
        layout_upd = {
            f"sliders[{j}].visible": (j == i) for j in range(len(folds))
        }
        buttons.append({
            "label": fold_year,
            "method": "update",
            "args": [{"z": [z], "text": [text]}, layout_upd, [0]],
        })

    fig = go.Figure(data=[choro, scatter_traps])
    fig.update_layout(
        map={
            "style": "open-street-map",
            "zoom": 11,
            "center": BH_CENTER,
            "domain": {"x": [0, 1], "y": [0.10, 1.0]},
        },
        sliders=sliders_all,
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "x": 0.01,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "top",
            "showactive": True,
        }],
        height=950,
        margin={"t": 60, "b": 20, "l": 50, "r": 20},
        title={
            "text": f"SKATER-CV — held-out region RMSE — {fold0}, C={c0}",
            "x": 0.5,
        },
        legend={"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
    )
    return fig


# ── Entry point ─────────────────────────────────────────────────────

def main() -> None:
    """Build and write the SKATER-CV RMSE choropleth HTML."""
    logger.info("Loading metrics, geojson, ovitrap locations")
    metrics = _load_metrics()
    geojson = _load_geojson()
    ovitrap_locs = _load_ovitrap_locations()

    fig = build_figure(metrics, geojson, ovitrap_locs)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUT_PATH)
    logger.info("RMSE map → %s", OUT_PATH)


if __name__ == "__main__":
    main()
