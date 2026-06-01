"""Sector-level dengue prediction dashboard.

Applies saved fold MLP models (trained on citywide data) to each BH
census sector. Features use EB-smoothed sector dengue lags and IDW
egg counts from the sector's centroid. Error is measured against
EB rates to suppress noise from low-population sectors.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

FOLD_YEARS = ["2015_16", "2018_19", "2023_24"]
FEATURE_ORDER = [
    "eb_rate_per_1000_lag1",
    "eb_rate_per_1000_lag2",
    "eb_rate_per_1000_lag3",
    "idw_egg_value_lag3",
    "idw_egg_value_lag4",
    "week_sin",
    "week_cos",
]
BH_CENTER = {"lat": -19.917, "lon": -43.934}


def _extract_week_num(biweek: str) -> int:
    return int(biweek.split("W")[1])


def load_sector_dengue() -> pd.DataFrame:
    path = Path("data/dvc/add_population_info/dengue_per_capita.csv")
    logger.info("Loading sector dengue data...")
    df = pd.read_csv(
        path,
        usecols=["sector_id", "biweek", "eb_rate_per_1000", "case_count"],
        dtype={"sector_id": str},
    )
    logger.info("  %d rows, %d sectors", len(df), df["sector_id"].nunique())
    return df


def load_sector_idw() -> pd.DataFrame:
    path = Path(
        "data/dvc/add_population_info/sector_centroids_with_idw.csv"
    )
    logger.info("Loading sector IDW egg data...")
    df = pd.read_csv(
        path,
        usecols=["CD_SETOR", "biweek", "idw_egg_value"],
        dtype={"CD_SETOR": str},
    )
    df = df.rename(columns={"CD_SETOR": "sector_id"})
    logger.info("  %d rows, %d sectors", len(df), df["sector_id"].nunique())
    return df


def build_feature_matrix(
    dengue: pd.DataFrame, idw: pd.DataFrame
) -> pd.DataFrame:
    """Vectorized lag creation per sector via groupby.shift."""
    logger.info("Building lag features (dengue)...")
    dengue = dengue.sort_values(["sector_id", "biweek"]).copy()
    grp_d = dengue.groupby("sector_id")["eb_rate_per_1000"]
    for lag in range(1, 4):
        dengue[f"eb_rate_per_1000_lag{lag}"] = grp_d.shift(lag)

    logger.info("Building lag features (IDW eggs)...")
    idw = idw.sort_values(["sector_id", "biweek"]).copy()
    grp_e = idw.groupby("sector_id")["idw_egg_value"]
    for lag in range(1, 5):
        idw[f"idw_egg_value_lag{lag}"] = grp_e.shift(lag)
    idw = idw.drop(
        columns=["idw_egg_value", "idw_egg_value_lag1", "idw_egg_value_lag2"]
    )

    logger.info("Merging and adding seasonality features...")
    df = dengue.merge(idw, on=["sector_id", "biweek"], how="inner")
    week_num = df["biweek"].apply(_extract_week_num)
    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)

    before = len(df)
    df = df.dropna(subset=FEATURE_ORDER).copy()
    logger.info(
        "  %d rows after dropping NaN lags (dropped %d)",
        len(df),
        before - len(df),
    )
    return df


def predict_fold(
    features: pd.DataFrame, test_year: str
) -> pd.DataFrame:
    """Apply saved fold model to all sectors for test_year biweeks."""
    model_path = Path(
        f"results/mlp_bh_rate_predictor/fold_{test_year}/model.joblib"
    )
    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]

    mask = features["biweek"].str.startswith(test_year)
    test_df = features[mask].copy()

    if test_df.empty:
        logger.warning("  No test data for %s", test_year)
        return pd.DataFrame()

    X_scaled = scaler.transform(test_df[FEATURE_ORDER].values)
    preds = np.maximum(model.predict(X_scaled), 0.0)

    test_df["predicted"] = preds
    actual = test_df["eb_rate_per_1000"]
    test_df["abs_error"] = (actual - preds).abs()
    test_df["sq_error"] = (actual - preds) ** 2
    test_df["fold"] = test_year

    logger.info(
        "  %s: %d predictions, mean RMSE=%.4f",
        test_year,
        len(test_df),
        np.sqrt(test_df["sq_error"].mean()),
    )
    return test_df[[
        "sector_id", "biweek", "fold",
        "eb_rate_per_1000", "predicted",
        "abs_error", "sq_error",
    ]]


def aggregate_sector_errors(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("sector_id")
        .agg(
            rmse=("sq_error", lambda x: np.sqrt(x.mean())),
            mae=("abs_error", "mean"),
            n=("sq_error", "count"),
        )
        .reset_index()
    )


def _fold_error_table(fold_df: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_df.groupby("sector_id")
        .agg(rmse=("sq_error", lambda x: np.sqrt(x.mean())))
        .reset_index()
    )


def _case_totals(
    sector_year_cases: pd.DataFrame,
    years: list[str],
) -> pd.DataFrame:
    return (
        sector_year_cases[sector_year_cases["year"].isin(years)]
        .groupby("sector_id")["total_cases"]
        .sum()
        .reset_index()
    )


def _make_choropleth(
    geojson: dict,
    sector_errors: pd.DataFrame,
    visible: bool,
    zmax: float,
) -> go.Choroplethmap:
    cases = (
        sector_errors["total_cases"].fillna(0).astype(int).tolist()
        if "total_cases" in sector_errors.columns
        else [0] * len(sector_errors)
    )
    return go.Choroplethmap(
        geojson=geojson,
        locations=sector_errors["sector_id"],
        z=sector_errors["rmse"],
        customdata=cases,
        featureidkey="properties.CD_SETOR",
        colorscale="YlOrRd",
        zmin=0,
        zmax=zmax,
        marker_opacity=0.7,
        marker_line_width=0.1,
        colorbar_title="RMSE",
        hovertemplate=(
            "<b>Sector %{location}</b><br>"
            "RMSE: %{z:.4f}<br>"
            "Cases: %{customdata}<extra></extra>"
        ),
        visible=visible,
    )


def _make_histogram(
    sector_errors: pd.DataFrame, visible: bool, xmax: float
) -> go.Histogram:
    bin_size = xmax / 60
    clipped = sector_errors["rmse"].clip(upper=xmax)
    return go.Histogram(
        x=clipped,
        marker_color="steelblue",
        opacity=0.75,
        xbins={"start": 0, "end": xmax + bin_size, "size": bin_size},
        visible=visible,
        showlegend=False,
        hovertemplate="RMSE: %{x:.3f}<br>Sectors: %{y}<extra></extra>",
    )


def build_figure(
    mean_errors: pd.DataFrame,
    fold_results: dict[str, pd.DataFrame],
    geojson: dict,
    sector_year_cases: pd.DataFrame,
) -> go.Figure:
    # Attach total cases (sum over all fold years) to mean view
    mean_ct = _case_totals(sector_year_cases, FOLD_YEARS)
    mean_view = mean_errors.merge(mean_ct, on="sector_id", how="left")

    views: dict[str, pd.DataFrame] = {"Mean (all folds)": mean_view}
    for yr in FOLD_YEARS:
        if yr in fold_results and not fold_results[yr].empty:
            err = _fold_error_table(fold_results[yr])
            ct = _case_totals(sector_year_cases, [yr])
            views[f"Test: {yr}"] = err.merge(ct, on="sector_id", how="left")

    zmax = float(mean_errors["rmse"].quantile(0.95))
    xmax = 10.0

    # Each view = 1 choropleth (even idx) + 1 histogram (odd idx)
    traces: list = []
    for i, (label, data) in enumerate(views.items()):
        visible = i == 0
        traces.append(_make_choropleth(geojson, data, visible, zmax))
        traces.append(_make_histogram(data, visible, xmax))

    n = len(traces)
    buttons = []
    for i, label in enumerate(views):
        visibility = [False] * n
        visibility[2 * i] = True       # choropleth
        visibility[2 * i + 1] = True   # histogram
        buttons.append({
            "label": label,
            "method": "update",
            "args": [
                {"visible": visibility},
                {"title": f"Sector Prediction Error — {label}"},
            ],
        })

    fig = go.Figure(data=traces)
    fig.update_layout(
        # Map occupies top 62% of figure height
        map={
            "domain": {"x": [0, 1], "y": [0.38, 1.0]},
            "style": "open-street-map",
            "zoom": 11,
            "center": BH_CENTER,
        },
        # Histogram occupies bottom 30%
        xaxis={
            "domain": [0.05, 0.95],
            "anchor": "y",
            "title": "RMSE (EB rate per 1 000)",
            "range": [0, xmax + xmax / 60],
        },
        yaxis={
            "domain": [0.0, 0.30],
            "anchor": "x",
            "title": "Sectors",
        },
        margin={"r": 20, "t": 80, "l": 60, "b": 60},
        height=1000,
        title="Sector Prediction Error — Mean (all folds)",
        updatemenus=[{
            "type": "dropdown",
            "direction": "down",
            "showactive": True,
            "buttons": buttons,
            "x": 0.01,
            "xanchor": "left",
            "y": 0.99,
            "yanchor": "top",
        }],
    )
    return fig


def main() -> None:
    dengue = load_sector_dengue()
    idw = load_sector_idw()
    features = build_feature_matrix(dengue, idw)

    logger.info("Running fold predictions...")
    fold_results: dict[str, pd.DataFrame] = {}
    all_preds: list[pd.DataFrame] = []
    for yr in FOLD_YEARS:
        logger.info("  Fold %s...", yr)
        result = predict_fold(features, yr)
        fold_results[yr] = result
        if not result.empty:
            all_preds.append(result)

    all_results = pd.concat(all_preds, ignore_index=True)
    logger.info(
        "Total predictions: %d across %d sectors",
        len(all_results),
        all_results["sector_id"].nunique(),
    )

    logger.info("Aggregating per-sector errors...")
    mean_errors = aggregate_sector_errors(all_results)
    logger.info(
        "  Mean RMSE=%.4f  Median RMSE=%.4f",
        mean_errors["rmse"].mean(),
        mean_errors["rmse"].median(),
    )

    logger.info("Loading GeoJSON...")
    with open(
        "data/dvc/process_population_data/"
        "bh_sectors_2022_with_populations.geojson"
    ) as f:
        geojson = json.load(f)

    logger.info("Computing sector case totals per year...")
    dengue_raw = dengue[["sector_id", "biweek", "case_count"]].copy()
    dengue_raw["year"] = (
        dengue_raw["biweek"].str.rsplit("W", n=1).str[0]
    )
    sector_year_cases = (
        dengue_raw.groupby(["sector_id", "year"])["case_count"]
        .sum()
        .reset_index()
        .rename(columns={"case_count": "total_cases"})
    )

    logger.info("Building map figure...")
    fig = build_figure(mean_errors, fold_results, geojson, sector_year_cases)

    output = Path("results/sector_mlp_dashboard.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    logger.info("Saved: %s", output)


if __name__ == "__main__":
    main()
