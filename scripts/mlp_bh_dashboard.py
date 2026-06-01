"""Interactive dashboard for MLP vs Naive predictor comparison.

Visualizes prediction results with time series, error metrics, and scatter plots.
Supports 4-fold cross-validation with year selector dropdown.
Saves as HTML for easy sharing.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EPIDEMY_YEARS = ["2012_13", "2015_16", "2018_19", "2023_24"]


def load_fold_results(base_dir: Path, test_year: str) -> tuple:
    """Load metrics and predictions for a specific fold."""
    fold_dir = base_dir / f"fold_{test_year}"

    with open(fold_dir / "metrics.json") as f:
        metrics = json.load(f)

    predictions = pd.read_csv(fold_dir / "predictions.csv")

    return metrics, predictions


def load_all_results(base_dir: Path) -> dict:
    """Load all CV fold results."""
    all_results = {}
    for year in EPIDEMY_YEARS:
        try:
            metrics, predictions = load_fold_results(base_dir, year)
            all_results[year] = {
                "metrics": metrics,
                "predictions": predictions,
            }
        except FileNotFoundError:
            logger.warning("Results for %s not found", year)
    return all_results


def _add_time_series_traces(
    fig: go.Figure,
    p: pd.DataFrame,
    year: str,
    first_year: str,
) -> None:
    visible = year == first_year
    p = p[p["split"] == "test"]
    for name, ycol, color, dash in [
        ("Actual", "target_rate", "black", None),
        ("MLP", "mlp_predicted", "blue", "dash"),
        ("Naive", "naive_predicted", "red", "dot"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=p["biweek"],
                y=p[ycol],
                mode="lines+markers",
                name=f"{name} ({year})",
                line={"color": color, "width": 2, "dash": dash},
                marker={"size": 6},
                visible=visible,
                meta={"year": year},
            ),
            row=1,
            col=1,
        )

    if "mean_eggs" in p.columns:
        fig.add_trace(
            go.Scatter(
                x=p["biweek"],
                y=p["mean_eggs"],
                mode="lines+markers",
                name=f"Eggs ({year})",
                line={"color": "green", "width": 2},
                marker={"size": 6, "symbol": "diamond"},
                yaxis="y5",
                visible=visible,
                meta={"year": year},
            ),
            row=1,
            col=1,
        )


def _add_metrics_traces(
    fig: go.Figure,
    metrics: dict,
    year: str,
    first_year: str,
) -> None:
    visible = year == first_year
    categories = ["Train RMSE", "Test RMSE", "Train MAE", "Test MAE"]
    mlp_vals = [
        metrics["mlp"]["train"]["rmse"],
        metrics["mlp"]["test"]["rmse"],
        metrics["mlp"]["train"]["mae"],
        metrics["mlp"]["test"]["mae"],
    ]
    naive_vals = [
        metrics["naive"]["train"]["rmse"],
        metrics["naive"]["test"]["rmse"],
        metrics["naive"]["train"]["mae"],
        metrics["naive"]["test"]["mae"],
    ]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=mlp_vals,
            name=f"MLP ({year})",
            marker_color="blue",
            visible=visible,
            meta={"year": year},
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=naive_vals,
            name=f"Naive ({year})",
            marker_color="red",
            visible=visible,
            meta={"year": year},
        ),
        row=1,
        col=2,
    )

    train_time_s = metrics["mlp"].get("train_time_s")
    if train_time_s is not None:
        fig.add_trace(
            go.Scatter(
                x=["Test RMSE"],
                y=[max(mlp_vals) * 1.18],
                mode="text",
                text=[f"Grid search: {train_time_s:.1f}s"],
                textfont={"size": 12, "color": "steelblue"},
                showlegend=False,
                visible=visible,
                meta={"year": year},
            ),
            row=1,
            col=2,
        )


def _add_residual_traces(
    fig: go.Figure,
    p: pd.DataFrame,
    year: str,
    first_year: str,
) -> None:
    visible = year == first_year
    p["mlp_residual"] = p["target_rate"] - p["mlp_predicted"]
    for split, color in [("train", "lightblue"), ("test", "orange")]:
        split_data = p[p["split"] == split]
        fig.add_trace(
            go.Scatter(
                x=split_data["biweek"],
                y=split_data["mlp_residual"],
                mode="markers",
                name=f"Residual {split} ({year})",
                marker={"color": color, "size": 8},
                visible=visible,
                meta={"year": year},
            ),
            row=2,
            col=1,
        )


def _add_scatter_traces(
    fig: go.Figure,
    p: pd.DataFrame,
    year: str,
    first_year: str,
) -> None:
    visible = year == first_year
    for model, pred_col in [
        ("MLP", "mlp_predicted"),
        ("Naive", "naive_predicted"),
    ]:
        symbol = "circle" if model == "MLP" else "diamond"
        for split, color in [("train", "blue"), ("test", "red")]:
            split_data = p[p["split"] == split]
            fig.add_trace(
                go.Scatter(
                    x=split_data["target_rate"],
                    y=split_data[pred_col],
                    mode="markers",
                    name=f"{model} {split} ({year})",
                    marker={
                        "color": color,
                        "size": 8,
                        "opacity": 0.6,
                        "symbol": symbol,
                    },
                    visible=visible,
                    meta={"year": year},
                ),
                row=2,
                col=2,
            )

    max_val = p["target_rate"].max() * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line={"color": "black", "width": 1, "dash": "dash"},
            name="Perfect Prediction",
            visible=visible,
            meta={"year": year},
        ),
        row=2,
        col=2,
    )


_INPUT_COLS = [
    "cases_per_1000_lag1",
    "cases_per_1000_lag2",
    "cases_per_1000_lag3",
    "mean_eggs_lag3",
    "mean_eggs_lag4",
    "week_sin",
    "week_cos",
]

_TABLE_HEADERS = [
    "Biweek", "Split",
    "Cases Lag1", "Cases Lag2", "Cases Lag3",
    "Eggs Lag3", "Eggs Lag4",
    "Week Sin", "Week Cos",
    "Actual Rate", "MLP Pred", "Naive Pred",
]

_DISPLAY_COLS = (
    ["biweek", "split"]
    + _INPUT_COLS
    + ["target_rate", "mlp_predicted", "naive_predicted"]
)


def _add_feature_table(
    fig: go.Figure,
    p: pd.DataFrame,
    year: str,
    first_year: str,
) -> None:
    row_colors = [
        "lightblue" if s == "train" else "lightyellow"
        for s in p["split"]
    ]
    cell_values = [
        p[col].round(4) if col not in {"biweek", "split"} else p[col]
        for col in _DISPLAY_COLS
    ]
    fig.add_trace(
        go.Table(
            header={
                "values": _TABLE_HEADERS,
                "fill_color": "steelblue",
                "font": {"color": "white", "size": 11},
                "align": "center",
            },
            cells={
                "values": cell_values,
                "fill_color": [row_colors] * len(_DISPLAY_COLS),
                "align": "center",
                "font": {"size": 10},
            },
            visible=(year == first_year),
            meta={"year": year},
        ),
        row=3,
        col=1,
    )


def _build_dropdown_buttons(
    year_indices: dict[str, list[int]], total_traces: int
) -> list[dict]:
    buttons = []
    for year in EPIDEMY_YEARS:
        if year not in year_indices:
            continue
        visibility = [False] * total_traces
        for idx in year_indices[year]:
            visibility[idx] = True
        buttons.append(
            {
                "label": f"Test: {year}",
                "method": "update",
                "args": [
                    {"visible": visibility},
                    {"title": f"MLP Dengue Predictor - Test Year: {year}"},
                ],
            }
        )
    return buttons


def create_interactive_dashboard(all_results: dict) -> go.Figure:
    """Create interactive dashboard with year selector using visibility toggling."""
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Time Series",
            "Error Metrics",
            "Residuals",
            "Scatter Plot",
            "Model Inputs",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "table", "colspan": 2}, None],
        ],
        row_heights=[0.35, 0.35, 0.30],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    year_indices: dict[str, list[int]] = {}
    first_year = next(iter(all_results))

    for year in EPIDEMY_YEARS:
        if year not in all_results:
            continue

        metrics = all_results[year]["metrics"]
        p = all_results[year]["predictions"].sort_values("biweek").copy()
        start = len(fig.data)

        _add_time_series_traces(fig, p, year, first_year)
        _add_metrics_traces(fig, metrics, year, first_year)
        _add_residual_traces(fig, p, year, first_year)
        _add_scatter_traces(fig, p, year, first_year)
        _add_feature_table(fig, p, year, first_year)

        year_indices[year] = list(range(start, len(fig.data)))

    buttons = _build_dropdown_buttons(year_indices, len(fig.data))
    first_title = f"MLP Dengue Predictor - Test Year: {next(iter(all_results))}"

    fig.update_layout(
        yaxis5={
            "title": "Mean Egg Count",
            "overlaying": "y",
            "side": "right",
            "showgrid": False,
            "anchor": "x",
        },
        barmode="group",
        updatemenus=[
            {
                "type": "dropdown",
                "direction": "down",
                "showactive": True,
                "buttons": buttons,
                "x": 0.23,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "pad": {"r": 10, "t": 10},
            }
        ],
        title={"text": first_title, "y": 0.98, "yanchor": "top"},
        annotations=[
            {
                "text": "Select Test Year:",
                "x": 0.0,
                "y": 1.15,
                "xref": "paper",
                "yref": "paper",
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        template="plotly_white",
        height=1350,
        margin={"t": 150, "r": 250},
        showlegend=True,
        legend={
            "x": 1.02,
            "y": 0.5,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "left",
            "yanchor": "middle",
            "bgcolor": "rgba(255, 255, 255, 0.9)",
            "bordercolor": "gray",
            "borderwidth": 1,
            "font": {"size": 11},
        },
    )

    fig.update_xaxes(title_text="Biweek", row=1, col=1)
    fig.update_xaxes(title_text="", row=1, col=2)
    fig.update_xaxes(title_text="Biweek", row=2, col=1)
    fig.update_xaxes(title_text="Actual Rate", row=2, col=2)
    fig.update_yaxes(title_text="Cases per 1000", row=1, col=1)
    fig.update_yaxes(title_text="Error Value", row=1, col=2)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_yaxes(title_text="Predicted Rate", row=2, col=2)

    return fig


def load_full_historical_data() -> pd.DataFrame:
    """Load city-wide dengue rate and mean egg count for all biweeks."""
    dengue = pd.read_csv(
        Path("data/dvc/add_population_info/dengue_citywide_per_capita.csv")
    )
    ovi = pd.read_csv(
        Path("data/dvc/add_population_info/ovitraps_data.csv"),
        low_memory=False,
    )
    ovi_agg = (
        ovi.groupby("biweek")["novos"].mean().reset_index()
        .rename(columns={"novos": "mean_eggs"})
    )
    df = dengue[["biweek", "cases_per_1000"]].merge(
        ovi_agg, on="biweek", how="outer"
    )
    return df.sort_values("biweek").reset_index(drop=True)


def create_full_history_figure(df: pd.DataFrame) -> go.Figure:
    """Dual-axis time series of dengue rate and egg count across all years."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df["biweek"],
            y=df["cases_per_1000"],
            mode="lines",
            name="Dengue (per 1000)",
            line={"color": "red", "width": 1.5},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df["biweek"],
            y=df["mean_eggs"],
            mode="lines",
            name="Mean egg count",
            line={"color": "green", "width": 1.5},
        ),
        secondary_y=True,
    )

    for yr in EPIDEMY_YEARS:
        yr_data = df[df["biweek"].str.startswith(yr)]
        if yr_data.empty:
            continue
        fig.add_vrect(
            x0=yr_data["biweek"].iloc[0],
            x1=yr_data["biweek"].iloc[-1],
            fillcolor="blue",
            opacity=0.10,
            line_width=0,
            annotation_text=yr,
            annotation_position="top left",
            annotation_font_size=10,
        )

    fig.update_layout(
        title="Full Historical Overview — Dengue Rate vs Egg Count "
              "(epidemic years shaded)",
        template="plotly_white",
        height=400,
        xaxis_title="Biweek",
        margin={"t": 60, "b": 80},
    )
    fig.update_yaxes(
        title_text="Cases per 1000", secondary_y=False, color="red"
    )
    fig.update_yaxes(
        title_text="Mean Egg Count", secondary_y=True, color="green"
    )
    return fig


def main() -> None:
    """Generate interactive dashboard HTML file with year selector."""
    results_dir = Path("results/mlp_bh_rate_predictor")

    if not results_dir.exists():
        logger.error(
            "Results directory not found: %s. "
            "Please run src/mlp_bh_rate_predictor.py first.",
            results_dir,
        )
        return

    logger.info("Loading all CV fold results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        logger.error(
            "No fold results found. Please run the predictor first."
        )
        return

    try:
        with open(results_dir / "cv_summary.json") as f:
            cv_summary = json.load(f)
    except FileNotFoundError:
        cv_summary = None

    logger.info(
        "Loaded %d folds: %s", len(all_results), list(all_results.keys())
    )

    logger.info("Loading full historical data...")
    hist_df = load_full_historical_data()
    history_fig = create_full_history_figure(hist_df)

    logger.info("Creating interactive dashboard with year selector...")
    dashboard = create_interactive_dashboard(all_results)

    output_path = results_dir / "dashboard.html"
    history_html = history_fig.to_html(
        full_html=False, include_plotlyjs="cdn"
    )
    dashboard_html = dashboard.to_html(
        full_html=False, include_plotlyjs=False
    )
    with open(output_path, "w") as f:
        f.write(
            "<html><head><meta charset='utf-8'></head><body>"
            + "<h2 style='font-family:sans-serif;margin:20px'>"
            "Full Historical Data</h2>"
            + history_html
            + "<h2 style='font-family:sans-serif;margin:20px'>"
            "CV Fold Results</h2>"
            + dashboard_html
            + "</body></html>"
        )
    logger.info("Dashboard saved to: %s", output_path)

    logger.info("=" * 60)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 60)

    for year in EPIDEMY_YEARS:
        if year in all_results:
            m = all_results[year]["metrics"]
            naive_rmse = m["naive"]["test"]["rmse"]
            if naive_rmse > 0:
                improve = (1 - m["mlp"]["test"]["rmse"] / naive_rmse) * 100
                improve_str = f"{improve:+.1f}%"
            else:
                improve_str = "N/A (Naive perfect)"
            logger.info(
                "%s: MLP RMSE=%.3f, Naive RMSE=%.3f, Improvement=%s",
                year,
                m["mlp"]["test"]["rmse"],
                naive_rmse,
                improve_str,
            )

    if cv_summary:
        logger.info("-" * 60)
        logger.info(
            "Mean MLP Test RMSE: %.4f",
            cv_summary["mean_mlp_test_rmse"],
        )
        logger.info(
            "Mean Naive Test RMSE: %.4f",
            cv_summary["mean_naive_test_rmse"],
        )

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
