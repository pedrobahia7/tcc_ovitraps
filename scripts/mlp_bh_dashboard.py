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
        rows=2,
        cols=2,
        subplot_titles=(
            "Time Series",
            "Error Metrics",
            "Residuals",
            "Scatter Plot",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
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
        height=950,
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

    logger.info("Creating interactive dashboard with year selector...")
    dashboard = create_interactive_dashboard(all_results)

    output_path = results_dir / "dashboard.html"
    dashboard.write_html(str(output_path))
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
