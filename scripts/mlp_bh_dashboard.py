"""Interactive dashboard for MLP vs Naive predictor comparison.

Visualizes prediction results with time series, error metrics, and scatter plots.
Supports 4-fold cross-validation with year selector dropdown.
Saves as HTML for easy sharing.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            print(f"Warning: Results for {year} not found")
    return all_results


def create_time_series_plot(
    predictions: pd.DataFrame, test_year: str
) -> go.Figure:
    """Create time series plot of actual vs predicted rates and egg counts."""
    predictions = predictions.sort_values("biweek").copy()

    fig = go.Figure()

    # Actual values
    fig.add_trace(
        go.Scatter(
            x=predictions["biweek"],
            y=predictions["target_rate"],
            mode="lines+markers",
            name="Actual Rate",
            line=dict(color="black", width=2),
            marker=dict(size=6),
        )
    )

    # MLP predictions
    fig.add_trace(
        go.Scatter(
            x=predictions["biweek"],
            y=predictions["mlp_predicted"],
            mode="lines+markers",
            name="MLP Predicted",
            line=dict(color="blue", width=2, dash="dash"),
            marker=dict(size=6),
        )
    )

    # Naive predictions
    fig.add_trace(
        go.Scatter(
            x=predictions["biweek"],
            y=predictions["naive_predicted"],
            mode="lines+markers",
            name="Naive Predicted",
            line=dict(color="red", width=2, dash="dot"),
            marker=dict(size=6),
        )
    )

    # Egg counts on secondary y-axis
    if "mean_eggs" in predictions.columns:
        fig.add_trace(
            go.Scatter(
                x=predictions["biweek"],
                y=predictions["mean_eggs"],
                mode="lines+markers",
                name="Mean Eggs",
                line=dict(color="green", width=2),
                marker=dict(size=6, symbol="diamond"),
                yaxis="y2",
            )
        )

    # Highlight train/test split
    train_data = predictions[predictions["split"] == "train"]
    test_data = predictions[predictions["split"] == "test"]

    if len(train_data) > 0 and len(test_data) > 0:
        last_train_idx = len(train_data) - 1
        fig.add_annotation(
            x=last_train_idx,
            y=predictions["target_rate"].max() * 0.9,
            text="Train→Test",
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-30,
        )

    fig.update_layout(
        title=f"Time Series - Test Year: {test_year}",
        xaxis_title="Biweek",
        yaxis_title="Cases per 1000 Population",
        yaxis2=dict(
            title="Mean Egg Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_metrics_comparison(metrics: dict, test_year: str) -> go.Figure:
    """Create bar chart comparing MLP vs Naive metrics."""
    categories = ["Train RMSE", "Test RMSE", "Train MAE", "Test MAE"]

    mlp_values = [
        metrics["mlp"]["train"]["rmse"],
        metrics["mlp"]["test"]["rmse"],
        metrics["mlp"]["train"]["mae"],
        metrics["mlp"]["test"]["mae"],
    ]

    naive_values = [
        metrics["naive"]["train"]["rmse"],
        metrics["naive"]["test"]["rmse"],
        metrics["naive"]["train"]["mae"],
        metrics["naive"]["test"]["mae"],
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(name="MLP", x=categories, y=mlp_values, marker_color="blue")
    )

    fig.add_trace(
        go.Bar(
            name="Naive", x=categories, y=naive_values, marker_color="red"
        )
    )

    # Add improvement annotation (handle zero naive RMSE)
    naive_rmse = metrics["naive"]["test"]["rmse"]
    if naive_rmse > 0:
        rmse_improve = (
            1 - metrics["mlp"]["test"]["rmse"] / naive_rmse
        ) * 100
        improve_text = f"RMSE Improvement: {rmse_improve:+.1f}%"
        improve_color = "green" if rmse_improve > 0 else "red"
    else:
        improve_text = "Naive RMSE = 0 (perfect)"
        improve_color = "gray"

    fig.add_annotation(
        x=0.5,
        y=max(max(mlp_values), max(naive_values)) * 1.1,
        text=improve_text,
        showarrow=False,
        font=dict(size=14, color=improve_color),
    )

    fig.update_layout(
        title=f"Error Metrics - Test Year: {test_year}",
        yaxis_title="Error Value",
        barmode="group",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_residual_plot(
    predictions: pd.DataFrame, test_year: str
) -> go.Figure:
    """Create residual plot showing prediction errors over time."""
    predictions = predictions.sort_values("biweek").copy()

    predictions["mlp_residual"] = (
        predictions["target_rate"] - predictions["mlp_predicted"]
    )
    predictions["naive_residual"] = (
        predictions["target_rate"] - predictions["naive_predicted"]
    )

    fig = go.Figure()

    # Color by split
    for split, color in [("train", "lightblue"), ("test", "orange")]:
        split_data = predictions[predictions["split"] == split]

        fig.add_trace(
            go.Scatter(
                x=split_data["biweek"],
                y=split_data["mlp_residual"],
                mode="markers",
                name=f"MLP Residual ({split})",
                marker=dict(color=color, size=8, symbol="circle"),
                opacity=0.7,
            )
        )

    # Add zero line
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))

    fig.update_layout(
        title=f"MLP Residuals - Test Year: {test_year}",
        xaxis_title="Biweek",
        yaxis_title="Residual (Actual - Predicted)",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def create_scatter_plot(
    predictions: pd.DataFrame, test_year: str
) -> go.Figure:
    """Create scatter plot of predicted vs actual values."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "MLP: Predicted vs Actual",
            "Naive: Predicted vs Actual",
        ),
    )

    # MLP scatter
    for split, color in [("train", "blue"), ("test", "red")]:
        split_data = predictions[predictions["split"] == split]

        fig.add_trace(
            go.Scatter(
                x=split_data["target_rate"],
                y=split_data["mlp_predicted"],
                mode="markers",
                name=f"MLP {split}",
                marker=dict(color=color, size=8, opacity=0.6),
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Naive scatter
    for split, color in [("train", "blue"), ("test", "red")]:
        split_data = predictions[predictions["split"] == split]

        fig.add_trace(
            go.Scatter(
                x=split_data["target_rate"],
                y=split_data["naive_predicted"],
                mode="markers",
                name=f"Naive {split}",
                marker=dict(
                    color=color, size=8, opacity=0.6, symbol="diamond"
                ),
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # Add diagonal reference line
    max_val = predictions["target_rate"].max() * 1.1
    min_val = 0

    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                name="Perfect Prediction",
                showlegend=col == 1,
            ),
            row=1,
            col=col,
        )

    fig.update_xaxes(title_text="Actual Rate", row=1, col=1)
    fig.update_xaxes(title_text="Actual Rate", row=1, col=2)
    fig.update_yaxes(title_text="Predicted Rate", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Rate", row=1, col=2)

    fig.update_layout(
        title=f"Predicted vs Actual - Test Year: {test_year}",
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_cv_summary_plot(all_results: dict) -> go.Figure:
    """Create summary plot comparing all CV folds."""
    years = list(all_results.keys())
    mlp_rmse = [
        all_results[y]["metrics"]["mlp"]["test"]["rmse"] for y in years
    ]
    naive_rmse = [
        all_results[y]["metrics"]["naive"]["test"]["rmse"] for y in years
    ]
    mlp_r2 = [
        all_results[y]["metrics"]["mlp"]["test"]["r2"] for y in years
    ]
    naive_r2 = [
        all_results[y]["metrics"]["naive"]["test"]["r2"] for y in years
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Test RMSE by Year", "Test R² by Year"),
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
    )

    # RMSE comparison
    fig.add_trace(
        go.Scatter(
            x=years,
            y=mlp_rmse,
            mode="lines+markers",
            name="MLP RMSE",
            marker=dict(size=10),
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=naive_rmse,
            mode="lines+markers",
            name="Naive RMSE",
            marker=dict(size=10, symbol="diamond"),
            line=dict(width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # R² comparison
    fig.add_trace(
        go.Scatter(
            x=years,
            y=mlp_r2,
            mode="lines+markers",
            name="MLP R²",
            marker=dict(size=10),
            line=dict(width=2),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=naive_r2,
            mode="lines+markers",
            name="Naive R²",
            marker=dict(size=10, symbol="diamond"),
            line=dict(width=2, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Cross-Validation Summary: All Folds",
        template="plotly_white",
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.update_xaxes(title_text="Test Year", row=1, col=1)
    fig.update_xaxes(title_text="Test Year", row=1, col=2)

    return fig


def create_fold_dashboard(
    metrics: dict, predictions: pd.DataFrame, test_year: str
) -> go.Figure:
    """Create comprehensive dashboard for a single fold."""
    time_series = create_time_series_plot(predictions, test_year)
    metrics_comp = create_metrics_comparison(metrics, test_year)
    residual = create_residual_plot(predictions, test_year)
    scatter = create_scatter_plot(predictions, test_year)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Time Series - {test_year}",
            f"Error Metrics - {test_year}",
            f"Residuals - {test_year}",
            f"Scatter Plot - {test_year}",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for trace in time_series.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in metrics_comp.data:
        fig.add_trace(trace, row=1, col=2)

    for trace in residual.data:
        fig.add_trace(trace, row=2, col=1)

    for trace in scatter.data:
        fig.add_trace(trace, row=2, col=2)

    # Add diagonal line to scatter
    max_val = predictions["target_rate"].max() * 1.1
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", width=1, dash="dash"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def create_interactive_dashboard(
    all_results: dict, cv_summary: dict
) -> go.Figure:
    """Create interactive dashboard with year selector using visibility toggling."""
    # Build a single figure with all traces from all years
    # Each trace gets a meta.year tag for visibility control
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

    buttons = []
    trace_idx = 0
    year_trace_map = {}

    for year in EPIDEMY_YEARS:
        if year not in all_results:
            continue

        metrics = all_results[year]["metrics"]
        predictions = all_results[year]["predictions"]

        # Time series traces (row 1, col 1): 4 traces (actual, mlp, naive, eggs)
        year_start = trace_idx
        p = predictions.sort_values("biweek")
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
                    line=dict(color=color, width=2, dash=dash),
                    marker=dict(size=6),
                    visible=(year == list(all_results.keys())[0]),
                    legendgroup=year,
                    showlegend=True,
                    meta=dict(year=year),
                ),
                row=1,
                col=1,
            )
            trace_idx += 1

        # Eggs trace
        if "mean_eggs" in p.columns:
            fig.add_trace(
                go.Scatter(
                    x=p["biweek"],
                    y=p["mean_eggs"],
                    mode="lines+markers",
                    name=f"Eggs ({year})",
                    line=dict(color="green", width=2),
                    marker=dict(size=6, symbol="diamond"),
                    yaxis="y5",
                    visible=(year == list(all_results.keys())[0]),
                    legendgroup=year,
                    showlegend=True,
                    meta=dict(year=year),
                ),
                row=1,
                col=1,
            )
            trace_idx += 1

        # Metrics comparison (row 1, col 2): 2 traces (MLP bar, Naive bar)
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
                visible=(year == list(all_results.keys())[0]),
                legendgroup=year,
                showlegend=True,
                meta=dict(year=year),
            ),
            row=1,
            col=2,
        )
        trace_idx += 1

        fig.add_trace(
            go.Bar(
                x=categories,
                y=naive_vals,
                name=f"Naive ({year})",
                marker_color="red",
                visible=(year == list(all_results.keys())[0]),
                legendgroup=year,
                showlegend=True,
                meta=dict(year=year),
            ),
            row=1,
            col=2,
        )
        trace_idx += 1

        # Residuals (row 2, col 1): 2 traces (train, test)
        p = p.sort_values("biweek")
        p["mlp_residual"] = p["target_rate"] - p["mlp_predicted"]
        for split, color in [("train", "lightblue"), ("test", "orange")]:
            split_data = p[p["split"] == split]
            fig.add_trace(
                go.Scatter(
                    x=split_data["biweek"],
                    y=split_data["mlp_residual"],
                    mode="markers",
                    name=f"Residual {split} ({year})",
                    marker=dict(color=color, size=8),
                    visible=(year == list(all_results.keys())[0]),
                    legendgroup=year,
                    showlegend=True,
                    meta=dict(year=year),
                ),
                row=2,
                col=1,
            )
            trace_idx += 1

        # Scatter plots (row 2, col 2): 4 traces + 1 diagonal
        for model, col, color, sym in [
            ("MLP", "mlp_predicted", "blue", "circle"),
            ("Naive", "naive_predicted", "red", "diamond"),
        ]:
            for split, split_color in [("train", "blue"), ("test", "red")]:
                split_data = p[p["split"] == split]
                fig.add_trace(
                    go.Scatter(
                        x=split_data["target_rate"],
                        y=split_data[
                            model.lower() + "_predicted"
                            if model == "MLP"
                            else "naive_predicted"
                        ],
                        mode="markers",
                        name=f"{model} {split} ({year})",
                        marker=dict(
                            color="blue" if split == "train" else "red",
                            size=8,
                            opacity=0.6,
                            symbol=sym,
                        ),
                        visible=(year == list(all_results.keys())[0]),
                        legendgroup=year,
                        showlegend=True,
                        meta=dict(year=year),
                    ),
                    row=2,
                    col=2,
                )
                trace_idx += 1

        # Diagonal line for scatter
        max_val = p["target_rate"].max() * 1.1
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="black", width=1, dash="dash"),
                showlegend=False,
                visible=(year == list(all_results.keys())[0]),
                meta=dict(year=year),
            ),
            row=2,
            col=2,
        )
        trace_idx += 1

        year_trace_map[year] = list(range(year_start, trace_idx))

        # Build button: show only this year's traces
        visibility = [False] * trace_idx
        for idx in year_trace_map[year]:
            visibility[idx] = True

        button = dict(
            label=f"Test: {year}",
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": f"MLP Dengue Predictor - Test Year: {year}",
                    "annotations": [],
                },
            ],
        )
        buttons.append(button)

    # Add y5 axis for eggs overlay
    fig.update_layout(
        yaxis5=dict(
            title="Mean Egg Count",
            overlaying="y",
            side="right",
            showgrid=False,
            anchor="x",
        ),
        barmode="group",
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                showactive=True,
                buttons=buttons,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                pad=dict(r=10, t=10),
            )
        ],
        title=f"MLP Dengue Predictor - Test Year: {list(all_results.keys())[0]}",
        annotations=[
            dict(
                text="Select Test Year:",
                x=0,
                y=1.12,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14),
            )
        ],
        template="plotly_white",
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
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
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run src/mlp_bh_rate_predictor.py first.")
        return

    print("Loading all CV fold results...")
    all_results = load_all_results(results_dir)

    if not all_results:
        print(
            "Error: No fold results found. Please run the predictor first."
        )
        return

    # Load CV summary
    try:
        with open(results_dir / "cv_summary.json") as f:
            cv_summary = json.load(f)
    except FileNotFoundError:
        cv_summary = None

    print(f"Loaded {len(all_results)} folds: {list(all_results.keys())}")

    print("Creating interactive dashboard with year selector...")
    dashboard = create_interactive_dashboard(all_results, cv_summary)

    output_path = results_dir / "dashboard.html"
    dashboard.write_html(str(output_path))

    print(f"Dashboard saved to: {output_path}")

    # Print summary metrics for all folds
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    for year in EPIDEMY_YEARS:
        if year in all_results:
            m = all_results[year]["metrics"]
            naive_rmse = m["naive"]["test"]["rmse"]
            if naive_rmse > 0:
                improve = (1 - m["mlp"]["test"]["rmse"] / naive_rmse) * 100
                improve_str = f"{improve:+.1f}%"
            else:
                improve_str = "N/A (Naive perfect)"
            print(
                f"{year}: MLP RMSE={m['mlp']['test']['rmse']:.3f}, "
                f"Naive RMSE={naive_rmse:.3f}, "
                f"Improvement={improve_str}"
            )

    if cv_summary:
        print("-" * 60)
        print(
            f"Mean MLP Test RMSE: {cv_summary['mean_mlp_test_rmse']:.4f}"
        )
        print(
            f"Mean Naive Test RMSE: {cv_summary['mean_naive_test_rmse']:.4f}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()
