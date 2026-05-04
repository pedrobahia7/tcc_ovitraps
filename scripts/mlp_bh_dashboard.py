"""Interactive dashboard for MLP vs Naive predictor comparison.

Visualizes prediction results with time series, error metrics, and scatter plots.
Saves as HTML for easy sharing.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(results_dir: Path) -> tuple:
    """Load metrics and predictions from results directory."""
    with open(results_dir / "metrics.json") as f:
        metrics = json.load(f)

    predictions = pd.read_csv(results_dir / "predictions.csv")

    return metrics, predictions


def create_time_series_plot(predictions: pd.DataFrame) -> go.Figure:
    """Create time series plot of actual vs predicted rates."""
    # Sort by biweek for proper ordering
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

    # Highlight train/test split with shapes (strings can't use add_vline)
    train_data = predictions[predictions["split"] == "train"]
    test_data = predictions[predictions["split"] == "test"]

    if len(train_data) > 0 and len(test_data) > 0:
        # Add annotation at split point
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
        title="Dengue Cases per 1000: Actual vs Predicted",
        xaxis_title="Biweek",
        yaxis_title="Cases per 1000 Population",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_metrics_comparison(metrics: dict) -> go.Figure:
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

    fig.update_layout(
        title="Error Metrics: MLP vs Naive Baseline",
        yaxis_title="Error Value",
        barmode="group",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_residual_plot(predictions: pd.DataFrame) -> go.Figure:
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
        title="MLP Prediction Residuals (Actual - Predicted)",
        xaxis_title="Biweek",
        yaxis_title="Residual",
        template="plotly_white",
        showlegend=True,
    )

    return fig


def create_scatter_plot(predictions: pd.DataFrame) -> go.Figure:
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

    # Add diagonal reference line to both subplots
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
        title="Predicted vs Actual Rates",
        template="plotly_white",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def create_dashboard(
    metrics: dict, predictions: pd.DataFrame
) -> go.Figure:
    """Create comprehensive dashboard with all plots."""
    # Create individual figures
    time_series = create_time_series_plot(predictions)
    metrics_comp = create_metrics_comparison(metrics)
    residual = create_residual_plot(predictions)
    scatter = create_scatter_plot(predictions)

    # Combine into subplots dashboard
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Time Series: Actual vs Predicted",
            "Error Metrics Comparison",
            "MLP Residuals Over Time",
            "Predicted vs Actual Scatter",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Add time series traces (row 1, col 1)
    for trace in time_series.data:
        fig.add_trace(trace, row=1, col=1)

    # Add metrics traces (row 1, col 2)
    for trace in metrics_comp.data:
        fig.add_trace(trace, row=1, col=2)

    # Add residual traces (row 2, col 1)
    for trace in residual.data:
        fig.add_trace(trace, row=2, col=1)

    # Add scatter traces (row 2, col 2)
    for trace in scatter.data:
        fig.add_trace(trace, row=2, col=2)

    # Add diagonal line to scatter plot
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
        title="MLP Dengue Rate Predictor: Dashboard",
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
    """Generate dashboard HTML file."""
    results_dir = Path("results/mlp_bh_rate_predictor")

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run src/mlp_bh_rate_predictor.py first.")
        return

    print("Loading results...")
    metrics, predictions = load_results(results_dir)

    print("Creating dashboard...")
    dashboard = create_dashboard(metrics, predictions)

    output_path = results_dir / "dashboard.html"
    dashboard.write_html(str(output_path))

    print(f"Dashboard saved to: {output_path}")

    # Print summary metrics
    print("\n" + "=" * 50)
    print("SUMMARY METRICS")
    print("=" * 50)
    print(f"MLP Test RMSE: {metrics['mlp']['test']['rmse']:.4f}")
    print(f"Naive Test RMSE: {metrics['naive']['test']['rmse']:.4f}")
    print(
        f"Improvement: {(1 - metrics['mlp']['test']['rmse'] / metrics['naive']['test']['rmse']) * 100:.1f}%"
    )
    print(f"MLP Test R²: {metrics['mlp']['test']['r2']:.4f}")
    print(f"Naive Test R²: {metrics['naive']['test']['r2']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
