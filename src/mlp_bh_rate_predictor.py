"""MLP model to predict city-wide dengue cases per 1000 population rate.

Predicts the next biweek's rate using:
- Last 3 biweek rates (cases_per_1000)
- Last 3 biweek egg counts (city-wide mean)

Trained exclusively on epidemic years with naive baseline comparison.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Epidemic years defined in project_utils.EPIDEMY_YEARS
EPIDEMY_YEARS = ["2012_13", "2015_16", "2018_19", "2023_24"]


def load_dengue_citywide() -> pd.DataFrame:
    """Load and aggregate dengue data to city-wide biweekly rates."""
    dengue_path = Path("data/processed/dengue_per_capita.csv")
    df = pd.read_csv(dengue_path)

    # Aggregate to city-wide: sum cases and population, compute rate
    citywide = (
        df.groupby("biweek")
        .agg({"case_count": "sum", "population": "sum"})
        .reset_index()
    )

    citywide["cases_per_1000"] = (
        citywide["case_count"] / citywide["population"] * 1000
    ).fillna(0)

    return citywide[["biweek", "cases_per_1000"]]


def load_ovitraps_citywide() -> pd.DataFrame:
    """Load and aggregate ovitraps data to city-wide mean egg counts by biweek."""
    ovitraps_path = Path("data/processed/ovitraps_data.csv")
    df = pd.read_csv(ovitraps_path, low_memory=False)

    # Group by biweek and compute city-wide mean egg count
    citywide = df.groupby("biweek")["novos"].mean().reset_index()
    citywide.rename(columns={"novos": "mean_eggs"}, inplace=True)

    return citywide


def create_lag_features(
    df: pd.DataFrame, target_col: str, lags: int = 3
) -> pd.DataFrame:
    """Create lag features for the target column."""
    df = df.copy().sort_values("biweek")

    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)

    return df


def prepare_features(
    dengue_df: pd.DataFrame, ovitraps_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge data sources and create feature matrix with lags."""
    # Merge dengue and ovitraps on biweek
    df = dengue_df.merge(ovitraps_df, on="biweek", how="inner")

    # Create lag features for both rate and eggs
    df = create_lag_features(df, "cases_per_1000", lags=3)
    df = create_lag_features(
        df, "mean_eggs", lags=5
    )  # Extended to 5 lags for eggs

    # Drop rows with NaN lags (first 5 rows due to eggs_lag5)
    df = df.dropna().copy()

    # Create target: next biweek rate (shift -1)
    df = df.sort_values("biweek")
    df["target_rate"] = df["cases_per_1000"].shift(-1)

    # Drop last row (no target available)
    df = df.dropna(subset=["target_rate"]).copy()

    return df


def split_by_year(df: pd.DataFrame, test_year: str = "2023_24") -> tuple:
    """Split data into train/test based on epidemic year."""
    # Extract year from biweek (format: YYYY_YYWNN)
    df["year"] = df["biweek"].str.extract(r"(\d{4}_\d{2})")

    train_df = df[df["year"] != test_year].copy()
    test_df = df[df["year"] == test_year].copy()

    return train_df, test_df


def get_epidemic_years(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to epidemic years only."""
    df["year"] = df["biweek"].str.extract(r"(\d{4}_\d{2})")
    return df[df["year"].isin(EPIDEMY_YEARS)].copy()


def naive_predictor(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Naive persistence model: predict last observed value."""
    # Naive prediction: use the most recent rate (lag1 = current rate)
    y_train_naive = train_df["cases_per_1000_lag1"].values
    y_test_naive = test_df["cases_per_1000_lag1"].values

    return y_train_naive, y_test_naive


def train_mlp(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Train MLP model and return predictions with metrics."""
    feature_cols = [
        "cases_per_1000_lag1",
        "cases_per_1000_lag2",
        "cases_per_1000_lag3",
        "mean_eggs_lag1",
        "mean_eggs_lag2",
        "mean_eggs_lag3",
        "mean_eggs_lag4",
        "mean_eggs_lag5",
    ]

    X_train = train_df[feature_cols].values
    y_train = train_df["target_rate"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_rate"].values

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(50, 25, 10),
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=5000,
        random_state=42,
        verbose=True,
    )

    mlp.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)

    return mlp, scaler, y_train_pred, y_test_pred, y_train, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100,
    }


def compute_feature_importance(
    mlp, X_test: np.ndarray, y_test: np.ndarray, feature_names: list
) -> pd.DataFrame:
    """Compute permutation feature importance."""
    result = permutation_importance(
        mlp, X_test, y_test, n_repeats=10, random_state=42
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    return importance_df


def save_results(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlp_train_pred: np.ndarray,
    mlp_test_pred: np.ndarray,
    naive_train_pred: np.ndarray,
    naive_test_pred: np.ndarray,
    mlp,
    scaler,
    metrics: dict,
    importance_df: pd.DataFrame,
) -> None:
    """Save all results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    train_out = train_df.copy()
    train_out["mlp_predicted"] = mlp_train_pred
    train_out["naive_predicted"] = naive_train_pred
    train_out["split"] = "train"

    test_out = test_df.copy()
    test_out["mlp_predicted"] = mlp_test_pred
    test_out["naive_predicted"] = naive_test_pred
    test_out["split"] = "test"

    predictions_df = pd.concat([train_out, test_out], ignore_index=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    # Save model
    joblib.dump(
        {"model": mlp, "scaler": scaler}, output_dir / "model.joblib"
    )

    # Save feature importance
    importance_df.to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    print(f"Results saved to {output_dir}")


def main() -> None:
    """Run the MLP prediction pipeline."""
    print("Loading dengue data...")
    dengue_df = load_dengue_citywide()
    print(f"  {len(dengue_df)} biweeks loaded")

    print("Loading ovitraps data...")
    ovitraps_df = load_ovitraps_citywide()
    print(f"  {len(ovitraps_df)} biweeks loaded")

    print("Preparing features with lags...")
    features_df = prepare_features(dengue_df, ovitraps_df)
    print(f"  {len(features_df)} samples after creating lags")

    print("Filtering to epidemic years...")
    epidemic_df = get_epidemic_years(features_df)
    print(f"  {len(epidemic_df)} samples in epidemic years")
    print(f"  Years: {epidemic_df['year'].unique()}")

    print("Splitting train/test...")
    train_df, test_df = split_by_year(epidemic_df, test_year="2023_24")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    print("Training MLP model...")
    mlp, scaler, mlp_train_pred, mlp_test_pred, y_train, y_test = (
        train_mlp(train_df, test_df)
    )

    print("Computing naive baseline...")
    naive_train_pred, naive_test_pred = naive_predictor(train_df, test_df)

    print("Computing metrics...")
    metrics = {
        "mlp": {
            "train": compute_metrics(y_train, mlp_train_pred),
            "test": compute_metrics(y_test, mlp_test_pred),
        },
        "naive": {
            "train": compute_metrics(y_train, naive_train_pred),
            "test": compute_metrics(y_test, naive_test_pred),
        },
    }

    feature_cols = [
        "cases_per_1000_lag1",
        "cases_per_1000_lag2",
        "cases_per_1000_lag3",
        "mean_eggs_lag1",
        "mean_eggs_lag2",
        "mean_eggs_lag3",
        "mean_eggs_lag4",
        "mean_eggs_lag5",
    ]
    X_test_scaled = scaler.transform(test_df[feature_cols].values)

    print("Computing feature importance...")
    importance_df = compute_feature_importance(
        mlp, X_test_scaled, y_test, feature_cols
    )

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"MLP Test RMSE: {metrics['mlp']['test']['rmse']:.4f}")
    print(f"Naive Test RMSE: {metrics['naive']['test']['rmse']:.4f}")
    print(f"MLP Test MAE: {metrics['mlp']['test']['mae']:.4f}")
    print(f"Naive Test MAE: {metrics['naive']['test']['mae']:.4f}")
    print(f"MLP Test R²: {metrics['mlp']['test']['r2']:.4f}")
    print(f"Naive Test R²: {metrics['naive']['test']['r2']:.4f}")
    print("=" * 50)

    # Save results
    output_dir = Path("results/mlp_bh_rate_predictor")
    save_results(
        output_dir,
        train_df,
        test_df,
        mlp_train_pred,
        mlp_test_pred,
        naive_train_pred,
        naive_test_pred,
        mlp,
        scaler,
        metrics,
        importance_df,
    )

    print("\nDone! Run the dashboard script to visualize results:")
    print("  python scripts/mlp_bh_dashboard.py")


if __name__ == "__main__":
    main()
