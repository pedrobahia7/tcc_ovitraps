"""MLP model to predict city-wide dengue cases per 1000 population rate.

Predicts the current biweek's rate using:
- Last 3 biweek rates (cases_per_1000)
- Last 5 biweek egg counts (city-wide mean)

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
import sys

sys.path.append("utils")
import project_utils
import ipdb 
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
    """Create lag features for the target column.

    Assumes df is indexed by a complete biweek sequence (no gaps),
    so shift(1) correctly yields the previous biweek.
    """
    for lag in range(1, lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


def prepare_features(
    dengue_df: pd.DataFrame, ovitraps_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge data sources and create feature matrix with lags.

    Reindexes to a complete biweek sequence before creating lags,
    so missing biweeks (common in ovitraps data) do not corrupt
    the lag values via positional shift.
    """
    # Merge dengue and ovitraps on biweek
    df = dengue_df.merge(ovitraps_df, on="biweek", how="outer")

    # --- Fix gaps: reindex to complete biweek sequence ---
    all_biweeks = project_utils.generate_all_biweeks(
        df["biweek"].min(), df["biweek"].max()
    )
    df = df.set_index("biweek").reindex(all_biweeks).reset_index()
    df.rename(columns={"index": "biweek"}, inplace=True)

    # Sort by biweek so shift() is chronological
    df = df.sort_values("biweek").reset_index(drop=True)
    assert df.index.is_monotonic_increasing, "Date range must be sorted"

    # Create lag features for dengue rate (3 lags)
    df = create_lag_features(df, "cases_per_1000", lags=3)

    # Create lag features for ovitraps eggs (5 lags)
    df = create_lag_features(df, "mean_eggs", lags=5)

    # Verify biweek sequence is complete after reset_index
    assert df["biweek"].tolist() == all_biweeks, (
        "Biweek sequence should be complete after reindex"
    )

    # Verify lag features on rows where all values are non-NaN
    # (reindexing introduces NaN for missing biweeks; skip those)
    valid_mask = (
        df["cases_per_1000"].notna()
        & df["mean_eggs"].notna()
        & df["cases_per_1000_lag1"].notna()
        & df["cases_per_1000_lag2"].notna()
        & df["cases_per_1000_lag3"].notna()
        & df["mean_eggs_lag1"].notna()
        & df["mean_eggs_lag2"].notna()
        & df["mean_eggs_lag3"].notna()
        & df["mean_eggs_lag4"].notna()
        & df["mean_eggs_lag5"].notna()
    )
    valid_idx = df[valid_mask].index

    np.random.seed(42)
    check_idx = np.random.choice(valid_idx[5:], size=1, replace=False)[0]
    row = df.loc[check_idx]

    for lag in range(1, 4):
        assert np.isclose(
            row[f"cases_per_1000_lag{lag}"],
            df.loc[check_idx - lag, "cases_per_1000"],
        ), f"Dengue lag{lag} mismatch at index {check_idx}"

    for lag in range(1, 6):
        assert np.isclose(
            row[f"mean_eggs_lag{lag}"],
            df.loc[check_idx - lag, "mean_eggs"],
        ), f"Eggs lag{lag} mismatch at index {check_idx}"

    # Remove current biweek eggs (not available as a feature)
    df = df.drop(columns=["mean_eggs"]).copy()

    # Drop rows with NaN lags (first 5 rows due to eggs_lag5 + any gaps)
    df = df.dropna().copy()

    # Target is the current biweek dengue rate (uses only past info via lags)
    df = df.rename(columns={"cases_per_1000": "target_rate"})

    assert df.isna().sum().sum() == 0, (
        "DataFrame should not contain NaN values"
    )
    # Verify target_rate matches original dengue rate for each biweek
    merged_check = df.merge(
        dengue_df[["biweek", "cases_per_1000"]],
        on="biweek",
        how="left",
    )
    assert np.allclose(
        df["target_rate"].values,
        merged_check["cases_per_1000"].values,
        equal_nan=True,
    ), "target_rate should match original cases_per_1000 for each biweek"
    assert "cases_per_1000" not in df.columns, (
        "cases_per_1000 should be renamed to target_rate"
    )
    assert "mean_eggs" not in df.columns, "mean_eggs should be removed"

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
    
    # Change to 0 any negative predictions (not meaningful for rates)
    y_train_pred = np.where(y_train_pred < 0, 0, y_train_pred)
    y_test_pred = np.where(y_test_pred < 0, 0, y_test_pred)

    # Check length of predictions matches true values
    assert len(y_train_pred) == len(y_train), "Train predictions length mismatch"
    assert len(y_test_pred) == len(y_test), "Test predictions length mismatch"

    # Check for NaN values in predictions
    assert not np.isnan(y_train_pred).any(), "NaN values in train predictions"
    assert not np.isnan(y_test_pred).any(), "NaN values in test predictions"

    # Check for reasonable prediction ranges (non-negative rates)
    assert (y_train_pred >= 0).all(), "Negative values in train predictions"
    assert (y_test_pred >= 0).all(), "Negative values in test predictions"

    # Check for variance in predictions (not all the same value)
    assert np.var(y_train_pred) > 0, "No variance in train predictions"
    assert np.var(y_test_pred) > 0, "No variance in test predictions"

    # Check for overfitting (train RMSE should be less than test RMSE)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    assert (train_rmse < test_rmse * 1.5), f"Possible overfitting: train RMSE {train_rmse:.4f}, test RMSE {test_rmse:.4f}"
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


def run_fold(
    epidemic_df: pd.DataFrame, test_year: str, feature_cols: list
) -> dict:
    """Run a single CV fold with given test year."""
    print(f"\n{'=' * 50}")
    print(f"Fold: Test on {test_year}")
    print("=" * 50)

    train_df, test_df = split_by_year(epidemic_df, test_year=test_year)
    print(
        f"  Train: {len(train_df)} samples ({train_df['year'].nunique()} years)"
    )
    print(
        f"  Test: {len(test_df)} samples ({test_df['year'].nunique()} year)"
    )

    print("Training MLP model...")
    mlp, scaler, mlp_train_pred, mlp_test_pred, y_train, y_test = (
        train_mlp(train_df, test_df)
    )

    print("Computing naive baseline...")
    naive_train_pred, naive_test_pred = naive_predictor(train_df, test_df)

    metrics = {
        "test_year": test_year,
        "mlp": {
            "train": compute_metrics(y_train, mlp_train_pred),
            "test": compute_metrics(y_test, mlp_test_pred),
        },
        "naive": {
            "train": compute_metrics(y_train, naive_train_pred),
            "test": compute_metrics(y_test, naive_test_pred),
        },
    }

    print(f"  MLP Test RMSE: {metrics['mlp']['test']['rmse']:.4f}")
    print(f"  Naive Test RMSE: {metrics['naive']['test']['rmse']:.4f}")

    # Compute feature importance
    X_test_scaled = scaler.transform(test_df[feature_cols].values)
    importance_df = compute_feature_importance(
        mlp, X_test_scaled, y_test, feature_cols
    )
    return {
        "metrics": metrics,
        "train_df": train_df,
        "test_df": test_df,
        "mlp_train_pred": mlp_train_pred,
        "mlp_test_pred": mlp_test_pred,
        "naive_train_pred": naive_train_pred,
        "naive_test_pred": naive_test_pred,
        "mlp": mlp,
        "scaler": scaler,
        "importance_df": importance_df,
    }


def main() -> None:
    """Run 4-fold leave-one-year-out cross validation."""
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
    print(f"  Years: {list(epidemic_df['year'].unique())}")

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

    # Run 4-fold CV
    all_results = {}
    for test_year in EPIDEMY_YEARS:
        fold_results = run_fold(epidemic_df, test_year, feature_cols)
        all_results[test_year] = fold_results

        # Save fold results
        output_dir = Path(
            f"results/mlp_bh_rate_predictor/fold_{test_year}"
        )
        save_results(
            output_dir,
            fold_results["train_df"],
            fold_results["test_df"],
            fold_results["mlp_train_pred"],
            fold_results["mlp_test_pred"],
            fold_results["naive_train_pred"],
            fold_results["naive_test_pred"],
            fold_results["mlp"],
            fold_results["scaler"],
            fold_results["metrics"],
            fold_results["importance_df"],
        )

    # Save aggregated CV results
    cv_metrics = {
        year: results["metrics"] for year, results in all_results.items()
    }
    cv_summary = {
        "folds": cv_metrics,
        "mean_mlp_test_rmse": np.mean(
            [m["mlp"]["test"]["rmse"] for m in cv_metrics.values()]
        ),
        "mean_naive_test_rmse": np.mean(
            [m["naive"]["test"]["rmse"] for m in cv_metrics.values()]
        ),
    }

    summary_dir = Path("results/mlp_bh_rate_predictor")
    summary_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_dir / "cv_summary.json", "w") as f:
        json.dump(cv_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Mean MLP Test RMSE: {cv_summary['mean_mlp_test_rmse']:.4f}")
    print(
        f"Mean Naive Test RMSE: {cv_summary['mean_naive_test_rmse']:.4f}"
    )
    print("=" * 60)
    print("\nDone! Run the dashboard script to visualize results:")
    print("  python scripts/mlp_bh_dashboard.py")


if __name__ == "__main__":
    main()
