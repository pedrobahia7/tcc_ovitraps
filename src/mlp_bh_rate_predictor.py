"""MLP model to predict city-wide dengue cases per 1000 population rate.

Predicts the current biweek's rate using:
- Last 3 biweek rates (cases_per_1000)
- Last 5 biweek egg counts (city-wide mean)

Trained exclusively on epidemic years with naive baseline comparison.
"""

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
# Epidemic years defined in project_utils.EPIDEMY_YEARS
EPIDEMY_YEARS = ["2012_13", "2015_16", "2018_19", "2023_24"]


def load_dengue_citywide() -> pd.DataFrame:
    """Load city-wide dengue biweekly rate (all years, no sector filter)."""
    path = Path(
        "data/dvc/add_population_info/dengue_citywide_per_capita.csv"
    )
    df = pd.read_csv(path)
    return df[["biweek", "cases_per_1000"]]


def load_ovitraps_citywide() -> pd.DataFrame:
    """Load and aggregate ovitraps data to city-wide mean egg counts by biweek."""
    ovitraps_path = Path("data/dvc/add_population_info/ovitraps_data.csv")
    df = pd.read_csv(ovitraps_path, low_memory=False)

    # Group by biweek and compute city-wide mean egg count
    citywide = df.groupby("biweek")["novos"].mean().reset_index()
    citywide.rename(columns={"novos": "mean_eggs"}, inplace=True)

    return citywide


def extract_week_number(biweek: str) -> int:
    """Extract numeric week index from biweek string like '2006_07W32'."""
    return int(biweek.split("W")[1])


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sin/cos cyclical encoding of biweek position within year.

    Dengue is strongly seasonal; without this the model has no sense
    of where in the season each sample falls.
    """
    week_num = df["biweek"].apply(extract_week_number)
    df = df.copy()
    df["week_sin"] = np.sin(2 * np.pi * week_num / 52)
    df["week_cos"] = np.cos(2 * np.pi * week_num / 52)
    return df


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

    # Create lags 1-4 for eggs then keep only 3+4 (biologically relevant
    # incubation window); drops fewer rows than keeping all 5 lags.
    df = create_lag_features(df, "mean_eggs", lags=4)
    df = df.drop(columns=["mean_eggs_lag1", "mean_eggs_lag2"])

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
        & df["mean_eggs_lag3"].notna()
        & df["mean_eggs_lag4"].notna()
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

    for lag in (3, 4):
        assert np.isclose(
            row[f"mean_eggs_lag{lag}"],
            df.loc[check_idx - lag, "mean_eggs"],
        ), f"Eggs lag{lag} mismatch at index {check_idx}"

    # Add cyclical seasonality before dropping target cols
    df = add_seasonality_features(df)

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


def train_mlp(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple:
    """Train MLP via grid search; return predictions with metrics."""
    feature_cols = [
        "cases_per_1000_lag1",
        "cases_per_1000_lag2",
        "cases_per_1000_lag3",
        "mean_eggs_lag3",
        "mean_eggs_lag4",
        "week_sin",
        "week_cos",
    ]
    X_train = train_df[feature_cols].values
    y_train = train_df["target_rate"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["target_rate"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        "hidden_layer_sizes": [(16,), (32,), (32, 16), (16, 8)],
        "alpha": [0.001, 0.01, 0.1],
    }
    base = MLPRegressor(
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=5000,
        random_state=42,
    )
    tscv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        base,
        param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=True,
    )
    t0 = time.perf_counter()
    grid.fit(X_train_scaled, y_train)
    train_time_s = time.perf_counter() - t0
    mlp = grid.best_estimator_
    print(f"  Best params: {grid.best_params_}")
    print(f"  Train time: {train_time_s:.1f}s")

    y_train_pred = np.maximum(mlp.predict(X_train_scaled), 0)
    y_test_pred = np.maximum(mlp.predict(X_test_scaled), 0)

    assert len(y_train_pred) == len(y_train)
    assert len(y_test_pred) == len(y_test)
    assert not np.isnan(y_train_pred).any()
    assert not np.isnan(y_test_pred).any()
    assert (y_train_pred >= 0).all()
    assert (y_test_pred >= 0).all()
    assert np.var(y_train_pred) > 0
    assert np.var(y_test_pred) > 0

    return mlp, scaler, y_train_pred, y_test_pred, y_train, y_test, train_time_s


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
    mlp, scaler, mlp_train_pred, mlp_test_pred, y_train, y_test, train_time_s = (
        train_mlp(train_df, test_df)
    )

    print("Computing naive baseline...")
    naive_train_pred, naive_test_pred = naive_predictor(train_df, test_df)

    metrics = {
        "test_year": test_year,
        "mlp": {
            "train": compute_metrics(y_train, mlp_train_pred),
            "test": compute_metrics(y_test, mlp_test_pred),
            "train_time_s": round(train_time_s, 2),
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
        "mean_eggs_lag3",
        "mean_eggs_lag4",
        "week_sin",
        "week_cos",
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
