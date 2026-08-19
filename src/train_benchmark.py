"""Fit an XGBoost benchmark on raw imputed + missing-flagged features.

Trained on the "benchmark input" (imputed, not WOE-transformed) — WOE
transforming for a tree model would be atypical and defeats the "raw
features, tree learns its own splits" comparison story. Uses the same
stratified split as the scorecard for a fair head-to-head comparison.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data_prep import (
    BENCHMARK_INPUT_CSV,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    make_train_test_split,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
BENCHMARK_MODEL_PATH = MODELS_DIR / "xgb_benchmark.joblib"
BENCHMARK_PREDICTIONS_CSV = METRICS_DIR / "xgb_predictions.csv"

MISSING_FLAG_COLS = [f"{c}_MISSING" for c in NUMERIC_COLS]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    present_flags = [c for c in MISSING_FLAG_COLS if c in df.columns]
    feature_cols = NUMERIC_COLS + present_flags + CATEGORICAL_COLS
    X = df[feature_cols].copy()
    for c in present_flags:
        X[c] = X[c].astype(int)
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype("category")
    return X


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BENCHMARK_INPUT_CSV)
    train, test = make_train_test_split(df)

    X_train = prepare_features(train)
    y_train = train[TARGET_COL]
    X_test = prepare_features(test)

    # No scale_pos_weight: HMEQ's ~20% default rate is mild imbalance, and
    # weighting distorts predicted PDs away from the true base rate
    # (verified empirically — mean PD came out ~28% vs. actual ~20%), which
    # matters because expected_loss.py multiplies PD directly into EL.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="auc",
        random_state=42,
        enable_categorical=True,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, BENCHMARK_MODEL_PATH)
    print(f"Saved benchmark model to {BENCHMARK_MODEL_PATH}")

    X_full = prepare_features(df)
    proba_full = model.predict_proba(X_full)[:, 1]

    split = pd.Series("test", index=df.index)
    split.loc[train.index] = "train"

    predictions = pd.DataFrame(
        {
            "account_id": df.index,
            "split": split,
            "BAD": df[TARGET_COL],
            "PD": proba_full,
        }
    )
    predictions.to_csv(BENCHMARK_PREDICTIONS_CSV, index=False)
    print(f"Saved predictions to {BENCHMARK_PREDICTIONS_CSV}")

    test_auc = roc_auc_score(test[TARGET_COL], predictions.loc[test.index, "PD"])
    print(f"Benchmark test AUC: {test_auc:.4f}")
    assert test_auc > 0.70, f"benchmark test AUC {test_auc:.4f} below sanity floor 0.70"


if __name__ == "__main__":
    main()
