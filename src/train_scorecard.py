"""Fit a logistic regression scorecard on WOE-transformed features.

Trains on the same stratified 70/30 split used by the XGBoost benchmark
(via data_prep.make_train_test_split) for a fair head-to-head comparison.
Scores the full dataset (train + test) and saves predictions with a
SPLIT column so expected_loss.py can compute portfolio-level EL while
downstream auditing can still filter to out-of-sample-only PDs.
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from optbinning import BinningProcess
from sklearn.linear_model import LogisticRegression

from src.data_prep import BINNING_INPUT_CSV, TARGET_COL, make_train_test_split
from src.woe_binning import BINNING_PROCESS_PATH, VARIABLE_NAMES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
SCORECARD_MODEL_PATH = MODELS_DIR / "scorecard_logreg.joblib"
SCORECARD_PREDICTIONS_CSV = METRICS_DIR / "scorecard_predictions.csv"

# Conventional PDO (points-to-double-the-odds) scaling anchors.
TARGET_SCORE = 600
TARGET_ODDS = 50  # odds of good:bad at the target score
PDO = 20
FACTOR = PDO / math.log(2)
OFFSET = TARGET_SCORE - FACTOR * math.log(TARGET_ODDS)


def pd_to_points(pd_values: np.ndarray) -> np.ndarray:
    """Convert PD (probability of default) to a bank-style credit score."""
    pd_clipped = np.clip(pd_values, 1e-6, 1 - 1e-6)
    odds_good = (1 - pd_clipped) / pd_clipped
    return OFFSET + FACTOR * np.log(odds_good)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BINNING_INPUT_CSV)
    train, test = make_train_test_split(df)

    binning_process: BinningProcess = BinningProcess.load(str(BINNING_PROCESS_PATH))

    X_train = binning_process.transform(train[VARIABLE_NAMES], metric="woe")
    X_test = binning_process.transform(test[VARIABLE_NAMES], metric="woe")
    y_train = train[TARGET_COL]

    # No class_weight balancing: HMEQ's ~20% default rate is mild imbalance,
    # and balancing distorts predicted PDs away from the true base rate
    # (verified empirically — mean PD came out ~35% vs. actual ~20%), which
    # matters because expected_loss.py multiplies PD directly into EL.
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, SCORECARD_MODEL_PATH)
    print(f"Saved scorecard model to {SCORECARD_MODEL_PATH}")

    # Score the full dataset (train + test) for downstream EL calc.
    X_full = binning_process.transform(df[VARIABLE_NAMES], metric="woe")
    proba_full = model.predict_proba(X_full)[:, 1]

    split = pd.Series("test", index=df.index)
    split.loc[train.index] = "train"

    predictions = pd.DataFrame(
        {
            "account_id": df.index,
            "split": split,
            "BAD": df[TARGET_COL],
            "PD": proba_full,
            "SCORE_POINTS": pd_to_points(proba_full),
        }
    )
    predictions.to_csv(SCORECARD_PREDICTIONS_CSV, index=False)
    print(f"Saved predictions to {SCORECARD_PREDICTIONS_CSV}")

    from sklearn.metrics import roc_auc_score

    test_auc = roc_auc_score(
        test[TARGET_COL], predictions.loc[test.index, "PD"]
    )
    print(f"Scorecard test AUC: {test_auc:.4f}")
    assert test_auc > 0.65, f"scorecard test AUC {test_auc:.4f} below sanity floor 0.65"


if __name__ == "__main__":
    main()
