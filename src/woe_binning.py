"""Fit a WOE/IV BinningProcess on the train split and save it, plus an IV
summary table for the scorecard's feature documentation.

Uses optbinning.BinningProcess rather than a hand-rolled WOE binner: one
fit/transform API across all variables, monotonic_trend='auto' per
variable by default (needed for scorecard defensibility), and it produces
a consolidated IV summary for free. Fit only on the train split to avoid
leaking test-set information into bin boundaries. Must run on the
"binning input" CSV (raw NaNs intact), since optbinning carves its own
"Missing" bin per variable only when real NaNs are present.
"""

from pathlib import Path

import pandas as pd
from optbinning import BinningProcess

from src.data_prep import (
    BINNING_INPUT_CSV,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    TARGET_COL,
    make_train_test_split,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
BINNING_PROCESS_PATH = MODELS_DIR / "binning_process.joblib"
IV_SUMMARY_CSV = METRICS_DIR / "iv_summary.csv"

VARIABLE_NAMES = NUMERIC_COLS + CATEGORICAL_COLS


def iv_band(iv: float) -> str:
    if iv < 0.02:
        return "not useful"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


def fit_binning_process(X_train: pd.DataFrame, y_train: pd.Series) -> BinningProcess:
    binning_process = BinningProcess(
        variable_names=VARIABLE_NAMES,
        categorical_variables=CATEGORICAL_COLS,
    )
    binning_process.fit(X_train, y_train)
    return binning_process


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(BINNING_INPUT_CSV)
    train, _ = make_train_test_split(df)
    X_train = train[VARIABLE_NAMES]
    y_train = train[TARGET_COL]

    binning_process = fit_binning_process(X_train, y_train)
    binning_process.save(str(BINNING_PROCESS_PATH))
    print(f"Saved fitted BinningProcess to {BINNING_PROCESS_PATH}")

    summary = binning_process.summary()[["name", "dtype", "status", "n_bins", "iv"]]
    summary = summary.sort_values("iv", ascending=False).reset_index(drop=True)
    summary["iv_band"] = summary["iv"].apply(iv_band)
    summary.to_csv(IV_SUMMARY_CSV, index=False)
    print(f"Saved IV summary to {IV_SUMMARY_CSV}")
    print(summary)

    assert (summary["iv"] > 0).all(), "every variable should have positive IV"


if __name__ == "__main__":
    main()
