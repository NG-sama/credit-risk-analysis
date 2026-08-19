"""Clean HMEQ and produce two processed datasets:

- hmeq_binning_input.csv: cleaned + missing-flags, NaNs left intact for
  optbinning to carve its own "Missing" bin per variable.
- hmeq_benchmark_input.csv: cleaned + missing-flags + fully imputed, for
  the XGBoost benchmark which can't natively use a WOE "Missing" bin.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_CSV = Path(__file__).resolve().parent.parent / "data" / "raw" / "hmeq.csv"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BINNING_INPUT_CSV = PROCESSED_DIR / "hmeq_binning_input.csv"
BENCHMARK_INPUT_CSV = PROCESSED_DIR / "hmeq_benchmark_input.csv"

TARGET_COL = "BAD"
NUMERIC_COLS = [
    "LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ",
    "CLAGE", "NINQ", "CLNO", "DEBTINC",
]
CATEGORICAL_COLS = ["REASON", "JOB"]


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip()
    return df


def add_missing_flags(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if df[col].isna().any():
            df[f"{col}_MISSING"] = df[col].isna().astype(int)
    return df


def make_binning_input(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned + flagged, numeric NaNs left intact for optbinning."""
    df = basic_clean(df)
    df = add_missing_flags(df, NUMERIC_COLS)
    return df


def make_benchmark_input(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaned + flagged + fully imputed, for the tree benchmark."""
    df = basic_clean(df)
    df = add_missing_flags(df, NUMERIC_COLS)
    for col in NUMERIC_COLS:
        median = df[col].median()
        df[col] = df[col].fillna(median)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna("MISSING")
    return df


def make_train_test_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.30,
    random_state: int = 42,
):
    """Shared stratified split so scorecard and benchmark train/test on
    identical rows for a fair head-to-head comparison."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_CSV)

    binning_input = make_binning_input(raw)
    binning_input.to_csv(BINNING_INPUT_CSV, index=False)
    print(f"Wrote {BINNING_INPUT_CSV} ({len(binning_input)} rows)")

    benchmark_input = make_benchmark_input(raw)
    assert benchmark_input[NUMERIC_COLS].isna().sum().sum() == 0, (
        "benchmark input should have zero remaining NaNs in numeric cols"
    )
    benchmark_input.to_csv(BENCHMARK_INPUT_CSV, index=False)
    print(f"Wrote {BENCHMARK_INPUT_CSV} ({len(benchmark_input)} rows)")


if __name__ == "__main__":
    main()
