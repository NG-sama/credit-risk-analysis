"""Compute PD -> EL via LGD/EAD assumptions, apply IFRS9 staging, and
aggregate Expected Loss by segment for a portfolio-level view.

Uses the champion model's PD (logistic scorecard — see evaluate.py's
documented champion decision) scored across the full dataset (train +
test), joined with cleaned/imputed bureau attributes for EAD/staging.

LIMITATION: HMEQ has no origination-date PD or time series, so true
IFRS9 "significant increase in credit risk" (SICR) detection is
impossible. Staging here is a documented cross-sectional proxy using
current PD level plus delinquency/derogatory bureau attributes as a
stand-in — it intentionally never uses the realized BAD label, which
would make the exercise circular/leaky.
"""

from pathlib import Path

import pandas as pd

from src.data_prep import BENCHMARK_INPUT_CSV

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
SCORECARD_PREDICTIONS_CSV = METRICS_DIR / "scorecard_predictions.csv"
SCORED_PORTFOLIO_CSV = PROCESSED_DIR / "scored_portfolio.csv"
EL_BY_SEGMENT_CSV = METRICS_DIR / "portfolio_el_by_segment.csv"

LGD_ASSUMPTION = 0.45  # flat LGD, standard convention for benchmarking

STAGE1_PD_MAX = 0.10
STAGE2_PD_MAX = 0.30
DELINQ_STAGE3_THRESHOLD = 2  # 2+ delinquent trade lines as objective-impairment proxy
DEROG_SICR_THRESHOLD = 1  # any derogatory record triggers Stage 2 (SICR proxy)

PD_BAND_EDGES = [0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0]
PD_BAND_LABELS = ["0-5%", "5-10%", "10-20%", "20-30%", "30-50%", "50-100%"]


def assign_stage(pd_value: float, delinq: float, derog: float) -> int:
    if pd_value >= STAGE2_PD_MAX or delinq >= DELINQ_STAGE3_THRESHOLD:
        return 3
    if pd_value >= STAGE1_PD_MAX or derog >= DEROG_SICR_THRESHOLD or delinq >= 1:
        return 2
    return 1


def build_scored_portfolio() -> pd.DataFrame:
    preds = pd.read_csv(SCORECARD_PREDICTIONS_CSV).set_index("account_id")
    features = pd.read_csv(BENCHMARK_INPUT_CSV)
    features.index.name = "account_id"

    df = features.join(preds[["split", "PD"]], how="inner")

    df["LGD"] = LGD_ASSUMPTION
    df["EAD"] = df["LOAN"]
    df["EL"] = df["PD"] * df["LGD"] * df["EAD"]

    df["STAGE"] = [
        assign_stage(pd_v, delinq, derog)
        for pd_v, delinq, derog in zip(df["PD"], df["DELINQ"], df["DEROG"])
    ]

    df["CREDIT_BAND"] = pd.cut(
        df["PD"], bins=PD_BAND_EDGES, labels=PD_BAND_LABELS, include_lowest=True
    )

    return df.reset_index()


def aggregate_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    segment_defs = {
        "REASON": df["REASON"],
        "JOB": df["JOB"],
        "PD_BAND": df["CREDIT_BAND"],
    }

    rows = []
    for segment_type, series in segment_defs.items():
        grouped = df.assign(_segment=series).groupby("_segment", observed=True)
        for segment_value, g in grouped:
            n = len(g)
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_value": segment_value,
                    "n_accounts": n,
                    "total_ead": g["EAD"].sum(),
                    "avg_pd": g["PD"].mean(),
                    "total_el": g["EL"].sum(),
                    "el_rate": g["EL"].sum() / g["EAD"].sum(),
                    "stage1_pct": (g["STAGE"] == 1).mean(),
                    "stage2_pct": (g["STAGE"] == 2).mean(),
                    "stage3_pct": (g["STAGE"] == 3).mean(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    scored = build_scored_portfolio()

    assert scored["EL"].notna().all() and (scored["EL"] >= 0).all()
    assert scored["PD"].between(0, 1).all()

    scored.to_csv(SCORED_PORTFOLIO_CSV, index=False)
    print(f"Saved scored portfolio to {SCORED_PORTFOLIO_CSV} ({len(scored)} rows)")

    stage_counts = scored["STAGE"].value_counts().sort_index()
    print("Stage distribution:")
    print(stage_counts)
    assert stage_counts.get(3, 0) < stage_counts.get(1, 0), (
        "expected far fewer Stage 3 (impaired) accounts than Stage 1 (performing)"
    )

    segment_summary = aggregate_by_segment(scored)
    assert segment_summary["segment_value"].notna().all()
    segment_summary.to_csv(EL_BY_SEGMENT_CSV, index=False)
    print(f"Saved segment EL summary to {EL_BY_SEGMENT_CSV}")
    print(segment_summary.to_string(index=False))


if __name__ == "__main__":
    main()
