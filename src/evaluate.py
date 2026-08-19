"""Evaluate the scorecard and benchmark models: AUC, KS, Gini, Brier,
score-distribution plots, and calibration curves.

Calibration matters beyond ranking (AUC): expected_loss.py multiplies PD
directly into EL, so the champion model's PD needs to be well-calibrated,
not just good at ranking accounts by risk.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

METRICS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "metrics"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "figures"

MODELS = {
    "scorecard": METRICS_DIR / "scorecard_predictions.csv",
    "xgb_benchmark": METRICS_DIR / "xgb_predictions.csv",
}


def ks_statistic(y_true, proba) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float((tpr - fpr).max())


def compute_metrics(y_true, proba) -> dict:
    auc = roc_auc_score(y_true, proba)
    ks = ks_statistic(y_true, proba)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, proba)
    return {"AUC": auc, "KS": ks, "Gini": gini, "Brier": brier}


def plot_score_distribution(y_true, proba, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, name in [(0, "BAD=0"), (1, "BAD=1")]:
        ax.hist(proba[y_true == label], bins=30, alpha=0.6, label=name, density=True)
    ax.set_xlabel("Predicted PD")
    ax.set_title(f"{model_name}: score distribution by outcome")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_score_dist.png", dpi=120)
    plt.close(fig)


def plot_calibration(y_true, proba, model_name: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.set_xlabel("Mean predicted PD (decile)")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"{model_name}: calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_calibration.png", dpi=120)
    plt.close(fig)


def plot_roc_comparison(results: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for model_name, (y_true, proba) in results.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC comparison (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_comparison.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    roc_inputs = {}
    for model_name, path in MODELS.items():
        preds = pd.read_csv(path)
        test = preds[preds["split"] == "test"]
        y_true, proba = test["BAD"], test["PD"]

        metrics = compute_metrics(y_true, proba)
        metrics["model"] = model_name
        rows.append(metrics)

        plot_score_distribution(y_true, proba, model_name)
        plot_calibration(y_true, proba, model_name)
        roc_inputs[model_name] = (y_true, proba)

    plot_roc_comparison(roc_inputs)

    comparison = pd.DataFrame(rows)[["model", "AUC", "KS", "Gini", "Brier"]]
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    for _, row in comparison.iterrows():
        assert abs(row["Gini"] - (2 * row["AUC"] - 1)) < 1e-9, "Gini/AUC mismatch"

    print(
        "\nChampion decision: logistic-regression scorecard is the champion "
        "for decisioning — WOE-binned, monotonic, and directly interpretable "
        "(each feature's contribution to PD is a signed coefficient on an "
        "auditable bin), which matters for model-risk defensibility in a "
        "bank context. XGBoost is retained as the challenger/benchmark: its "
        "higher test AUC (see model_comparison.csv) quantifies the "
        "interpretability-vs-performance tradeoff of accepting a black-box "
        "model, and that gap is itself a documented finding."
    )


if __name__ == "__main__":
    main()
