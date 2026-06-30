#!/usr/bin/env python3
"""Plot a single weighted ROC and PR curve for DeepHP ensemble voting.

This script uses already-generated artifacts:
- {run_id}_f{fold}_predictions_corrected.json for fold probabilities and labels
- {run_id}_ensemble_weights_{strategy}.json for the ensemble weights

It does not rerun model inference or rebuild the ensemble outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve


def load_fold_predictions(run_id: str, input_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    fold_data: dict[int, dict[str, np.ndarray]] = {}

    for fold_idx in range(5):
        pred_file = input_dir / f"{run_id}_f{fold_idx}_predictions_corrected.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"Missing corrected prediction file: {pred_file}")

        with pred_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        fold_data[fold_idx] = {
            "labels": np.asarray(data["labels"], dtype=int),
            "probabilities": np.asarray(data["probabilities"], dtype=float),
        }

    return fold_data


def load_fold_weights(run_id: str, strategy: str, input_dir: Path) -> dict[int, float]:
    weights_file = input_dir / f"{run_id}_ensemble_weights_{strategy}.json"
    if not weights_file.exists():
        raise FileNotFoundError(f"Missing ensemble weights file: {weights_file}")

    with weights_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    fold_weights = {int(key): float(value) for key, value in data["fold_weights"].items()}
    return fold_weights


def build_weighted_samples(
    fold_data: dict[int, dict[str, np.ndarray]], fold_weights: dict[int, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool fold predictions into one weighted sample set for curve generation."""
    label_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    sample_weight_parts: list[np.ndarray] = []

    for fold_idx in sorted(fold_data):
        labels = fold_data[fold_idx]["labels"]
        probabilities = fold_data[fold_idx]["probabilities"]
        fold_weight = float(fold_weights[fold_idx])

        if len(labels) != len(probabilities):
            raise ValueError(
                f"Fold {fold_idx} has {len(labels)} labels and {len(probabilities)} probabilities"
            )

        if len(labels) == 0:
            continue

        per_sample_weight = fold_weight / len(labels)
        label_parts.append(labels)
        probability_parts.append(probabilities)
        sample_weight_parts.append(np.full(len(labels), per_sample_weight, dtype=float))

    if not label_parts:
        raise ValueError("No fold samples were loaded")

    return (
        np.concatenate(label_parts),
        np.concatenate(probability_parts),
        np.concatenate(sample_weight_parts),
    )


def save_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    sample_weights: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    roc_fpr, roc_tpr, _ = roc_curve(labels, probabilities, sample_weight=sample_weights)
    roc_auc = auc(roc_fpr, roc_tpr)

    pr_precision, pr_recall, _ = precision_recall_curve(labels, probabilities, sample_weight=sample_weights)
    pr_ap = average_precision_score(labels, probabilities, sample_weight=sample_weights)
    prevalence = float(labels.mean())

    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(14, 6))

    roc_axis.plot(roc_fpr, roc_tpr, color="#1f77b4", linewidth=2.5, label=f"ROC AUC = {roc_auc:.3f}")
    roc_axis.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    roc_axis.set_title("ROC Curve")
    roc_axis.set_xlabel("False Positive Rate")
    roc_axis.set_ylabel("True Positive Rate")
    roc_axis.set_xlim(0, 1)
    roc_axis.set_ylim(0, 1)
    roc_axis.grid(True, alpha=0.25)
    roc_axis.legend(loc="lower right")

    pr_axis.step(pr_recall, pr_precision, where="post", color="#d62728", linewidth=2.5, label=f"AP = {pr_ap:.3f}")
    pr_axis.hlines(
        prevalence,
        0,
        1,
        color="gray",
        linestyle="--",
        linewidth=1.0,
        alpha=0.7,
        label=f"No Skill = {prevalence:.3f}",
    )
    pr_axis.set_title("Precision-Recall Curve")
    pr_axis.set_xlabel("Recall")
    pr_axis.set_ylabel("Precision")
    pr_axis.set_xlim(0, 1)
    pr_axis.set_ylim(0, 1)
    pr_axis.grid(True, alpha=0.25)
    pr_axis.legend(loc="lower left")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot DeepHP ensemble ROC and PR curves from saved outputs")
    parser.add_argument("--run", default="01_34.4", help="Run ID")
    parser.add_argument("--strategy", default="f1", help="Ensemble weight strategy")
    parser.add_argument(
        "--input_dir",
        default="/home/twyla/Documents/Classes/masterThesis/finalResults/deepHP/convnext_tiny_weight_perfold_gamma_4.0_DANN_1.0_1.0_LR_1e-5_dropout_0.4_batchsize_64",
        help="Directory containing the corrected prediction JSONs and weights JSON",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory for the output image. Defaults to the input directory.",
    )
    parser.add_argument("--model_name", default="convnext_tiny", help="Model name for the output filename")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_data = load_fold_predictions(args.run, input_dir)
    fold_weights = load_fold_weights(args.run, args.strategy, input_dir)

    missing_weights = sorted(set(fold_data) - set(fold_weights))
    if missing_weights:
        raise ValueError(f"Weights file is missing folds: {missing_weights}")

    labels, probabilities, sample_weights = build_weighted_samples(fold_data, fold_weights)

    output_path = output_dir / f"{args.run}_ensemble_roc_pr_{args.model_name}_{args.strategy}.png"
    title = f"{args.run} DeepHP Ensemble Voting ({args.model_name}, {args.strategy})"
    save_curves(labels, probabilities, sample_weights, output_path, title)

    print(f"Saved ROC/PR curves to {output_path}")


if __name__ == "__main__":
    main()