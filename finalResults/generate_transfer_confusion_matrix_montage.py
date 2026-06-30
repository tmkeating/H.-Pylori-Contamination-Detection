from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROFILE_FUNCTION_PATTERN = re.compile(r"function\s+set_profile_(SEARCHER(?:\d+)?)\(\)")
POS_WEIGHT_PATTERN = re.compile(r"export\s+POS_WEIGHT=([^\n]+)")
GAMMA_PATTERN = re.compile(r"export\s+GAMMA=([^\n]+)")
TRANSFER_FOLDER_PATTERN = re.compile(r"transfer_convnext_tiny_[^_]+_(SEARCHER\d*)$")
SUMMARY_PATTERNS = {
    "ensemble_voting": {
        "glob": "ensemble_voting_holdout_predictions_*.csv",
        "predicted_column": "Ensemble_Pred",
        "score_column": "Mean_Ensemble_Prob",
        "title": "Ensemble Voting",
    },
    "hybrid_ensemble": {
        "glob": "hybrid_ensemble_holdout_predictions_*.csv",
        "predicted_column": "Predicted",
        "score_column": "Predicted_Probability",
        "title": "Hybrid Ensemble",
    },
    "meta_classifier": {
        "glob": "meta_classifier_holdout_predictions_*.csv",
        "predicted_column": "Predicted",
        "score_column": "Predicted_Probability",
        "title": "Meta Classifier",
    },
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent
    default_profiles = workspace_root / "profiles.sh"
    default_output_dir = workspace_root / "transfer_confusion_matrix_montages"

    parser = argparse.ArgumentParser(
        description=(
            "Generate combined confusion-matrix montages for all transfer folders, "
            "grouped by summary type with no-transfer on the left and transfer on the right."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=workspace_root,
        help="Directory containing transfer_convnext_tiny folders.",
    )
    parser.add_argument(
        "--profiles",
        type=Path,
        default=default_profiles,
        help=f"Path to profiles.sh. Defaults to {default_profiles.name}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for montage images. Defaults to {default_output_dir.name}.",
    )
    return parser.parse_args()


def parse_profiles(profiles_path: Path) -> dict[str, dict[str, float]]:
    text = profiles_path.read_text(encoding="utf-8")
    profiles: dict[str, dict[str, float]] = {}

    for match in PROFILE_FUNCTION_PATTERN.finditer(text):
        profile_name = match.group(1)
        block_start = match.end()
        block_end = text.find("}\n", block_start)
        if block_end == -1:
            raise ValueError(f"Could not find end of function block for {profile_name}.")

        block = text[block_start:block_end]
        pos_weight_match = POS_WEIGHT_PATTERN.search(block)
        gamma_match = GAMMA_PATTERN.search(block)
        if pos_weight_match is None or gamma_match is None:
            continue

        try:
            pos_weight = float(pos_weight_match.group(1).strip().strip('"'))
            gamma = float(gamma_match.group(1).strip().strip('"'))
        except ValueError:
            continue

        profiles[profile_name] = {"Pos_Weight": pos_weight, "Gamma": gamma}

    if not profiles:
        raise ValueError("No SEARCHER profiles with numeric POS_WEIGHT/GAMMA were found.")

    return profiles


def resolve_profile_from_folder(folder_name: str, known_profiles: set[str]) -> tuple[str, str]:
    match = TRANSFER_FOLDER_PATTERN.search(folder_name)
    if match is None:
        raise ValueError(f"Folder name does not match expected transfer pattern: {folder_name}")

    searcher_token = match.group(1)
    if searcher_token in known_profiles:
        return searcher_token, "FALSE"
    if searcher_token == "SEARCHER":
        return "SEARCHER", "FALSE"

    suffix = searcher_token.removeprefix("SEARCHER")
    if suffix == "01":
        return "SEARCHER", "TRUE"
    if suffix == "12":
        return "SEARCHER1", "TRUE"
    if suffix.endswith("1"):
        base_suffix = suffix[:-1]
        if base_suffix.isdigit():
            return f"SEARCHER{base_suffix}", "TRUE"

    return searcher_token, "FALSE"


def collect_transfer_folders(input_dir: Path) -> list[Path]:
    folders = sorted(
        path for path in input_dir.iterdir() if path.is_dir() and path.name.startswith("transfer_convnext_tiny_")
    )
    if not folders:
        raise ValueError("No transfer_convnext_tiny folders found.")
    return folders


def load_prediction_table(csv_path: Path, predicted_column: str, score_column: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {"Actual", predicted_column, score_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    table = frame[["Actual", predicted_column, score_column]].copy()
    table["Actual"] = pd.to_numeric(table["Actual"], errors="raise").astype(int)
    table[predicted_column] = pd.to_numeric(table[predicted_column], errors="raise").astype(int)
    table[score_column] = pd.to_numeric(table[score_column], errors="raise")
    return table


def compute_confusion_counts(table: pd.DataFrame, predicted_column: str) -> list[list[int]]:
    actual = table["Actual"]
    predicted = table[predicted_column]
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    return [[true_negative, false_positive], [false_negative, true_positive]]


def build_entries(input_dir: Path, profiles_path: Path) -> dict[str, list[dict[str, object]]]:
    profiles = parse_profiles(profiles_path)
    entries_by_summary = {summary_type: [] for summary_type in SUMMARY_PATTERNS}

    for folder in collect_transfer_folders(input_dir):
        profile_name, transfer_learning = resolve_profile_from_folder(folder.name, set(profiles))
        if profile_name not in profiles:
            raise ValueError(f"No matching profile found in profiles.sh for folder {folder.name}")

        pos_weight = profiles[profile_name]["Pos_Weight"]
        gamma = profiles[profile_name]["Gamma"]

        for summary_type, config in SUMMARY_PATTERNS.items():
            matches = sorted(folder.glob(str(config["glob"])))
            if not matches:
                raise ValueError(f"No {summary_type} holdout prediction file found in {folder.name}")

            csv_path = matches[0]
            predicted_column = str(config["predicted_column"])
            score_column = str(config["score_column"])
            table = load_prediction_table(csv_path, predicted_column, score_column)
            confusion = compute_confusion_counts(table, predicted_column)
            entries_by_summary[summary_type].append(
                {
                    "folder_name": folder.name,
                    "profile_name": profile_name,
                    "transfer_learning": transfer_learning,
                    "pos_weight": pos_weight,
                    "gamma": gamma,
                    "csv_path": csv_path,
                    "confusion": confusion,
                    "prediction_table": table,
                    "score_column": score_column,
                }
            )

    for summary_type in entries_by_summary:
        entries_by_summary[summary_type].sort(
            key=lambda entry: (
                str(entry["transfer_learning"]),
                float(entry["pos_weight"]),
                float(entry["gamma"]),
                str(entry["folder_name"]),
            )
        )

    return entries_by_summary


def draw_confusion_matrix(ax: plt.Axes, confusion: list[list[int]], title: str) -> None:
    image = ax.imshow(confusion, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=8)
    ax.set_yticklabels(["True 0", "True 1"], fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(length=0)

    max_value = max(max(row) for row in confusion) if confusion else 0
    threshold = max_value / 2 if max_value else 0
    for row_index, row in enumerate(confusion):
        for column_index, value in enumerate(row):
            color = "white" if value > threshold else "black"
            ax.text(column_index, row_index, str(value), ha="center", va="center", color=color, fontsize=9)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("black")


def build_montage(summary_type: str, entries: list[dict[str, object]], output_dir: Path) -> Path:
    false_entries = [entry for entry in entries if entry["transfer_learning"] == "FALSE"]
    true_entries = [entry for entry in entries if entry["transfer_learning"] == "TRUE"]
    if not false_entries and not true_entries:
        raise ValueError(f"No entries found for {summary_type}")

    rows_per_side = 4
    columns_per_side = 3
    total_columns = columns_per_side * 2

    fig, axes = plt.subplots(rows_per_side, total_columns, figsize=(22, 14), squeeze=False)
    fig.suptitle(f"{SUMMARY_PATTERNS[summary_type]['title']} Confusion Matrices", fontsize=18)
    fig.text(0.245, 0.955, "Transfer Learning = FALSE", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.text(0.755, 0.955, "Transfer Learning = TRUE", ha="center", va="center", fontsize=14, fontweight="bold")
    fig.add_artist(
        plt.Line2D(
            [0.5, 0.5],
            [0.06, 0.94],
            transform=fig.transFigure,
            color="#444444",
            linewidth=2,
            linestyle="--",
        )
    )

    side_groups = [
        (false_entries, "No Transfer Learning", 0),
        (true_entries, "With Transfer Learning", columns_per_side),
    ]

    for group_entries, header, column_offset in side_groups:
        center_column = column_offset + 1
        axes[0][center_column].set_title(header, fontsize=13, pad=20)
        for index, entry in enumerate(group_entries):
            row_index = index // columns_per_side
            column_index = column_offset + (index % columns_per_side)
            if row_index >= rows_per_side:
                break

            ax = axes[row_index][column_index]
            panel_title = f"PW {float(entry['pos_weight']):.1f} | G {float(entry['gamma']):.1f}"
            draw_confusion_matrix(ax, entry["confusion"], panel_title)

    for row_index in range(rows_per_side):
        for column_index in range(total_columns):
            ax = axes[row_index][column_index]
            if ax.has_data():
                continue
            ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = output_dir / f"transfer_confusion_montage_{summary_type}.png"
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_roc_points(actual: list[int], scores: list[float]) -> tuple[list[float], list[float]]:
    paired = sorted(zip(scores, actual), key=lambda item: item[0], reverse=True)
    positives = sum(actual)
    negatives = len(actual) - positives
    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0]

    tpr_values = [0.0]
    fpr_values = [0.0]
    true_positive = 0
    false_positive = 0

    for score, group in _group_by_score(paired):
        for _, label in group:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
        tpr_values.append(true_positive / positives)
        fpr_values.append(false_positive / negatives)

    if fpr_values[-1] != 1.0 or tpr_values[-1] != 1.0:
        fpr_values.append(1.0)
        tpr_values.append(1.0)
    return fpr_values, tpr_values


def compute_pr_points(actual: list[int], scores: list[float]) -> tuple[list[float], list[float]]:
    paired = sorted(zip(scores, actual), key=lambda item: item[0], reverse=True)
    positives = sum(actual)
    if positives == 0:
        return [0.0, 1.0], [1.0, 1.0]

    precision_values = [1.0]
    recall_values = [0.0]
    true_positive = 0
    false_positive = 0

    for score, group in _group_by_score(paired):
        for _, label in group:
            if label == 1:
                true_positive += 1
            else:
                false_positive += 1
        recall = true_positive / positives
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
        recall_values.append(recall)
        precision_values.append(precision)

    deduped_recall: list[float] = []
    deduped_precision: list[float] = []
    for recall, precision in zip(recall_values, precision_values):
        if deduped_recall and recall == deduped_recall[-1]:
            deduped_precision[-1] = max(deduped_precision[-1], precision)
            continue
        deduped_recall.append(recall)
        deduped_precision.append(precision)

    return deduped_recall, deduped_precision


def _group_by_score(paired: list[tuple[float, int]]) -> list[tuple[float, list[tuple[float, int]]]]:
    groups: list[tuple[float, list[tuple[float, int]]]] = []
    for score, label in paired:
        if not groups or groups[-1][0] != score:
            groups.append((score, [(score, label)]))
            continue
        groups[-1][1].append((score, label))
    return groups


def compute_auc(x_values: list[float], y_values: list[float]) -> float:
    area = 0.0
    for index in range(1, len(x_values)):
        width = x_values[index] - x_values[index - 1]
        height = (y_values[index] + y_values[index - 1]) / 2
        area += width * height
    return area


def build_curve_label(entry: dict[str, object], summary_type: str) -> str:
    return (
        f"{SUMMARY_PATTERNS[summary_type]['title']} | "
        f"PW {float(entry['pos_weight']):.1f} | G {float(entry['gamma']):.1f}"
    )


def plot_combined_curves(
    entries_by_summary: dict[str, list[dict[str, object]]],
    summary_type: str,
    transfer_learning: str,
    curve_type: str,
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(14, 10))

    entries = entries_by_summary[summary_type]
    filtered_entries = [entry for entry in entries if entry["transfer_learning"] == transfer_learning]
    color_map = plt.get_cmap("tab20")
    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (3, 2, 1, 2))]

    for index, entry in enumerate(filtered_entries):
        line_color = color_map(index % color_map.N)
        line_style = line_styles[index % len(line_styles)]

        table = entry["prediction_table"]
        score_column = str(entry["score_column"])
        actual = [int(value) for value in table["Actual"].tolist()]
        scores = [float(value) for value in table[score_column].tolist()]

        if curve_type == "roc":
            x_values, y_values = compute_roc_points(actual, scores)
            auc_value = compute_auc(x_values, y_values)
            ax.plot(
                x_values,
                y_values,
                linewidth=1.6,
                alpha=0.8,
                color=line_color,
                linestyle=line_style,
                label=f"{build_curve_label(entry, summary_type)} | AUC {auc_value:.3f}",
            )
        else:
            x_values, y_values = compute_pr_points(actual, scores)
            auc_value = compute_auc(x_values, y_values)
            ax.step(
                x_values,
                y_values,
                where="post",
                linewidth=1.6,
                alpha=0.8,
                color=line_color,
                linestyle=line_style,
                label=f"{build_curve_label(entry, summary_type)} | AUC {auc_value:.3f}",
            )

    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="#666666", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        title = "Combined ROC Curves"
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        title = "Combined Precision-Recall Curves"

    transfer_label = "With Transfer Learning" if transfer_learning == "TRUE" else "No Transfer Learning"
    ax.set_title(f"{title} ({format_summary_title(summary_type)}, {transfer_label})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()

    output_path = output_dir / (
        f"transfer_{summary_type}_{curve_type}_{transfer_learning.lower()}.png"
    )
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def format_summary_title(summary_type: str) -> str:
    return str(SUMMARY_PATTERNS[summary_type]["title"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entries_by_summary = build_entries(args.input_dir, args.profiles)
    for summary_type, entries in entries_by_summary.items():
        output_path = build_montage(summary_type, entries, args.output_dir)
        print(f"Saved montage to {output_path}")

    for summary_type in SUMMARY_PATTERNS:
        for curve_type in ["roc", "pr"]:
            for transfer_learning in ["FALSE", "TRUE"]:
                output_path = plot_combined_curves(
                    entries_by_summary,
                    summary_type,
                    transfer_learning,
                    curve_type,
                    args.output_dir,
                )
                print(f"Saved curve plot to {output_path}")


if __name__ == "__main__":
    main()