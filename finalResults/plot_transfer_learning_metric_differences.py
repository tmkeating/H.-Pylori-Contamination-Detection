from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_SOURCES = {
    "grand_cv_averages": "grand_cv_averages_all.csv",
    "ensemble_voting": "transfer_ensemble_voting_summaries_all.csv",
    "hybrid_ensemble": "transfer_hybrid_ensemble_summaries_all.csv",
    "meta_classifier": "transfer_meta_classifier_summaries_all.csv",
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent
    default_input_dir = workspace_root
    default_output_dir = workspace_root / "transfer_learning_metric_plots"

    parser = argparse.ArgumentParser(
        description=(
            "Generate one transfer-learning difference chart per summary type "
            "using only metrics shared across all summary sources."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing grand_cv_averages_all.csv and the transfer summary CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for generated charts. Defaults to {default_output_dir.name}.",
    )
    return parser.parse_args()


def normalize_transfer_learning(value: object) -> str:
    return str(value).strip().upper()


def sanitize_name(value: str) -> str:
    safe_name = value.lower()
    for old, new in {
        " ": "_",
        "/": "_",
        "(": "",
        ")": "",
        "+": "plus",
        "-": "minus",
    }.items():
        safe_name = safe_name.replace(old, new)
    return safe_name


def format_metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def format_summary_label(summary_type: str) -> str:
    return summary_type.replace("_", " ").title()


def load_summary_frame(csv_path: Path, summary_type: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {
        "Metric",
        "Pos_Weight",
        "Gamma",
        "Mean",
        "Transfer Learning",
        "Std",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in {csv_path.name}: {missing}")

    frame = frame.copy()
    frame["Metric"] = frame["Metric"].astype(str).str.strip()
    frame["Transfer Learning"] = frame["Transfer Learning"].map(normalize_transfer_learning)
    frame = frame.loc[frame["Transfer Learning"].isin(["TRUE", "FALSE"])].copy()
    if frame.empty:
        raise ValueError(f"No TRUE/FALSE transfer-learning rows found in {csv_path.name}.")

    frame["Pos_Weight"] = pd.to_numeric(frame["Pos_Weight"], errors="raise")
    frame["Gamma"] = pd.to_numeric(frame["Gamma"], errors="raise")
    frame["Mean"] = pd.to_numeric(frame["Mean"], errors="raise")
    std_series = pd.to_numeric(frame["Std"], errors="coerce")
    frame["Std"] = pd.Series(std_series, index=frame.index).fillna(0.0)
    frame["Summary_Type"] = summary_type
    return frame


def compute_pairwise_differences(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["Metric", "Pos_Weight", "Gamma"]
    pivoted = (
        frame.pivot_table(
            index=keys,
            columns="Transfer Learning",
            values=["Mean", "Std"],
            aggfunc="first",
        )
        .reset_index()
    )
    flattened_columns = list(pivoted.columns)
    pivoted.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in flattened_columns
    ]

    required_columns = ["Mean_TRUE", "Mean_FALSE", "Std_TRUE", "Std_FALSE"]
    missing_columns = [column for column in required_columns if column not in pivoted.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing TRUE/FALSE columns after pivot: {missing}")

    if pivoted[required_columns].isna().any().any():
        raise ValueError("At least one Metric/Pos_Weight/Gamma pair is missing TRUE or FALSE values.")

    pivoted["Difference_True_Minus_False"] = pivoted["Mean_TRUE"] - pivoted["Mean_FALSE"]
    pivoted["Propagated_Std_True_Minus_False"] = (
        pivoted["Std_TRUE"].pow(2) + pivoted["Std_FALSE"].pow(2)
    ).pow(0.5)
    return pivoted.sort_values(by=keys).reset_index(drop=True)


def summarize_differences(pairwise: pd.DataFrame, summary_type: str) -> pd.DataFrame:
    summary = (
        pairwise.groupby("Metric", as_index=False)
        .agg(
            Average_Difference_True_Minus_False=("Difference_True_Minus_False", "mean"),
            Pair_Count=("Difference_True_Minus_False", "size"),
            Variance_Sum=(
                "Propagated_Std_True_Minus_False",
                lambda series: series.pow(2).sum(),
            ),
        )
        .sort_values(by="Metric")
        .reset_index(drop=True)
    )
    summary["Propagated_Std_Average_Difference"] = summary.apply(
        lambda row: sqrt(row["Variance_Sum"]) / row["Pair_Count"],
        axis=1,
    )
    summary = summary.drop(columns=["Variance_Sum"])
    summary["Summary_Type"] = summary_type
    return summary


def build_summary_frames(input_dir: Path) -> pd.DataFrame:
    raw_frames: list[pd.DataFrame] = []
    metric_sets: list[set[str]] = []

    for summary_type, filename in SUMMARY_SOURCES.items():
        csv_path = input_dir / filename
        if not csv_path.is_file():
            raise ValueError(f"Required input file not found: {csv_path}")

        frame = load_summary_frame(csv_path, summary_type)
        raw_frames.append(frame)
        metric_sets.append(set(frame["Metric"].drop_duplicates().tolist()))

    shared_metrics = set.intersection(*metric_sets)
    if not shared_metrics:
        raise ValueError("No shared metrics found across the four summary sources.")

    summaries: list[pd.DataFrame] = []
    for frame in raw_frames:
        filtered = pd.DataFrame(frame.loc[frame["Metric"].isin(shared_metrics), :].copy())
        summary_type = str(filtered["Summary_Type"].iloc[0])
        pairwise = compute_pairwise_differences(filtered)
        summaries.append(summarize_differences(pairwise, summary_type))

    return pd.concat(summaries, ignore_index=True)


def plot_summary_differences(frame: pd.DataFrame, summary_type: str, output_path: Path) -> None:
    summary_frame = pd.DataFrame(frame.loc[frame["Summary_Type"].eq(summary_type), :].copy())
    summary_frame = summary_frame.sort_values(
        by=["Average_Difference_True_Minus_False"], ascending=False
    ).reset_index(drop=True)

    labels = [format_metric_label(metric) for metric in summary_frame["Metric"]]
    values = summary_frame["Average_Difference_True_Minus_False"]
    errors = summary_frame["Propagated_Std_Average_Difference"]
    colors = ["#2e7d32" if value >= 0 else "#c62828" for value in values]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(
        labels,
        values,
        xerr=errors,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"Average Transfer-Learning Difference by Metric ({format_summary_label(summary_type)})")
    ax.set_xlabel("Average Difference (TRUE - FALSE)")
    ax.set_ylabel("Shared Metric")
    ax.set_xlim(-0.2, 0.2)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_frame = build_summary_frames(args.input_dir)
    for summary_type in SUMMARY_SOURCES:
        output_path = (
            args.output_dir / f"transfer_learning_metric_differences_{sanitize_name(summary_type)}.png"
        )
        plot_summary_differences(summary_frame, summary_type, output_path)
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()