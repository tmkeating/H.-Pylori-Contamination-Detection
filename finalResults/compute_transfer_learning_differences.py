from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from math import sqrt


SUMMARY_SOURCES = {
    "grand_cv_averages": "grand_cv_averages_all.csv",
    "ensemble_voting": "transfer_ensemble_voting_summaries_all.csv",
    "hybrid_ensemble": "transfer_hybrid_ensemble_summaries_all.csv",
    "meta_classifier": "transfer_meta_classifier_summaries_all.csv",
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent
    default_input = workspace_root
    default_output_dir = workspace_root

    parser = argparse.ArgumentParser(
        description=(
            "Compute TRUE - FALSE transfer-learning differences for each "
            "Metric/Pos_Weight/Gamma pair and average them per metric."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Input CSV file or directory containing the summary CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where per-summary difference CSV files will be written.",
    )
    return parser.parse_args()


def normalize_transfer_learning(value: object) -> str:
    return str(value).strip().upper()


def resolve_inputs(input_path: Path) -> list[tuple[str, Path]]:
    if input_path.is_file():
        return [(input_path.stem, input_path)]

    resolved_inputs: list[tuple[str, Path]] = []
    for summary_type, filename in SUMMARY_SOURCES.items():
        csv_path = input_path / filename
        if csv_path.is_file():
            resolved_inputs.append((summary_type, csv_path))

    if not resolved_inputs:
        raise ValueError(f"No supported summary CSVs found in {input_path}")

    return resolved_inputs


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    required_columns = {
        "Metric",
        "Pos_Weight",
        "Gamma",
        "Mean",
        "Std",
        "Transfer Learning",
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    frame["Transfer Learning"] = frame["Transfer Learning"].map(normalize_transfer_learning)
    frame = frame.loc[frame["Transfer Learning"].isin(["TRUE", "FALSE"])].copy()
    if frame.empty:
        raise ValueError("No rows found with Transfer Learning values TRUE/FALSE.")

    frame["Pos_Weight"] = pd.to_numeric(frame["Pos_Weight"], errors="raise")
    frame["Gamma"] = pd.to_numeric(frame["Gamma"], errors="raise")
    frame["Mean"] = pd.to_numeric(frame["Mean"], errors="raise")
    frame["Std"] = pd.to_numeric(frame["Std"], errors="coerce").fillna(0.0)
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
        .sort_index(axis=1)
        .reset_index()
    )

    pivoted.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in pivoted.columns.to_flat_index()
    ]

    required_pivoted_columns = ["Mean_TRUE", "Mean_FALSE", "Std_TRUE", "Std_FALSE"]
    missing_pivoted_columns = [
        column for column in required_pivoted_columns if column not in pivoted.columns
    ]
    if missing_pivoted_columns:
        missing = ", ".join(missing_pivoted_columns)
        raise ValueError(f"Expected TRUE/FALSE Mean and Std columns after pivot. Missing: {missing}")

    if pivoted[required_pivoted_columns].isna().any().any():
        raise ValueError(
            "At least one Metric/Pos_Weight/Gamma pair is missing TRUE or FALSE Mean/Std values."
        )

    pivoted = pivoted.rename(
        columns={
            "Mean_TRUE": "Mean_Transfer_Learning_True",
            "Mean_FALSE": "Mean_Transfer_Learning_False",
            "Std_TRUE": "Std_Transfer_Learning_True",
            "Std_FALSE": "Std_Transfer_Learning_False",
        }
    )
    pivoted["Difference_True_Minus_False"] = (
        pivoted["Mean_Transfer_Learning_True"] - pivoted["Mean_Transfer_Learning_False"]
    )
    pivoted["Propagated_Std_True_Minus_False"] = (
        pivoted["Std_Transfer_Learning_True"].pow(2)
        + pivoted["Std_Transfer_Learning_False"].pow(2)
    ).pow(0.5)
    pivoted = pivoted.sort_values(["Metric", "Pos_Weight", "Gamma"]).reset_index(drop=True)
    return pivoted


def compute_metric_averages(pairwise: pd.DataFrame) -> pd.DataFrame:
    summary = (
        pairwise.groupby("Metric", as_index=False)
        .agg(
            Average_Difference_True_Minus_False=("Difference_True_Minus_False", "mean"),
            Pair_Count=("Difference_True_Minus_False", "size"),
            Mean_Propagated_Variance=(
                "Propagated_Std_True_Minus_False",
                lambda series: series.pow(2).sum(),
            ),
        )
        .sort_values("Metric")
        .reset_index(drop=True)
    )
    summary["Propagated_Std_Average_Difference"] = summary.apply(
        lambda row: sqrt(row["Mean_Propagated_Variance"]) / row["Pair_Count"],
        axis=1,
    )
    summary = summary.drop(columns=["Mean_Propagated_Variance"])
    return summary


def build_output_paths(output_dir: Path, summary_type: str) -> tuple[Path, Path]:
    average_path = output_dir / f"transfer_learning_metric_average_differences_{summary_type}.csv"
    pairwise_path = output_dir / f"transfer_learning_pairwise_differences_{summary_type}.csv"
    return average_path, pairwise_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for summary_type, csv_path in resolve_inputs(args.input):
        frame = load_and_validate(csv_path)
        pairwise = compute_pairwise_differences(frame)
        summary = compute_metric_averages(pairwise)
        average_path, pairwise_path = build_output_paths(args.output_dir, summary_type)

        summary.to_csv(average_path, index=False)
        pairwise.to_csv(pairwise_path, index=False)

        print(f"Saved metric averages to {average_path}")
        print(f"Saved pairwise differences to {pairwise_path}")


if __name__ == "__main__":
    main()