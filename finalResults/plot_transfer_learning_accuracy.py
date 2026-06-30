from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


SUMMARY_FILE_PATTERN = re.compile(r"transfer_(.+)_summaries_all\.csv$")
EXPECTED_SUMMARY_TYPES = [
    "ensemble_voting",
    "hybrid_ensemble",
    "meta_classifier",
]
GRAND_CV_SUMMARY_TYPE = "grand_cv_averages"


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent
    default_input = workspace_root
    default_output_dir = workspace_root / "transfer_learning_metric_plots"

    parser = argparse.ArgumentParser(
        description="Plot each metric with error bars for transfer learning runs from one CSV or a directory of collated CSVs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input CSV file. Defaults to {default_input.name}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory for generated plots. Defaults to {default_output_dir.name}.",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also generate combined shared-metric bar charts (disabled by default).",
    )
    return parser.parse_args()


def normalize_transfer_learning(value: object) -> str:
    return str(value).strip().upper()


def resolve_input_csvs(input_path: Path) -> list[tuple[Path, str | None]]:
    if input_path.is_file():
        return [(input_path, extract_summary_type(input_path))]

    csv_entries: list[tuple[Path, str | None]] = []

    grand_cv_path = input_path / "grand_cv_averages_all.csv"
    if grand_cv_path.is_file():
        csv_entries.append((grand_cv_path, GRAND_CV_SUMMARY_TYPE))

    csv_entries.extend(
        (
            input_path / f"transfer_{summary_type}_summaries_all.csv",
            summary_type,
        )
        for summary_type in EXPECTED_SUMMARY_TYPES
        if (input_path / f"transfer_{summary_type}_summaries_all.csv").is_file()
    )

    if not csv_entries:
        raise ValueError(f"No collated summary CSVs found in {input_path}")

    return csv_entries


def extract_summary_type(csv_path: Path) -> str | None:
    if csv_path.name == "grand_cv_averages_all.csv":
        return GRAND_CV_SUMMARY_TYPE
    match = SUMMARY_FILE_PATTERN.match(csv_path.name)
    if match is None:
        return None
    return match.group(1)


def build_plot_frame(csv_path: Path) -> pd.DataFrame:
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
        raise ValueError(f"Missing required columns: {missing}")

    frame["Metric"] = frame["Metric"].astype(str).str.strip()
    frame["Transfer Learning"] = frame["Transfer Learning"].map(normalize_transfer_learning)
    frame = frame.loc[frame["Transfer Learning"].isin(["TRUE", "FALSE"])].copy()
    if frame.empty:
        raise ValueError("No rows found for Transfer Learning values TRUE/FALSE.")

    frame["Pos_Weight"] = pd.to_numeric(frame["Pos_Weight"], errors="raise")
    frame["Gamma"] = pd.to_numeric(frame["Gamma"], errors="raise")
    frame["Mean"] = pd.to_numeric(frame["Mean"], errors="raise")
    frame["Std"] = pd.to_numeric(frame["Std"], errors="coerce").fillna(0.0)
    frame = frame.sort_values(by=["Metric", "Pos_Weight", "Gamma", "Transfer Learning"])
    frame["X_Label"] = frame.apply(
        lambda row: f"PW {row['Pos_Weight']:.1f} | G {row['Gamma']:.1f}",
        axis=1,
    )

    return frame


def build_combined_plot_frame(input_entries: list[tuple[Path, str | None]]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path, summary_type in input_entries:
        if summary_type is None:
            continue
        frame = build_plot_frame(csv_path).copy()
        frame["Summary_Type"] = summary_type
        frames.append(frame)

    if not frames:
        raise ValueError("No summary CSVs available for combined plots.")

    combined = pd.concat(frames, ignore_index=True)
    summary_metric_sets = [set(frame["Metric"].drop_duplicates().tolist()) for frame in frames]
    shared_metrics = set.intersection(*summary_metric_sets)
    if not shared_metrics:
        raise ValueError("No shared metrics found across the selected summary CSVs.")

    combined = combined.loc[combined["Metric"].isin(shared_metrics)].copy()
    return combined.sort_values(
        by=["Metric", "Pos_Weight", "Gamma", "Summary_Type", "Transfer Learning"]
    ).reset_index(drop=True)


def sanitize_metric_name(metric: str) -> str:
    safe_name = metric.lower()
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


def format_summary_type(summary_type: str | None) -> str:
    if summary_type is None:
        return ""
    return summary_type.replace("_", " ").title()


def plot_combined_metric(frame: pd.DataFrame, metric: str, output_path: Path) -> None:
    metric_frame = frame.loc[frame["Metric"].eq(metric)].copy()
    ordered_labels = metric_frame[["Pos_Weight", "Gamma", "X_Label"]].drop_duplicates().sort_values(
        by=["Pos_Weight", "Gamma"]
    )
    x_positions = np.arange(len(ordered_labels))
    summary_types = [
        GRAND_CV_SUMMARY_TYPE,
        *EXPECTED_SUMMARY_TYPES,
    ]
    summary_types = [
        summary_type for summary_type in summary_types if summary_type in metric_frame["Summary_Type"].unique()
    ]
    transfer_values = ["FALSE", "TRUE"]
    total_series = len(summary_types) * len(transfer_values)
    bar_width = 0.8 / max(total_series, 1)

    summary_colors = {
        GRAND_CV_SUMMARY_TYPE: "#1b9e77",
        "ensemble_voting": "#d95f02",
        "hybrid_ensemble": "#7570b3",
        "meta_classifier": "#e7298a",
    }
    transfer_hatches = {"FALSE": "", "TRUE": "//"}
    label_positions = dict(zip(ordered_labels["X_Label"], x_positions))

    fig, ax = plt.subplots(figsize=(16, 7))
    offset_index = 0
    for summary_type in summary_types:
        for transfer_value in transfer_values:
            series = metric_frame.loc[
                metric_frame["Summary_Type"].eq(summary_type)
                & metric_frame["Transfer Learning"].eq(transfer_value)
            ].copy()
            if series.empty:
                offset_index += 1
                continue

            series["Base_X"] = series["X_Label"].map(label_positions)
            series = series.sort_values(by="Base_X")
            offsets = (offset_index - (total_series - 1) / 2) * bar_width
            ax.bar(
                series["Base_X"] + offsets,
                series["Mean"],
                width=bar_width,
                yerr=series["Std"],
                capsize=3,
                color=summary_colors.get(summary_type, "#666666"),
                edgecolor="black",
                linewidth=0.6,
                hatch=transfer_hatches[transfer_value],
                label=f"{format_summary_type(summary_type)} | TL {transfer_value}",
            )
            offset_index += 1

    ax.set_title(f"{metric} Across Shared Summaries")
    ax.set_xlabel("Pos_Weight | Gamma")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_labels["X_Label"], rotation=45, ha="right")
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric(frame: pd.DataFrame, metric: str, output_path: Path, summary_type: str | None) -> None:
    transfer_values = [
        transfer_value
        for transfer_value in ["FALSE", "TRUE"]
        if transfer_value in frame["Transfer Learning"].unique()
    ]
    if not transfer_values:
        raise ValueError(f"No transfer-learning values available for metric {metric}")

    pos_weights = sorted(frame["Pos_Weight"].unique().tolist())
    gammas = sorted(frame["Gamma"].unique().tolist())
    vmin = frame["Mean"].min()
    vmax = frame["Mean"].max()
    cmap = LinearSegmentedColormap.from_list(
        "blue_green_yellow",
        ["#2166ac", "#1a9850", "#ffe34d"],
        N=256,
    )
    cmap.set_bad(color="#f0f0f0")

    fig_width = max(8, 6 * len(transfer_values))
    fig, axes = plt.subplots(
        1,
        len(transfer_values),
        figsize=(fig_width, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(transfer_values) == 1:
        axes = [axes]

    image = None
    for axis, transfer_value in zip(axes, transfer_values):
        heatmap_frame = frame.loc[frame["Transfer Learning"].eq(transfer_value)].copy()
        matrix = heatmap_frame.pivot_table(
            index="Pos_Weight",
            columns="Gamma",
            values="Mean",
            aggfunc="mean",
        ).reindex(index=pos_weights, columns=gammas)

        matrix_values = matrix.to_numpy(dtype=float)
        masked_values = np.ma.masked_invalid(matrix_values)
        image = axis.imshow(masked_values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")

        axis.set_title(f"Transfer Learning = {transfer_value}")
        axis.set_xlabel("Gamma")
        axis.set_xticks(np.arange(len(gammas)))
        axis.set_xticklabels([f"{gamma:.1f}" for gamma in gammas])
        axis.set_yticks(np.arange(len(pos_weights)))
        axis.set_yticklabels([f"{pos_weight:.1f}" for pos_weight in pos_weights])

        for row_index in range(len(pos_weights)):
            for col_index in range(len(gammas)):
                value = matrix_values[row_index, col_index]
                if np.isnan(value):
                    continue
                normalized = 0.0 if vmax == vmin else (value - vmin) / (vmax - vmin)
                red, green, blue, _ = cmap(normalized)
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "black" if luminance > 0.5 else "white"
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    summary_label = format_summary_type(summary_type)
    title = f"{metric} Matrix by Pos_Weight and Gamma"
    if summary_label:
        title = f"{metric} Matrix by Pos_Weight and Gamma ({summary_label})"
    fig.suptitle(title)
    axes[0].set_ylabel("Pos_Weight")
    if image is not None:
        fig.colorbar(image, ax=axes, fraction=0.03, pad=0.04, label=f"Mean {metric}")
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_entries = resolve_input_csvs(args.input)

    for csv_path, summary_type in input_entries:
        plot_frame = build_plot_frame(csv_path)
        summary_prefix = ""
        if summary_type is not None:
            summary_prefix = f"{sanitize_metric_name(summary_type)}_"

        metrics = plot_frame["Metric"].drop_duplicates().tolist()
        total_metrics = len(metrics)
        for index, metric in enumerate(metrics, start=1):
            metric_frame = plot_frame.loc[plot_frame["Metric"].eq(metric)].copy()
            output_path = args.output_dir / (
                f"transfer_learning_{summary_prefix}{sanitize_metric_name(metric)}.png"
            )
            plot_metric(metric_frame, metric, output_path, summary_type)
            print(f"[{index}/{total_metrics}] Saved matrix plot to {output_path}")

    if args.include_combined:
        combined_frame = build_combined_plot_frame(input_entries)
        combined_metrics = combined_frame["Metric"].drop_duplicates().tolist()
        total_combined_metrics = len(combined_metrics)
        for index, metric in enumerate(combined_metrics, start=1):
            output_path = args.output_dir / f"transfer_learning_combined_{sanitize_metric_name(metric)}.png"
            plot_combined_metric(combined_frame, metric, output_path)
            print(f"[{index}/{total_combined_metrics}] Saved combined bar plot to {output_path}")


if __name__ == "__main__":
    main()