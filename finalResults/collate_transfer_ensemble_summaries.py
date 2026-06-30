from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PROFILE_FUNCTION_PATTERN = re.compile(r"function\s+set_profile_(SEARCHER(?:\d+)?)\(\)")
POS_WEIGHT_PATTERN = re.compile(r"export\s+POS_WEIGHT=([^\n]+)")
GAMMA_PATTERN = re.compile(r"export\s+GAMMA=([^\n]+)")
TRANSFER_FOLDER_PATTERN = re.compile(r"transfer_convnext_tiny_[^_]+_(SEARCHER\d*)$")
SUMMARY_GLOBS = {
    "ensemble_voting": "ensemble_voting_summary_*.csv",
    "hybrid_ensemble": "hybrid_ensemble_summary_*.csv",
    "meta_classifier": "meta_classifier_summary_*.csv",
}


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parent
    default_profiles = workspace_root / "profiles.sh"
    default_output_dir = workspace_root

    parser = argparse.ArgumentParser(
        description=(
            "Collate transfer summary metrics from transfer_convnext_tiny folders "
            "and attach Pos_Weight/Gamma values from profiles.sh."
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
        help="Directory where the per-summary-type CSV files will be written.",
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

        pos_weight_raw = pos_weight_match.group(1).strip().strip('"')
        gamma_raw = gamma_match.group(1).strip().strip('"')

        try:
            pos_weight = float(pos_weight_raw)
            gamma = float(gamma_raw)
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


def build_collated_frame(input_dir: Path, profiles_path: Path) -> pd.DataFrame:
    profiles = parse_profiles(profiles_path)
    records: list[dict[str, object]] = []

    for folder in collect_transfer_folders(input_dir):
        profile_name, transfer_learning = resolve_profile_from_folder(folder.name, set(profiles))
        if profile_name not in profiles:
            raise ValueError(f"No matching profile found in profiles.sh for folder {folder.name}")

        summary_paths: list[tuple[str, Path]] = []
        for summary_type, pattern in SUMMARY_GLOBS.items():
            summary_paths.extend((summary_type, path) for path in sorted(folder.glob(pattern)))
        if not summary_paths:
            raise ValueError(f"No matching summary CSV found in {folder.name}")

        for summary_type, summary_path in summary_paths:
            summary_frame = pd.read_csv(summary_path)
            if set(summary_frame.columns) != {"Metric", "Value"}:
                raise ValueError(
                    f"Expected Metric/Value columns in {summary_path.name}, found {list(summary_frame.columns)}"
                )

            for row in summary_frame.itertuples(index=False):
                value = float(row.Value)
                records.append(
                    {
                        "Metric": str(row.Metric).strip(),
                        "Pos_Weight": profiles[profile_name]["Pos_Weight"],
                        "Gamma": profiles[profile_name]["Gamma"],
                        "Mean": value,
                        "Transfer Learning": transfer_learning,
                        "Std": pd.NA,
                        "Formatted": f"{value:.4f}",
                        "Summary_Type": summary_type,
                        "Profile": profile_name,
                        "Source_Folder": folder.name,
                        "Source_File": summary_path.name,
                    }
                )

    if not records:
        raise ValueError("No ensemble summary records were collected.")

    frame = pd.DataFrame.from_records(records)
    frame = frame.sort_values(
        by=["Summary_Type", "Metric", "Transfer Learning", "Pos_Weight", "Gamma", "Source_Folder"]
    ).reset_index(drop=True)
    return frame


def main() -> None:
    args = parse_args()
    frame = build_collated_frame(args.input_dir, args.profiles)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for summary_type, summary_frame in frame.groupby("Summary_Type"):
        output_path = args.output_dir / f"transfer_{summary_type}_summaries_all.csv"
        output_frame = summary_frame.drop(columns=["Summary_Type"]).reset_index(drop=True)
        output_frame.to_csv(output_path, index=False)
        print(f"Saved {summary_type} CSV to {output_path}")
        print(f"Rows written: {len(output_frame)}")


if __name__ == "__main__":
    main()