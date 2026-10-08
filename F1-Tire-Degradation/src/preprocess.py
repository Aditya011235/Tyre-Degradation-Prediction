"""
src/preprocess.py
-----------------
Data loading + light cleaning for the tire degradation project.

Supports:
- Loading from a local CSV path (recommended for reproducibility).
- Loading via kagglehub (requires Kaggle credentials and internet access).

Usage:
    python -m src.preprocess --csv_path data/simulated_dataset.csv
    python -m src.preprocess --use_kagglehub --out_csv data/simulated_dataset.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

DEFAULT_TARGET = "Tire_Degradation"


@dataclass(frozen=True)
class DatasetConfig:
    target: str = DEFAULT_TARGET


def standardize_columns(df: pd.DataFrame, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    """
    Standardize known column naming issues in the dataset.
    The Kaggle dataset used in the notebook has a typo: 'Tire degreadation'.
    """
    df = df.copy()
    if "Tire degreadation" in df.columns and target not in df.columns:
        df = df.rename(columns={"Tire degreadation": target})
    return df


def load_from_csv(csv_path: str, *, target: str = DEFAULT_TARGET) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return standardize_columns(df, target=target)


def load_from_kagglehub(
    *,
    dataset: str = "samwelnjehia/simple-tire-wear-and-degradation-simulated-dataset",
    filename: str = "simulated_dataset.csv",
    target: str = DEFAULT_TARGET,
) -> pd.DataFrame:
    """
    Loads the dataset using kagglehub. This requires Kaggle credentials and internet.
    """
    import kagglehub
    from kagglehub import KaggleDatasetAdapter

    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset,
        filename,
    )
    return standardize_columns(df, target=target)


def main() -> None:
    p = argparse.ArgumentParser(description="Load and standardize the tire degradation dataset.")
    p.add_argument("--csv_path", type=str, default=None, help="Path to a local CSV to load.")
    p.add_argument(
        "--use_kagglehub", action="store_true", help="Load from kagglehub instead of local CSV."
    )
    p.add_argument(
        "--out_csv", type=str, default=None, help="Optional path to write the loaded dataset."
    )
    p.add_argument(
        "--target", type=str, default=DEFAULT_TARGET, help="Target column name to standardize to."
    )
    args = p.parse_args()

    if args.use_kagglehub:
        df = load_from_kagglehub(target=args.target)
    else:
        if not args.csv_path:
            raise SystemExit("Provide --csv_path or pass --use_kagglehub")
        df = load_from_csv(args.csv_path, target=args.target)

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote CSV to: {args.out_csv}")

    print(f"Shape: {df.shape}")
    print(df.head())


if __name__ == "__main__":
    main()
