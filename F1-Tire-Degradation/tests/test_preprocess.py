import pandas as pd

from src.preprocess import DEFAULT_TARGET, standardize_columns


def test_standardize_columns_renames_known_typo():
    df = pd.DataFrame({"Tire degreadation": [1, 2, 3]})
    result = standardize_columns(df)
    assert DEFAULT_TARGET in result.columns
    assert "Tire degreadation" not in result.columns


def test_standardize_columns_noop_when_target_already_present():
    df = pd.DataFrame({DEFAULT_TARGET: [1, 2, 3], "Tire degreadation": [4, 5, 6]})
    result = standardize_columns(df)
    assert list(result[DEFAULT_TARGET]) == [1, 2, 3]
    assert "Tire degreadation" in result.columns
