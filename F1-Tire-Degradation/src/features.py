"""
src/features.py
---------------
Feature selection + preprocessing for the tire degradation model.

Numeric features (Tire_wear, Humidity, Ambient_Temperature) are standardized.
Event is one-hot encoded rather than ordinally factorized, since it is a
nominal category with no natural ordering. The encoder is fit as part of the
sklearn Pipeline built in src/train.py, so its learned categories are saved
and loaded together with the model (a single model.joblib) instead of a
separate JSON artifact. Unseen Event values at inference time are encoded as
an all-zero vector rather than raising.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["Tire_wear", "Humidity", "Ambient_Temperature"]
CATEGORICAL_FEATURES = ["Event"]
FEATURE_COLUMNS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer for the model Pipeline."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("event", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Validates and returns the raw feature columns (before preprocessing)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURE_COLUMNS].copy()
