import pytest

from src.features import FEATURE_COLUMNS, build_preprocessor, select_features


def test_select_features_returns_expected_columns(synthetic_df):
    X = select_features(synthetic_df)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(synthetic_df)


def test_select_features_raises_on_missing_column(synthetic_df):
    df = synthetic_df.drop(columns=["Humidity"])
    with pytest.raises(ValueError):
        select_features(df)


def test_preprocessor_one_hot_encodes_event_and_scales_numeric(synthetic_df):
    X = select_features(synthetic_df)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    n_events = synthetic_df["Event"].nunique()
    assert transformed.shape == (len(X), 3 + n_events)


def test_preprocessor_handles_unseen_event_at_transform_time(synthetic_df):
    X = select_features(synthetic_df)
    preprocessor = build_preprocessor()
    preprocessor.fit(X)

    unseen = X.iloc[[0]].copy()
    unseen["Event"] = "Unseen GP"

    # Should not raise, despite an Event category never seen during fit.
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
