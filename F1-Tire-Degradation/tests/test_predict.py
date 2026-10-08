from joblib import dump, load

from src.features import select_features
from src.train import build_candidates


def test_saved_pipeline_round_trips_through_joblib(tmp_path, synthetic_df):
    X = select_features(synthetic_df)
    y = synthetic_df["Tire_Degradation"]

    pipeline = build_candidates()["Linear Regression"]
    pipeline.fit(X, y)

    model_path = tmp_path / "model.joblib"
    dump(pipeline, model_path)
    loaded = load(model_path)

    preds = loaded.predict(X.iloc[:5])
    assert len(preds) == 5


def test_predict_handles_previously_unseen_event(tmp_path, synthetic_df):
    X = select_features(synthetic_df)
    y = synthetic_df["Tire_Degradation"]

    pipeline = build_candidates()["Random Forest"]
    pipeline.fit(X, y)

    unseen = X.iloc[[0]].copy()
    unseen["Event"] = "Unseen GP"

    # Previously this required a hand-rolled factorizer with a -1 fallback;
    # the OneHotEncoder(handle_unknown="ignore") pipeline handles it directly.
    pred = pipeline.predict(unseen)
    assert pred.shape == (1,)
