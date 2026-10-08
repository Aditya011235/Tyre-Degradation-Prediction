from sklearn.pipeline import Pipeline

from src.features import select_features
from src.train import (
    build_candidates,
    cross_validate_all,
    evaluate,
    select_best,
    tune_random_forest,
)


def test_build_candidates_returns_one_pipeline_per_model():
    candidates = build_candidates()
    assert set(candidates) == {
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "MLP Regressor",
    }
    assert all(isinstance(pipeline, Pipeline) for pipeline in candidates.values())


def test_cross_validate_all_and_select_best(synthetic_df):
    X = select_features(synthetic_df)
    y = synthetic_df["Tire_Degradation"]
    candidates = build_candidates()

    results = cross_validate_all(candidates, X, y, cv_folds=3)

    assert set(results) == set(candidates)
    for metrics in results.values():
        assert {"cv_r2_mean", "cv_r2_std", "cv_rmse_mean", "cv_rmse_std"} <= set(metrics)

    best_name = select_best(results)
    assert best_name in candidates


def test_evaluate_returns_expected_metric_keys(synthetic_df):
    X = select_features(synthetic_df)
    y = synthetic_df["Tire_Degradation"]

    pipeline = build_candidates()["Linear Regression"]
    pipeline.fit(X, y)

    metrics = evaluate("Linear Regression", pipeline, X, y)
    assert set(metrics) == {"model", "r2", "rmse", "mae"}
    # A linear signal with mild noise should fit reasonably well.
    assert metrics["r2"] > 0.5


def test_tune_random_forest_returns_fitted_pipeline(synthetic_df):
    X = select_features(synthetic_df)
    y = synthetic_df["Tire_Degradation"]

    pipeline, best_params, best_cv_r2 = tune_random_forest(X, y, cv_folds=3, n_iter=3)

    assert isinstance(best_params, dict)
    assert -1.0 <= best_cv_r2 <= 1.0
    preds = pipeline.predict(X.iloc[:5])
    assert len(preds) == 5
