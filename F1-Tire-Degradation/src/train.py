"""
src/train.py
------------
Compare regressors via cross-validation, refit the best one, evaluate it on
a held-out test split, and save it as a single self-contained pipeline
(preprocessing + model) to model.joblib.

Usage:
    python -m src.train --csv_path data/simulated_dataset.csv
    python -m src.train --use_kagglehub
    python -m src.train --csv_path data/simulated_dataset.csv --tune
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .features import build_preprocessor, select_features
from .preprocess import DEFAULT_TARGET, load_from_csv, load_from_kagglehub

RANDOM_STATE = 42

RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [200, 300, 400, 600],
    "model__max_depth": [None, 8, 16, 24],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 1.0],
}


def build_candidates() -> dict[str, Pipeline]:
    """One Pipeline (preprocess + model) per candidate regressor."""
    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(64, 32),
            random_state=RANDOM_STATE,
            max_iter=800,
            early_stopping=True,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", reg)])
        for name, reg in regressors.items()
    }


def cross_validate_all(
    candidates: dict[str, Pipeline], X_train, y_train, cv_folds: int
) -> dict[str, dict[str, float]]:
    """K-fold CV for each candidate; preprocessing is refit inside every fold."""
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=kfold,
            scoring=["r2", "neg_root_mean_squared_error"],
            n_jobs=-1,
        )
        results[name] = {
            "cv_r2_mean": float(scores["test_r2"].mean()),
            "cv_r2_std": float(scores["test_r2"].std()),
            "cv_rmse_mean": float(-scores["test_neg_root_mean_squared_error"].mean()),
            "cv_rmse_std": float(scores["test_neg_root_mean_squared_error"].std()),
        }
    return results


def select_best(cv_results: dict[str, dict[str, float]]) -> str:
    items = list(cv_results.items())
    items.sort(key=lambda kv: (-kv[1]["cv_r2_mean"], kv[1]["cv_rmse_mean"]))
    return items[0][0]


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict[str, str | float]:
    preds = pipeline.predict(X_test)
    return {
        "model": name,
        "r2": float(r2_score(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mae": float(mean_absolute_error(y_test, preds)),
    }


def tune_random_forest(
    X_train, y_train, cv_folds: int, n_iter: int
) -> tuple[Pipeline, dict[str, object], float]:
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        RF_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE),
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def main() -> None:
    p = argparse.ArgumentParser(description="Train tire degradation models and save the best.")
    p.add_argument("--csv_path", type=str, default=None, help="Path to local CSV dataset.")
    p.add_argument("--use_kagglehub", action="store_true", help="Load dataset via kagglehub.")
    p.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Target column name.")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--cv_folds", type=int, default=5, help="K-fold CV splits for model selection.")
    p.add_argument(
        "--tune",
        action="store_true",
        help="If Random Forest wins model selection, tune it via RandomizedSearchCV.",
    )
    p.add_argument("--tune_iter", type=int, default=20, help="RandomizedSearchCV iterations.")
    p.add_argument("--out_dir", type=str, default="outputs")
    args = p.parse_args()

    if args.use_kagglehub:
        df = load_from_kagglehub(target=args.target)
    else:
        if not args.csv_path:
            raise SystemExit("Provide --csv_path or pass --use_kagglehub")
        df = load_from_csv(args.csv_path, target=args.target)

    X = select_features(df)
    y = df[args.target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE
    )

    candidates = build_candidates()
    cv_results = cross_validate_all(candidates, X_train, y_train, args.cv_folds)
    best_name = select_best(cv_results)

    best_pipeline = candidates[best_name]
    best_pipeline.fit(X_train, y_train)
    holdout_metrics = evaluate(best_name, best_pipeline, X_test, y_test)

    tuning_info = None
    if args.tune and best_name == "Random Forest":
        tuned_pipeline, best_params, best_cv_r2 = tune_random_forest(
            X_train, y_train, args.cv_folds, args.tune_iter
        )
        tuned_metrics = evaluate("Random Forest (tuned)", tuned_pipeline, X_test, y_test)
        tuning_info = {
            "best_params": best_params,
            "cv_r2": best_cv_r2,
            "holdout_metrics": tuned_metrics,
        }
        if float(tuned_metrics["r2"]) > float(holdout_metrics["r2"]):
            best_name = "Random Forest (tuned)"
            best_pipeline = tuned_pipeline
            holdout_metrics = tuned_metrics

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "model.joblib")
    dump(best_pipeline, model_path)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "holdout_metrics": holdout_metrics,
                "cv_results": cv_results,
                "tuning": tuning_info,
            },
            f,
            indent=2,
        )

    print("Training complete.")
    print(f"Best model: {best_name}")
    print(f"Holdout R2: {holdout_metrics['r2']:.4f}  RMSE: {holdout_metrics['rmse']:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
