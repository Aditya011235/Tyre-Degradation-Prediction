"""
src/predict.py
--------------
Load a saved pipeline (preprocessing + model, self-contained) and run
predictions.

Usage:
    python -m src.predict --input_csv data/sample_input.csv --out_csv outputs/preds.csv
    python -m src.predict --tire_wear 0.42 --humidity 55 \
        --ambient_temperature 22 --event "Monaco GP"
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
from joblib import load

from .features import select_features


def main() -> None:
    p = argparse.ArgumentParser(description="Predict tire degradation using a saved model.")
    p.add_argument("--out_dir", type=str, default="outputs", help="Directory with model.joblib.")

    p.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="CSV with Tire_wear, Humidity, Ambient_Temperature, Event",
    )
    p.add_argument("--out_csv", type=str, default=None)

    p.add_argument("--tire_wear", type=float, default=None)
    p.add_argument("--humidity", type=float, default=None)
    p.add_argument("--ambient_temperature", type=float, default=None)
    p.add_argument("--event", type=str, default=None)
    args = p.parse_args()

    model_path = os.path.join(args.out_dir, "model.joblib")
    pipeline = load(model_path)

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        X = select_features(df)
        preds = pipeline.predict(X)

        out = df.copy()
        out["predicted_tire_degradation"] = preds
        print(out.head(20).to_string(index=False))

        if args.out_csv:
            out.to_csv(args.out_csv, index=False)
            print(f"\nWrote predictions to: {args.out_csv}")
        return

    required = [args.tire_wear, args.humidity, args.ambient_temperature, args.event]
    if any(v is None for v in required):
        raise SystemExit(
            "Provide --input_csv OR all of: --tire_wear --humidity "
            "--ambient_temperature --event"
        )

    df = pd.DataFrame([{
        "Tire_wear": args.tire_wear,
        "Humidity": args.humidity,
        "Ambient_Temperature": args.ambient_temperature,
        "Event": args.event,
    }])

    X = select_features(df)
    pred = float(pipeline.predict(X)[0])
    print(f"Predicted Tire Degradation: {pred:.6f}")


if __name__ == "__main__":
    main()
