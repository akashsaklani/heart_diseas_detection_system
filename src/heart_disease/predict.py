"""Prediction on new patient data (workflow steps 7-8)."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache

import joblib
import pandas as pd

from . import config


class ValidationError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_model(path: str = str(config.MODEL_PATH)):
    try:
        bundle = joblib.load(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Trained model not found. Run `python -m heart_disease.train` first."
        ) from e
    return bundle["model"], bundle["features"]


def validate_patient(patient: dict) -> dict:
    """Check all 11 features are present, numeric and in a sane range."""
    missing = [f for f in config.FEATURES if f not in patient or patient[f] in (None, "")]
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    clean = {}
    for f in config.FEATURES:
        try:
            v = float(patient[f])
        except (TypeError, ValueError):
            raise ValidationError(f"'{f}' must be a number, got {patient[f]!r}")
        lo, hi = config.FEATURE_RANGES[f]
        if not lo <= v <= hi:
            raise ValidationError(f"'{f}' must be between {lo} and {hi}, got {v}")
        clean[f] = v
    return clean


def predict_one(patient: dict) -> dict:
    model, features = load_model()
    row = validate_patient(patient)
    X = pd.DataFrame([row], columns=features)
    cls = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    return {
        "prediction": cls,
        "label": f"class {cls}",
        "probability_class_0": round(float(proba[0]), 4),
        "probability_class_1": round(float(proba[1]), 4),
        "disclaimer": config.DISCLAIMER,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Predict for one patient.")
    for f in config.FEATURES:
        p.add_argument(f"--{f}", required=True, help=config.FEATURE_INFO[f])
    a = p.parse_args()
    print(json.dumps(predict_one({f: getattr(a, f) for f in config.FEATURES}), indent=2))


if __name__ == "__main__":
    main()
