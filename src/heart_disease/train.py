"""End-to-end training pipeline.

Usage:
    python -m heart_disease.train                # train + evaluate + save
    python -m heart_disease.train --compare      # also cross-validate 5 algorithms
    python -m heart_disease.train --data path/to/heart.csv --no-plots
"""
from __future__ import annotations

import argparse
import json

import joblib

from . import config, data, eda, evaluate
from .model import build_random_forest


def run(data_path=config.DATA_RAW, make_plots: bool = True, compare: bool = False) -> dict:
    # Steps 1-2: load + preprocess
    df, report = data.load_clean(data_path)
    print(f"Loaded {report.rows_raw} rows | missing={report.missing_total} | "
          f"duplicates removed={report.duplicates_removed} | clean rows={report.rows_clean}")

    # Step 3: analysis
    if make_plots:
        eda.run_all(df)

    # Step 4: split
    X_train, X_test, y_train, y_test = data.train_test(df)
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Step 5: train
    model = build_random_forest()
    model.fit(X_train, y_train)
    print("Model Training Completed Successfully")

    # Step 6: evaluate
    metrics = evaluate.compute_metrics(model, X_test, y_test)
    metrics["data"] = {
        "rows_raw": report.rows_raw,
        "duplicates_removed": report.duplicates_removed,
        "rows_clean": report.rows_clean,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features": len(config.FEATURES),
        "source_file": str(data_path),
    }
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")

    if make_plots:
        evaluate.save_confusion_matrix(model, X_test, y_test)
        evaluate.save_feature_importance(model, list(X_train.columns))

    if compare:
        X, y = data.split_xy(df)
        table = evaluate.compare_models(X, y)
        table.to_csv(config.COMPARISON_PATH, index=False)
        print("\nCross-validated comparison:\n", table.to_string(index=False))

    # Persist model + metrics
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": config.FEATURES}, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {config.MODEL_PATH}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default=str(config.DATA_RAW))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--compare", action="store_true")
    a = p.parse_args()
    run(a.data, make_plots=not a.no_plots, compare=a.compare)


if __name__ == "__main__":
    main()
