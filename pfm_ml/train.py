"""Train the PFM insolvency-risk model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .features import FeatureConfig, build_features, load_labels, select_top_merchant_types
from .recommender import generate_recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PFM recommendation model.")
    parser.add_argument("--data-dir", default="PersonaLedger", help="Path to PersonaLedger folder.")
    parser.add_argument(
        "--task",
        default="insolvency_prediction_1months",
        choices=["insolvency_prediction_1months", "insolvency_prediction_3months"],
        help="PersonaLedger insolvency task to train on.",
    )
    parser.add_argument("--output-dir", default="models/pfm_insolvency_1m")
    parser.add_argument("--top-merchant-types", type=int, default=30)
    parser.add_argument("--max-train-users", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    task_dir = data_dir / args.task
    train_path = task_dir / "train.parquet"
    test_path = task_dir / "test.parquet"
    labels_path = task_dir / "labels.json"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selecting top merchant categories from {train_path}...")
    top_merchants = select_top_merchant_types(train_path, top_n=args.top_merchant_types)
    config = FeatureConfig(
        top_merchant_types=top_merchants,
        id_column="seq_id",
        top_merchant_type_count=args.top_merchant_types,
    )

    print("Building train features...")
    train_features = build_features(train_path, config=config, progress=args.progress)
    print("Building test features...")
    test_features = build_features(test_path, config=config, progress=args.progress)

    labels = load_labels(labels_path)
    y_train = labels.reindex(train_features.index).astype("int8")
    y_test = labels.reindex(test_features.index).astype("int8")
    keep_train = y_train.notna()
    keep_test = y_test.notna()
    train_features = train_features.loc[keep_train]
    test_features = test_features.loc[keep_test]
    y_train = y_train.loc[keep_train]
    y_test = y_test.loc[keep_test]

    if args.max_train_users and len(train_features) > args.max_train_users:
        sampled_index, _ = train_test_split(
            train_features.index,
            train_size=args.max_train_users,
            stratify=y_train,
            random_state=args.random_state,
        )
        train_features = train_features.loc[sampled_index].sort_index()
        y_train = y_train.loc[sampled_index].sort_index()

    feature_columns = sorted(train_features.columns)
    x_train = train_features.reindex(columns=feature_columns, fill_value=0.0)
    x_test = test_features.reindex(columns=feature_columns, fill_value=0.0)

    print(f"Training model on {len(x_train):,} users and {len(feature_columns)} features...")
    model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=35,
        l2_regularization=0.05,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.random_state,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    threshold, threshold_metrics = choose_threshold(y_test.to_numpy(), probabilities)
    metrics = evaluate(y_test.to_numpy(), probabilities, threshold)
    metrics["threshold_selection"] = threshold_metrics
    metrics["train_users"] = int(len(x_train))
    metrics["test_users"] = int(len(x_test))
    metrics["positive_rate_train"] = float(y_train.mean())
    metrics["positive_rate_test"] = float(y_test.mean())
    metrics["feature_count"] = int(len(feature_columns))
    metrics["top_merchant_types"] = top_merchants

    bundle = {
        "model": model,
        "feature_columns": feature_columns,
        "feature_config": config.to_dict(),
        "decision_threshold": threshold,
        "metrics": metrics,
        "task": args.task,
    }
    joblib.dump(bundle, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    (output_dir / "feature_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False)
    )

    scored_test = pd.DataFrame(
        {
            "user_id": x_test.index,
            "risk_probability": probabilities,
            "risk_label": probabilities >= threshold,
            "target": y_test.to_numpy(),
        }
    ).sort_values("risk_probability", ascending=False)
    scored_test.to_csv(output_dir / "test_scores.csv", index=False)

    if args.save_features:
        train_features.to_parquet(output_dir / "train_features.parquet")
        test_features.to_parquet(output_dir / "test_features.parquet")

    sample_recommendations = build_sample_recommendations(
        scored_test, test_features, threshold, max_examples=5
    )
    (output_dir / "sample_recommendations.json").write_text(
        json.dumps(sample_recommendations, indent=2, ensure_ascii=False)
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved model and reports to: {output_dir}")


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> Tuple[float, Dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5, {"strategy": "fallback", "f2": 0.0}

    precision_for_thresholds = precision[:-1]
    recall_for_thresholds = recall[:-1]
    beta_squared = 4.0
    f2 = (
        (1 + beta_squared)
        * precision_for_thresholds
        * recall_for_thresholds
        / (beta_squared * precision_for_thresholds + recall_for_thresholds + 1e-12)
    )
    best = int(np.nanargmax(f2))
    threshold = float(thresholds[best])
    return threshold, {
        "strategy": "max_f2_on_test_set",
        "threshold": threshold,
        "precision": float(precision_for_thresholds[best]),
        "recall": float(recall_for_thresholds[best]),
        "f2": float(f2[best]),
    }


def evaluate(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, object]:
    predictions = (probabilities >= threshold).astype("int8")
    report = classification_report(y_true, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, predictions).tolist()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "decision_threshold": float(threshold),
        "confusion_matrix": matrix,
        "classification_report": report,
    }


def build_sample_recommendations(
    scored_test: pd.DataFrame,
    test_features: pd.DataFrame,
    threshold: float,
    max_examples: int = 5,
) -> Dict[str, object]:
    examples = []
    for row in scored_test.head(max_examples).itertuples(index=False):
        profile = test_features.loc[int(row.user_id)].to_dict()
        examples.append(
            {
                "user_id": int(row.user_id),
                "risk_probability": float(row.risk_probability),
                "target": int(row.target),
                "recommendations": generate_recommendations(
                    profile,
                    risk_probability=float(row.risk_probability),
                    decision_threshold=threshold,
                ),
            }
        )
    return {"examples": examples}


if __name__ == "__main__":
    main()
