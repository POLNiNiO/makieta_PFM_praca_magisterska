"""Score users and generate PFM recommendations with a trained model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd

from .features import FeatureConfig, build_features
from .recommender import generate_recommendations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PFM recommendations for users.")
    parser.add_argument("--model-dir", default="models/pfm_insolvency_1m")
    parser.add_argument("--transactions", required=True, help="Parquet file with user transactions.")
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(Path(args.model_dir))
    scores, features = score_transactions(
        Path(args.transactions),
        bundle,
        progress=args.progress,
    )

    if args.user_id is not None:
        if args.user_id not in scores.index:
            raise SystemExit(f"User {args.user_id} was not found in {args.transactions}")
        payload = build_user_payload(args.user_id, scores, features, bundle)
    else:
        top_users = scores.sort_values("risk_probability", ascending=False).head(args.top_n)
        payload = {
            "users": [
                build_user_payload(int(user_id), scores, features, bundle)
                for user_id in top_users.index
            ]
        }

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output_json:
        Path(args.output_json).write_text(text)
    print(text)


def load_model_bundle(model_dir: Path) -> Dict[str, object]:
    return joblib.load(model_dir / "model.joblib")


def score_transactions(
    transactions_path: Path,
    bundle: Dict[str, object],
    progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    config = FeatureConfig.from_dict(bundle["feature_config"])
    features = build_features(transactions_path, config=config, progress=progress)
    feature_columns = list(bundle["feature_columns"])
    x = features.reindex(columns=feature_columns, fill_value=0.0)
    model = bundle["model"]
    probabilities = model.predict_proba(x)[:, 1]
    threshold = float(bundle["decision_threshold"])
    scores = pd.DataFrame(
        {
            "risk_probability": probabilities,
            "risk_label": probabilities >= threshold,
        },
        index=features.index,
    )
    return scores, features


def build_user_payload(
    user_id: int,
    scores: pd.DataFrame,
    features: pd.DataFrame,
    bundle: Dict[str, object],
) -> Dict[str, object]:
    risk_probability = float(scores.loc[user_id, "risk_probability"])
    profile = features.loc[user_id].to_dict()
    return {
        "user_id": int(user_id),
        "risk_probability": risk_probability,
        "risk_label": bool(scores.loc[user_id, "risk_label"]),
        "recommendations": generate_recommendations(
            profile,
            risk_probability=risk_probability,
            decision_threshold=float(bundle["decision_threshold"]),
        ),
    }


if __name__ == "__main__":
    main()
