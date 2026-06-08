"""Feature engineering for transaction-level PFM data.

The PersonaLedger benchmark stores transactions as parquet files and labels in
JSON files. This module keeps the source data read-only and builds user-level
features in row-group chunks, so large transaction files do not need to be
loaded into memory all at once.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


USER_ID_ALIASES = ("seq_id", "user_id", "customer_id", "client_id")

PFM_CATEGORIES = [
    "housing",
    "utilities",
    "groceries",
    "dining",
    "transport",
    "healthcare",
    "insurance",
    "taxes",
    "education",
    "shopping",
    "entertainment",
    "subscriptions",
    "travel",
    "pets",
    "gifts_donations",
    "debt_payments",
    "other",
]

DISCRETIONARY_CATEGORIES = {
    "dining",
    "shopping",
    "entertainment",
    "subscriptions",
    "travel",
    "gifts_donations",
}

ESSENTIAL_CATEGORIES = {
    "housing",
    "utilities",
    "groceries",
    "transport",
    "healthcare",
    "insurance",
    "taxes",
    "education",
    "debt_payments",
}


@dataclass
class FeatureConfig:
    """Configuration needed to reproduce the exact training feature matrix."""

    top_merchant_types: List[str]
    id_column: Optional[str] = None
    top_merchant_type_count: int = 30

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FeatureConfig":
        return cls(
            top_merchant_types=list(payload.get("top_merchant_types", [])),
            id_column=payload.get("id_column") or None,
            top_merchant_type_count=int(payload.get("top_merchant_type_count", 30)),
        )


def normalize_text(value: object) -> str:
    """Normalize free-text categories to stable ASCII lower-case labels."""

    if value is None or pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[_/&+-]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def slugify(value: object) -> str:
    slug = normalize_text(value)
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
    return slug[:80] or "unknown"


def merchant_type_to_pfm_category(merchant_type: object) -> str:
    """Map granular merchant labels to dashboard-friendly PFM categories."""

    text = normalize_text(merchant_type)

    if any(token in text for token in ("mortgage", "rent")):
        return "housing"
    if any(
        token in text
        for token in (
            "electricity",
            "water bill",
            "internet",
            "cable",
            "mobile phone",
            "phone bill",
            "utility",
            "utilities",
        )
    ):
        return "utilities"
    if any(token in text for token in ("grocery", "groceries", "supermarket", "market")):
        return "groceries"
    if any(
        token in text
        for token in (
            "dining",
            "restaurant",
            "cafe",
            "coffee",
            "food",
            "beverage",
            "bar",
            "liquor",
        )
    ):
        return "dining"
    if "insurance" in text:
        return "insurance"
    if any(token in text for token in ("tax", "government")):
        return "taxes"
    if any(token in text for token in ("medical", "health", "pharmacy", "dental")):
        return "healthcare"
    if any(token in text for token in ("veterinary", "pet")):
        return "pets"
    if any(token in text for token in ("travel", "hotel", "airline", "flight")):
        return "travel"
    if any(token in text for token in ("gas", "auto", "car", "bike", "transport")):
        return "transport"
    if any(token in text for token in ("education", "tuition", "school", "bookstore")):
        return "education"
    if any(token in text for token in ("streaming", "subscription", "software")):
        return "subscriptions"
    if any(token in text for token in ("entertainment", "museum", "sports", "fitness", "music")):
        return "entertainment"
    if any(token in text for token in ("gift", "donation", "charity")):
        return "gifts_donations"
    if "payment" in text:
        return "debt_payments"
    if any(
        token in text
        for token in (
            "retail",
            "electronics",
            "clothing",
            "furniture",
            "home goods",
            "housewares",
            "online retailer",
            "home improvement",
        )
    ):
        return "shopping"
    return "other"


def infer_user_id_column(columns: Sequence[str], preferred: Optional[str] = None) -> str:
    if preferred and preferred in columns:
        return preferred
    for candidate in USER_ID_ALIASES:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a user identifier column. Expected one of: "
        + ", ".join(USER_ID_ALIASES)
    )


def parse_bool_label(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    return int(str(value).strip().lower() in {"true", "1", "yes", "y"})


def load_labels(labels_path: Union[Path, str]) -> pd.Series:
    """Load PersonaLedger labels as an integer Series indexed by user id."""

    path = Path(labels_path)
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected labels JSON to be an object: {path}")
    labels = {int(user_id): parse_bool_label(value) for user_id, value in payload.items()}
    return pd.Series(labels, name="target", dtype="int8").sort_index()


def select_top_merchant_types(
    parquet_path: Union[Path, str],
    top_n: int = 30,
    id_column: Optional[str] = None,
) -> List[str]:
    """Select merchant types with the highest positive spend in the training data."""

    path = Path(parquet_path)
    parquet = pq.ParquetFile(path)
    totals: Dict[str, float] = defaultdict(float)

    for row_group in range(parquet.metadata.num_row_groups):
        table = parquet.read_row_group(row_group, columns=["merchant_type", "amount"])
        df = table.to_pandas()
        merchant_type = df["merchant_type"].map(normalize_text)
        amount = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        positive_amount = amount.clip(lower=0.0)
        chunk_totals = positive_amount.groupby(merchant_type).sum()
        for merchant, value in chunk_totals.items():
            if merchant != "unknown" and value > 0:
                totals[str(merchant)] += float(value)

    return [
        merchant
        for merchant, _ in sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_n]
    ]


def build_features(
    parquet_path: Union[Path, str],
    config: FeatureConfig,
    progress: bool = False,
) -> pd.DataFrame:
    """Build a numeric user-level feature matrix from a transaction parquet file."""

    path = Path(parquet_path)
    parquet = pq.ParquetFile(path)
    columns = list(parquet.schema_arrow.names)
    user_id_column = infer_user_id_column(columns, config.id_column)
    required_columns = [
        user_id_column,
        "timestamp",
        "merchant_type",
        "card_present_or_not",
        "amount",
    ]

    partials: List[pd.DataFrame] = []
    for row_group in range(parquet.metadata.num_row_groups):
        if progress:
            print(
                f"Feature extraction: {path.name} row group "
                f"{row_group + 1}/{parquet.metadata.num_row_groups}",
                flush=True,
            )
        table = parquet.read_row_group(row_group, columns=required_columns)
        chunk = table.to_pandas()
        partials.append(_aggregate_chunk(chunk, user_id_column, config.top_merchant_types))

    if not partials:
        return pd.DataFrame()

    combined = _combine_partials(partials)
    features = _add_derived_features(combined)
    features.index.name = "user_id"
    return features.sort_index()


def _aggregate_chunk(
    chunk: pd.DataFrame,
    user_id_column: str,
    top_merchant_types: Sequence[str],
) -> pd.DataFrame:
    df = chunk.rename(columns={user_id_column: "user_id"}).copy()
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["user_id"])
    df["user_id"] = df["user_id"].astype("int64")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["outflow"] = df["amount"].clip(lower=0.0)
    df["inflow"] = (-df["amount"].clip(upper=0.0)).clip(lower=0.0)
    df["amount_square"] = df["amount"] * df["amount"]
    df["is_outflow"] = (df["amount"] > 0).astype("int8")
    df["is_inflow"] = (df["amount"] < 0).astype("int8")

    card_status = df["card_present_or_not"].map(normalize_text)
    df["card_not_present_tx"] = card_status.isin({"no", "online", "none", "na", "n a"}).astype(
        "int8"
    )
    df["card_present_tx"] = (card_status == "yes").astype("int8")

    timestamp = df["timestamp"]
    df["weekend_tx"] = (timestamp.dt.dayofweek >= 5).fillna(False).astype("int8")
    df["night_tx"] = ((timestamp.dt.hour >= 22) | (timestamp.dt.hour < 6)).fillna(False).astype(
        "int8"
    )

    df["merchant_type_norm"] = df["merchant_type"].map(normalize_text)
    df["pfm_category"] = df["merchant_type_norm"].map(merchant_type_to_pfm_category)
    df["tx_marker"] = 1

    grouped = df.groupby("user_id", observed=True)
    base = grouped.agg(
        transaction_count=("amount", "size"),
        outflow_transaction_count=("is_outflow", "sum"),
        inflow_transaction_count=("is_inflow", "sum"),
        outflow_amount=("outflow", "sum"),
        inflow_amount=("inflow", "sum"),
        signed_amount_sum=("amount", "sum"),
        amount_abs_sum=("amount", lambda values: values.abs().sum()),
        amount_square_sum=("amount_square", "sum"),
        max_outflow_amount=("outflow", "max"),
        max_inflow_amount=("inflow", "max"),
        card_not_present_count=("card_not_present_tx", "sum"),
        card_present_count=("card_present_tx", "sum"),
        weekend_transaction_count=("weekend_tx", "sum"),
        night_transaction_count=("night_tx", "sum"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
    )

    pfm_spend = _wide_group_sum(df, "pfm_category", "outflow", "pfm_spend__", PFM_CATEGORIES)
    pfm_count = _wide_group_sum(df, "pfm_category", "tx_marker", "pfm_tx__", PFM_CATEGORIES)
    merchant_spend = _wide_group_sum(
        df[df["merchant_type_norm"].isin(top_merchant_types)],
        "merchant_type_norm",
        "outflow",
        "merchant_spend__",
        top_merchant_types,
    )

    result = pd.concat([base, pfm_spend, pfm_count, merchant_spend], axis=1)
    numeric_columns = result.select_dtypes(include=["number"]).columns
    result[numeric_columns] = result[numeric_columns].fillna(0.0)
    return result


def _wide_group_sum(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    prefix: str,
    allowed_values: Iterable[str],
) -> pd.DataFrame:
    allowed = list(allowed_values)
    columns = [prefix + slugify(value) for value in allowed]

    if df.empty:
        return pd.DataFrame(columns=columns, dtype="float64")

    wide = (
        df.groupby(["user_id", category_column], observed=True)[value_column]
        .sum()
        .unstack(fill_value=0.0)
    )

    rename = {value: prefix + slugify(value) for value in wide.columns}
    wide = wide.rename(columns=rename)
    for column in columns:
        if column not in wide.columns:
            wide[column] = 0.0
    return wide[columns]


def _combine_partials(partials: Sequence[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(partials, axis=0, sort=False)
    timestamp_columns = ["first_timestamp", "last_timestamp"]
    numeric = data.drop(columns=timestamp_columns, errors="ignore").fillna(0.0)
    combined = numeric.groupby(level=0, observed=True).sum()

    if "first_timestamp" in data:
        combined["first_timestamp"] = data.groupby(level=0, observed=True)["first_timestamp"].min()
    if "last_timestamp" in data:
        combined["last_timestamp"] = data.groupby(level=0, observed=True)["last_timestamp"].max()
    return combined


def _safe_divide(
    numerator: pd.Series,
    denominator: Union[pd.Series, float],
    fill_value: float = 0.0,
):
    denominator = pd.Series(denominator, index=numerator.index) if np.isscalar(denominator) else denominator
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _add_derived_features(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()

    transaction_count = result["transaction_count"].replace(0, np.nan)
    result["avg_signed_transaction_amount"] = result["signed_amount_sum"] / transaction_count
    result["avg_abs_transaction_amount"] = result["amount_abs_sum"] / transaction_count
    variance = result["amount_square_sum"] / transaction_count - (
        result["avg_signed_transaction_amount"] ** 2
    )
    result["std_signed_transaction_amount"] = np.sqrt(variance.clip(lower=0.0)).fillna(0.0)

    span_seconds = (
        result["last_timestamp"] - result["first_timestamp"]
    ).dt.total_seconds()
    result["observation_days"] = (span_seconds / 86400.0).fillna(0.0).clip(lower=1.0)
    result["observation_months"] = (result["observation_days"] / 30.0).clip(lower=1.0 / 30.0)
    result["transactions_per_day"] = result["transaction_count"] / result["observation_days"]
    result["outflow_per_day"] = result["outflow_amount"] / result["observation_days"]
    result["inflow_per_day"] = result["inflow_amount"] / result["observation_days"]
    result["monthly_outflow_estimate"] = result["outflow_amount"] / result["observation_months"]
    result["monthly_inflow_estimate"] = result["inflow_amount"] / result["observation_months"]

    result["net_cashflow_amount"] = result["inflow_amount"] - result["outflow_amount"]
    result["monthly_net_cashflow_estimate"] = (
        result["monthly_inflow_estimate"] - result["monthly_outflow_estimate"]
    )
    result["outflow_to_inflow_ratio"] = _safe_divide(
        result["outflow_amount"], result["inflow_amount"], fill_value=999.0
    )
    result["cashflow_margin"] = _safe_divide(
        result["net_cashflow_amount"], result["inflow_amount"], fill_value=-1.0
    )
    result["card_not_present_share"] = _safe_divide(
        result["card_not_present_count"], result["transaction_count"]
    )
    result["weekend_transaction_share"] = _safe_divide(
        result["weekend_transaction_count"], result["transaction_count"]
    )
    result["night_transaction_share"] = _safe_divide(
        result["night_transaction_count"], result["transaction_count"]
    )
    result["inflow_transaction_share"] = _safe_divide(
        result["inflow_transaction_count"], result["transaction_count"]
    )

    pfm_spend_columns = [column for column in result.columns if column.startswith("pfm_spend__")]
    merchant_spend_columns = [
        column for column in result.columns if column.startswith("merchant_spend__")
    ]

    for column in pfm_spend_columns:
        share_column = "pfm_share__" + column.removeprefix("pfm_spend__")
        result[share_column] = _safe_divide(result[column], result["outflow_amount"])

    for column in merchant_spend_columns:
        share_column = "merchant_share__" + column.removeprefix("merchant_spend__")
        result[share_column] = _safe_divide(result[column], result["outflow_amount"])

    discretionary_columns = [
        "pfm_spend__" + category for category in sorted(DISCRETIONARY_CATEGORIES)
    ]
    essential_columns = ["pfm_spend__" + category for category in sorted(ESSENTIAL_CATEGORIES)]
    discretionary_existing = [column for column in discretionary_columns if column in result]
    essential_existing = [column for column in essential_columns if column in result]

    result["discretionary_outflow"] = result[discretionary_existing].sum(axis=1)
    result["essential_outflow"] = result[essential_existing].sum(axis=1)
    result["discretionary_share"] = _safe_divide(
        result["discretionary_outflow"], result["outflow_amount"]
    )
    result["essential_share"] = _safe_divide(result["essential_outflow"], result["outflow_amount"])
    result["largest_pfm_category_share"] = (
        result[pfm_spend_columns].max(axis=1) / result["outflow_amount"].replace(0, np.nan)
    ).fillna(0.0)
    result["pfm_category_count"] = (result[pfm_spend_columns] > 0).sum(axis=1)

    result = result.drop(columns=["first_timestamp", "last_timestamp"], errors="ignore")
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return result.astype("float64")
