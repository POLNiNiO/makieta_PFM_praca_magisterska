"""Local PFM dashboard server.

The dashboard stores user-entered data locally and uses the already trained
`pfm_ml` algorithm for insolvency-risk scoring. The model package itself is not
modified by this server.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import tempfile
import uuid
from datetime import date, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlparse

os.environ.setdefault("ARROW_USER_SIMD_LEVEL", "NONE")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pfm_ml.features import FeatureConfig, build_features  # noqa: E402
from pfm_ml.recommender import generate_recommendations  # noqa: E402

MODEL_DIR = PROJECT_ROOT / "models" / "pfm_insolvency_1m"
PROFILE_PATH = ROOT / "data" / "user_profile.json"

DEFAULT_CATEGORIES = [
    {"slug": "housing", "label": "Mieszkanie", "merchant_type": "Monthly Rent", "scope": "expense"},
    {"slug": "utilities", "label": "Rachunki i media", "merchant_type": "Electricity Bill", "scope": "expense"},
    {"slug": "groceries", "label": "Zakupy spożywcze", "merchant_type": "grocery", "scope": "expense"},
    {"slug": "dining", "label": "Restauracje", "merchant_type": "dining", "scope": "expense"},
    {"slug": "transport", "label": "Transport", "merchant_type": "gas", "scope": "expense"},
    {"slug": "healthcare", "label": "Zdrowie", "merchant_type": "medical", "scope": "expense"},
    {"slug": "insurance", "label": "Ubezpieczenia", "merchant_type": "insurance", "scope": "expense"},
    {"slug": "taxes", "label": "Podatki i opłaty", "merchant_type": "tax payment", "scope": "expense"},
    {"slug": "education", "label": "Edukacja", "merchant_type": "education", "scope": "expense"},
    {"slug": "shopping", "label": "Zakupy detaliczne", "merchant_type": "retail", "scope": "expense"},
    {"slug": "entertainment", "label": "Rozrywka", "merchant_type": "entertainment", "scope": "expense"},
    {"slug": "subscriptions", "label": "Subskrypcje", "merchant_type": "Streaming Service", "scope": "expense"},
    {"slug": "travel", "label": "Podróże", "merchant_type": "travel", "scope": "expense"},
    {"slug": "pets", "label": "Zwierzęta", "merchant_type": "veterinary care", "scope": "expense"},
    {"slug": "gifts_donations", "label": "Prezenty i darowizny", "merchant_type": "gift", "scope": "expense"},
    {"slug": "debt_payments", "label": "Spłaty zadłużenia", "merchant_type": "payment", "scope": "expense"},
    {"slug": "salary", "label": "Praca / pensja", "merchant_type": "salary", "scope": "income"},
    {"slug": "freelance", "label": "Freelance / zlecenie", "merchant_type": "salary", "scope": "income"},
    {"slug": "business_income", "label": "Działalność gospodarcza", "merchant_type": "salary", "scope": "income"},
    {"slug": "interest_income", "label": "Odsetki", "merchant_type": "interest", "scope": "income"},
    {"slug": "dividends", "label": "Dywidendy", "merchant_type": "investment income", "scope": "income"},
    {"slug": "rental_income", "label": "Najem", "merchant_type": "rental income", "scope": "income"},
    {"slug": "refund", "label": "Zwrot", "merchant_type": "refund", "scope": "income"},
    {"slug": "benefits", "label": "Świadczenia", "merchant_type": "benefits", "scope": "income"},
    {"slug": "gift_income", "label": "Prezent / wsparcie", "merchant_type": "gift", "scope": "income"},
    {"slug": "other_income", "label": "Inny przychód", "merchant_type": "salary", "scope": "income"},
    {"slug": "other", "label": "Inne", "merchant_type": "retail", "scope": "both"},
]


class ProfileStore:
    def __init__(self) -> None:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not PROFILE_PATH.exists():
            self.save(empty_profile())
        self.bundle = joblib.load(MODEL_DIR / "model.joblib")
        self.model = self.bundle["model"]
        self.feature_columns = list(self.bundle["feature_columns"])
        self.threshold = float(self.bundle["decision_threshold"])
        self.feature_config = FeatureConfig.from_dict(self.bundle["feature_config"])

    def load(self) -> Dict[str, Any]:
        return normalize_profile(json.loads(PROFILE_PATH.read_text()))

    def save(self, profile: Dict[str, Any]) -> None:
        PROFILE_PATH.write_text(json.dumps(normalize_profile(profile), ensure_ascii=False, indent=2))

    def bootstrap(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics(),
            "profile": self.profile_payload(),
            "categories": self.categories(),
            "disclaimer": (
                "Friendly tips mają charakter edukacyjny i informacyjny. "
                "Nie są poradą inwestycyjną ani rekomendacją zakupu konkretnego produktu."
            ),
        }

    def profile_payload(self) -> Dict[str, Any]:
        profile = self.load()
        return {
            "data": profile,
            "analysis": self.analyze(profile),
        }

    def metrics(self) -> Dict[str, Any]:
        metrics = dict(self.bundle.get("metrics", {}))
        report = metrics.get("classification_report", {})
        risk_report = report.get("1", {}) if isinstance(report, dict) else {}
        return {
            "task": self.bundle.get("task", "insolvency_prediction_1months"),
            "roc_auc": metrics.get("roc_auc"),
            "average_precision": metrics.get("average_precision"),
            "decision_threshold": self.threshold,
            "precision_risk": risk_report.get("precision"),
            "recall_risk": risk_report.get("recall"),
            "feature_count": metrics.get("feature_count"),
        }

    def categories(self) -> List[Dict[str, str]]:
        profile = self.load()
        by_slug = {category["slug"]: category for category in DEFAULT_CATEGORIES}
        for category in profile.get("custom_categories", []):
            by_slug[category["slug"]] = category
        return sorted(by_slug.values(), key=lambda item: item["label"].lower())

    def add_category(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        label = clean_text(payload.get("label"))
        if not label:
            raise ValueError("Nazwa kategorii jest wymagana.")
        scope = clean_text(payload.get("scope")).lower() or "expense"
        if scope not in {"expense", "income", "both"}:
            raise ValueError("Zakres kategorii musi być: expense, income albo both.")
        merchant_type = clean_text(payload.get("merchant_type")) or infer_merchant_type(label, scope)
        category = {
            "slug": slugify(label),
            "label": label,
            "merchant_type": merchant_type,
            "scope": scope,
        }
        profile = self.load()
        existing = {item["slug"] for item in self.categories()}
        if category["slug"] not in existing:
            profile["custom_categories"].append(category)
            self.save(profile)
        return category

    def add_transaction(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tx_type = clean_text(payload.get("type")).lower()
        if tx_type not in {"expense", "income"}:
            raise ValueError("Typ transakcji musi być expense albo income.")
        amount = positive_float(payload.get("amount"), "Kwota")
        category_label = clean_text(payload.get("category")) or (
            "Praca / pensja" if tx_type == "income" else "Inne"
        )
        date_value = parse_date(payload.get("date"))
        merchant_type = merchant_for_category(category_label, self.categories(), tx_type)

        transaction = {
            "id": new_id(),
            "type": tx_type,
            "date": date_value,
            "category": category_label,
            "merchant_name": clean_text(payload.get("merchant_name")) or category_label,
            "merchant_type": merchant_type,
            "channel": clean_text(payload.get("channel")) or "card",
            "amount": amount,
            "note": clean_text(payload.get("note")),
            "created_at": utc_now(),
        }

        profile = self.load()
        self.ensure_category(category_label, merchant_type, tx_type, profile)
        profile["transactions"].append(transaction)
        self.save(profile)
        return transaction

    def add_asset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        asset = {
            "id": new_id(),
            "name": clean_text(payload.get("name")) or "Aktywo",
            "type": clean_text(payload.get("type")) or "cash",
            "amount": positive_float(payload.get("amount"), "Wartość aktywa"),
            "liquidity": clean_text(payload.get("liquidity")) or "high",
            "created_at": utc_now(),
        }
        profile = self.load()
        profile["assets"].append(asset)
        self.save(profile)
        return asset

    def add_liability(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        liability = {
            "id": new_id(),
            "name": clean_text(payload.get("name")) or "Zobowiązanie",
            "type": clean_text(payload.get("type")) or "loan",
            "balance": positive_float(payload.get("balance"), "Saldo zobowiązania"),
            "monthly_payment": positive_float(payload.get("monthly_payment", 0), "Rata"),
            "interest_rate": non_negative_float(payload.get("interest_rate", 0), "Oprocentowanie"),
            "created_at": utc_now(),
        }
        profile = self.load()
        profile["liabilities"].append(liability)
        self.save(profile)
        return liability

    def delete_item(self, collection: str, item_id: str) -> bool:
        if collection not in {"transactions", "assets", "liabilities"}:
            raise ValueError("Nieobsługiwany typ danych.")
        profile = self.load()
        before = len(profile[collection])
        profile[collection] = [item for item in profile[collection] if item.get("id") != item_id]
        changed = len(profile[collection]) != before
        if changed:
            self.save(profile)
        return changed

    def reset(self) -> None:
        self.save(empty_profile())

    def analyze(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        features = self.model_features(profile)
        kpis = self.kpis(profile, features)
        if features is None:
            return {
                "status": "empty",
                "risk_probability": None,
                "risk_label": False,
                "kpis": kpis,
                "signals": self.signals(profile, None, kpis),
                "categories": self.categories_from_transactions(profile),
                "recommendations": self.friendly_tips(profile, None, kpis),
                "model_message": missing_data_message(profile),
            }

        frame = pd.DataFrame([features]).reindex(columns=self.feature_columns, fill_value=0.0)
        probability = float(self.model.predict_proba(frame)[0, 1])
        model_recommendations = generate_recommendations(features, probability, self.threshold)
        friendly_tips = self.friendly_tips(profile, probability, kpis)
        return {
            "status": "scored",
            "risk_probability": probability,
            "risk_label": bool(probability >= self.threshold),
            "kpis": kpis,
            "signals": self.signals(profile, features, kpis),
            "categories": self.categories_from_transactions(profile),
            "recommendations": model_recommendations + friendly_tips,
            "model_message": "Scoring oparty o Twoje wpisane transakcje.",
        }

    def model_features(self, profile: Dict[str, Any]) -> Optional[Dict[str, float]]:
        transactions = profile.get("transactions", [])
        if not transactions:
            return None
        if not any(transaction.get("type") == "expense" for transaction in transactions):
            return None

        rows = []
        for transaction in transactions:
            tx_type = transaction.get("type", "expense")
            amount = float(transaction.get("amount", 0.0))
            signed_amount = amount if tx_type == "expense" else -amount
            rows.append(
                {
                    "timestamp": f"{transaction.get('date') or date.today().isoformat()} 12:00:00",
                    "merchant_name": transaction.get("merchant_name") or transaction.get("category"),
                    "merchant_type": transaction.get("merchant_type")
                    or infer_merchant_type(transaction.get("category"), tx_type),
                    "card_present_or_not": channel_to_card_value(transaction.get("channel")),
                    "amount": signed_amount,
                    "seq_id": 0,
                }
            )
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory(prefix="pfm_dashboard_") as tmpdir:
            transaction_path = Path(tmpdir) / "transactions.parquet"
            df.to_parquet(transaction_path, index=False)
            features = build_features(transaction_path, config=self.feature_config)
        if features.empty:
            return None
        profile_features = {column: float(value) for column, value in features.iloc[0].items()}
        override_monthly_feature_estimates(profile, profile_features)
        return profile_features

    def kpis(self, profile: Dict[str, Any], features: Optional[Dict[str, float]]) -> Dict[str, float]:
        assets_total = sum(float(item.get("amount", 0.0)) for item in profile.get("assets", []))
        liquid_assets = sum(
            float(item.get("amount", 0.0))
            for item in profile.get("assets", [])
            if item.get("liquidity", "high") == "high"
        )
        liabilities_total = sum(
            float(item.get("balance", 0.0)) for item in profile.get("liabilities", [])
        )
        monthly_debt_payment = sum(
            float(item.get("monthly_payment", 0.0)) for item in profile.get("liabilities", [])
        )

        if features:
            total_inflow = sum_transaction_amount(profile, "income")
            total_outflow = sum_transaction_amount(profile, "expense")
            monthly_inflow, monthly_outflow = monthly_estimates(profile)
            monthly_net = monthly_inflow - monthly_outflow
        else:
            total_inflow = sum_transaction_amount(profile, "income")
            total_outflow = sum_transaction_amount(profile, "expense")
            monthly_inflow, monthly_outflow = monthly_estimates(profile)
            monthly_net = monthly_inflow - monthly_outflow

        buffer_months = liquid_assets / monthly_outflow if monthly_outflow > 0 else 0.0
        debt_service_ratio = (
            monthly_debt_payment / monthly_inflow if monthly_inflow > 0 else 0.0
        )
        return {
            "total_inflow": round(total_inflow, 2),
            "total_outflow": round(total_outflow, 2),
            "monthly_inflow": round(monthly_inflow, 2),
            "monthly_outflow": round(monthly_outflow, 2),
            "monthly_net": round(monthly_net, 2),
            "assets_total": round(assets_total, 2),
            "liquid_assets": round(liquid_assets, 2),
            "liabilities_total": round(liabilities_total, 2),
            "net_worth": round(assets_total - liabilities_total, 2),
            "monthly_debt_payment": round(monthly_debt_payment, 2),
            "buffer_months": round(buffer_months, 2),
            "debt_service_ratio": round(debt_service_ratio, 4),
        }

    def signals(
        self,
        profile: Dict[str, Any],
        features: Optional[Dict[str, float]],
        kpis: Dict[str, float],
    ) -> Dict[str, float]:
        if features:
            base = {
                "cashflow_margin": float(features.get("cashflow_margin", 0.0)),
                "discretionary_share": float(features.get("discretionary_share", 0.0)),
                "transactions_per_day": float(features.get("transactions_per_day", 0.0)),
                "card_not_present_share": float(features.get("card_not_present_share", 0.0)),
                "outflow_to_inflow_ratio": float(features.get("outflow_to_inflow_ratio", 0.0)),
            }
        else:
            base = {
                "cashflow_margin": safe_divide(kpis["monthly_net"], kpis["monthly_inflow"]),
                "discretionary_share": 0.0,
                "transactions_per_day": 0.0,
                "card_not_present_share": 0.0,
                "outflow_to_inflow_ratio": safe_divide(
                    kpis["monthly_outflow"], kpis["monthly_inflow"]
                ),
            }
        base["buffer_months"] = float(kpis["buffer_months"])
        base["debt_service_ratio"] = float(kpis["debt_service_ratio"])
        base["transaction_count"] = len(profile.get("transactions", []))
        return base

    def categories_from_features(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        rows = []
        for category in DEFAULT_CATEGORIES:
            slug = category["slug"]
            if category.get("scope") == "income":
                continue
            amount = float(features.get("pfm_spend__" + slug, 0.0))
            share = float(features.get("pfm_share__" + slug, 0.0))
            rows.append({"category": slug, "label": category["label"], "amount": round(amount, 2), "share": share})
        return sorted(rows, key=lambda row: row["amount"], reverse=True)

    def categories_from_transactions(self, profile: Dict[str, Any]) -> List[Dict[str, Any]]:
        totals: Dict[str, float] = {}
        outflow = sum_transaction_amount(profile, "expense")
        for transaction in profile.get("transactions", []):
            if transaction.get("type") != "expense":
                continue
            label = transaction.get("category") or "Inne"
            totals[label] = totals.get(label, 0.0) + float(transaction.get("amount", 0.0))
        return [
            {
                "category": slugify(label),
                "label": label,
                "amount": round(amount, 2),
                "share": safe_divide(amount, outflow),
            }
            for label, amount in sorted(totals.items(), key=lambda item: item[1], reverse=True)
        ]

    def friendly_tips(
        self,
        profile: Dict[str, Any],
        risk_probability: Optional[float],
        kpis: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        tips: List[Dict[str, Any]] = []
        monthly_net = float(kpis["monthly_net"])
        monthly_outflow = float(kpis["monthly_outflow"])
        buffer_months = float(kpis["buffer_months"])

        if risk_probability is not None and risk_probability >= self.threshold:
            tips.append(tip("critical", "Friendly Tip: najpierw płynność", "Model widzi podwyższone ryzyko. Zanim pojawią się pomysły inwestycyjne, priorytetem jest gotówka, bufor i kontrola rat.", None))

        if monthly_net > 0 and monthly_outflow > 0 and buffer_months < 3:
            tips.append(
                tip(
                    "medium",
                    "Friendly Tip: automatyczny przelew na poduszkę",
                    "Masz dodatnią nadwyżkę, ale bufor jest krótszy niż 3 miesiące wydatków. Rozważ przelew części nadwyżki na konto oszczędnościowe albo lokatę krótkoterminową.",
                    min(monthly_net * 0.5, max(monthly_outflow * 3 - kpis["liquid_assets"], 0)),
                )
            )
        elif monthly_net > 0 and monthly_outflow > 0 and buffer_months >= 3:
            tips.append(
                tip(
                    "low",
                    "Friendly Tip: konserwatywna część portfela",
                    "Bufor wygląda zdrowo. Nadwyżkę możesz rozważyć w produktach o niskiej zmienności, np. lokacie bankowej lub obligacjach skarbowych, zależnie od horyzontu i akceptacji ryzyka.",
                    monthly_net * 0.4,
                )
            )

        if has_travel_signal(profile):
            tips.append(
                tip(
                    "low",
                    "Friendly Tip: sprawdź ubezpieczenie podróżne",
                    "W wydatkach pojawia się sygnał podróży. Jeżeli faktycznie planujesz wyjazd, sprawdź polisę podróżną i koszty leczenia za granicą.",
                    None,
                )
            )

        for liability in profile.get("liabilities", []):
            interest_rate = float(liability.get("interest_rate", 0.0))
            monthly_payment = float(liability.get("monthly_payment", 0.0))
            if interest_rate >= 10 or (monthly_payment > 0 and kpis["debt_service_ratio"] > 0.35):
                tips.append(
                    tip(
                        "medium",
                        "Friendly Tip: sprawdź refinansowanie zobowiązania",
                        f"Zobowiązanie „{liability.get('name', 'kredyt')}” wygląda na istotne dla budżetu. Warto porównać oferty refinansowania lub nadpłaty, szczególnie jeśli Twoje oprocentowanie jest wysokie.",
                        None,
                    )
                )
                break

        if kpis["net_worth"] < 0:
            tips.append(
                tip(
                    "high",
                    "Friendly Tip: plan redukcji zadłużenia",
                    "Suma zobowiązań przewyższa aktywa. Ustal kolejność spłat i unikaj zwiększania ryzykownych inwestycji przed ustabilizowaniem bilansu.",
                    None,
                )
            )

        return tips

    def ensure_category(
        self,
        label: str,
        merchant_type: str,
        scope: str = "both",
        profile: Optional[Dict[str, Any]] = None,
    ) -> None:
        profile = profile or self.load()
        slug = slugify(label)
        existing = {item["slug"] for item in DEFAULT_CATEGORIES + profile.get("custom_categories", [])}
        if slug not in existing:
            profile["custom_categories"].append(
                {
                    "slug": slug,
                    "label": label,
                    "merchant_type": merchant_type,
                    "scope": scope,
                }
            )


STORE = ProfileStore()


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            self.send_json(STORE.bootstrap())
            return
        if parsed.path == "/api/profile":
            self.send_json(STORE.profile_payload())
            return
        return super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            payload = self.read_json()
            if parsed.path == "/api/transactions":
                STORE.add_transaction(payload)
            elif parsed.path == "/api/assets":
                STORE.add_asset(payload)
            elif parsed.path == "/api/liabilities":
                STORE.add_liability(payload)
            elif parsed.path == "/api/categories":
                STORE.add_category(payload)
            elif parsed.path == "/api/reset":
                STORE.reset()
            else:
                self.send_json({"error": "Nieznany endpoint."}, status=404)
                return
            self.send_json(STORE.profile_payload())
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 3 and parts[0] == "api":
            collection = parts[1]
            item_id = unquote(parts[2])
            try:
                changed = STORE.delete_item(collection, item_id)
                if not changed:
                    self.send_json({"error": "Nie znaleziono wpisu."}, status=404)
                    return
                self.send_json(STORE.profile_payload())
            except ValueError as error:
                self.send_json({"error": str(error)}, status=400)
            return
        self.send_json({"error": "Nieznany endpoint."}, status=404)

    def read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length).decode("utf-8")
        return json.loads(raw or "{}")

    def translate_path(self, path: str) -> str:
        parsed = urlparse(path)
        requested = unquote(parsed.path.lstrip("/")) or "index.html"
        safe_path = Path(requested)
        if ".." in safe_path.parts:
            safe_path = Path("index.html")
        return str(ROOT / safe_path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def empty_profile() -> Dict[str, Any]:
    return {
        "transactions": [],
        "assets": [],
        "liabilities": [],
        "custom_categories": [],
    }


def normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    normalized = empty_profile()
    for key in normalized:
        if isinstance(profile.get(key), list):
            normalized[key] = profile[key]
    usage: Dict[str, set[str]] = {}
    for transaction in normalized["transactions"]:
        label_slug = slugify(transaction.get("category", ""))
        usage.setdefault(label_slug, set()).add(transaction.get("type", "expense"))
    for category in normalized["custom_categories"]:
        if "scope" not in category:
            category_usage = usage.get(category.get("slug") or slugify(category.get("label", "")), set())
            if category_usage == {"income"}:
                category["scope"] = "income"
            elif category_usage == {"expense"}:
                category["scope"] = "expense"
            else:
                category["scope"] = "both"
    return normalized


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def positive_float(value: Any, field_name: str) -> float:
    amount = non_negative_float(value, field_name)
    if amount <= 0:
        raise ValueError(f"{field_name} musi być większa od zera.")
    return amount


def non_negative_float(value: Any, field_name: str) -> float:
    try:
        amount = float(parse_money_text(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} musi być liczbą.") from exc
    if amount < 0:
        raise ValueError(f"{field_name} nie może być ujemna.")
    return round(amount, 2)


def parse_money_text(value: Any) -> str:
    text = str(value or "0").strip().lower()
    text = text.replace("zł", "").replace("pln", "").replace("\xa0", " ")
    text = re.sub(r"\s+", "", text)
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", ".")
    text = re.sub(r"[^0-9.\-]", "", text)
    return text or "0"


def parse_date(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return date.today().isoformat()
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except ValueError as exc:
        raise ValueError("Data musi mieć format RRRR-MM-DD.") from exc


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "custom"


def merchant_for_category(
    label: str,
    categories: Iterable[Dict[str, str]],
    tx_type: str,
) -> str:
    label_slug = slugify(label)
    for category in categories:
        category_scope = category.get("scope", "both")
        matches_type = category_scope in {tx_type, "both"}
        matches_label = slugify(category["label"]) == label_slug or category["slug"] == label_slug
        if matches_type and matches_label:
            return category.get("merchant_type") or infer_merchant_type(label, tx_type)
    return infer_merchant_type(label, tx_type)


def infer_merchant_type(label: Any, tx_type: str = "expense") -> str:
    if tx_type == "income":
        return "salary"
    text = clean_text(label).lower()
    if any(word in text for word in ("podró", "travel", "hotel", "lot", "wakac")):
        return "travel"
    if any(word in text for word in ("kredyt", "pożycz", "rata", "loan")):
        return "payment"
    if any(word in text for word in ("czynsz", "miesz", "rent", "mortgage")):
        return "Monthly Rent"
    if any(word in text for word in ("jedzenie", "restaur", "kawa", "dining")):
        return "dining"
    if any(word in text for word in ("spoż", "grocery", "sklep")):
        return "grocery"
    if any(word in text for word in ("ubez", "insurance")):
        return "insurance"
    if any(word in text for word in ("paliwo", "auto", "transport")):
        return "gas"
    return "retail"


def channel_to_card_value(value: Any) -> str:
    channel = clean_text(value).lower()
    if channel in {"online", "transfer", "subscription"}:
        return "no"
    if channel == "cash":
        return "none"
    return "yes"


def sum_transaction_amount(profile: Dict[str, Any], tx_type: str) -> float:
    return round(
        sum(
            float(item.get("amount", 0.0))
            for item in profile.get("transactions", [])
            if item.get("type") == tx_type
        ),
        2,
    )


def monthly_estimates(profile: Dict[str, Any]) -> tuple[float, float]:
    return (
        round(sum_transaction_amount(profile, "income"), 2),
        round(sum_transaction_amount(profile, "expense"), 2),
    )


def missing_data_message(profile: Dict[str, Any]) -> str:
    transactions = profile.get("transactions", [])
    has_income = any(transaction.get("type") == "income" for transaction in transactions)
    has_expense = any(transaction.get("type") == "expense" for transaction in transactions)
    if has_income and not has_expense:
        return "Dodaj przynajmniej jeden wydatek, aby uruchomić scoring modelu."
    if has_expense and not has_income:
        return "Dodaj przychód, aby model lepiej ocenił relację wydatków do wpływów."
    return "Dodaj przychód i wydatek, aby uruchomić scoring modelu."


def override_monthly_feature_estimates(
    profile: Dict[str, Any],
    features: Dict[str, float],
) -> None:
    monthly_inflow, monthly_outflow = monthly_estimates(profile)
    total_inflow = sum_transaction_amount(profile, "income")
    total_outflow = sum_transaction_amount(profile, "expense")
    net_cashflow = total_inflow - total_outflow

    features["inflow_amount"] = total_inflow
    features["outflow_amount"] = total_outflow
    features["net_cashflow_amount"] = net_cashflow
    features["monthly_inflow_estimate"] = monthly_inflow
    features["monthly_outflow_estimate"] = monthly_outflow
    features["monthly_net_cashflow_estimate"] = monthly_inflow - monthly_outflow
    features["observation_days"] = 30.0
    features["observation_months"] = 1.0
    features["transactions_per_day"] = len(profile.get("transactions", [])) / 30.0
    features["inflow_per_day"] = total_inflow / 30.0
    features["outflow_per_day"] = total_outflow / 30.0
    features["outflow_to_inflow_ratio"] = safe_divide(total_outflow, total_inflow) if total_inflow else 999.0
    features["cashflow_margin"] = safe_divide(net_cashflow, total_inflow) if total_inflow else -1.0


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def has_travel_signal(profile: Dict[str, Any]) -> bool:
    for transaction in profile.get("transactions", []):
        text = f"{transaction.get('category', '')} {transaction.get('merchant_name', '')}".lower()
        if any(word in text for word in ("podró", "travel", "hotel", "lot", "airline", "wakac")):
            return True
    return False


def tip(priority: str, title: str, message: str, amount: Optional[float]) -> Dict[str, Any]:
    return {
        "priority": priority,
        "kind": "friendly_tip",
        "title": title,
        "message": message,
        "category": None,
        "suggested_amount": round(amount, 2) if amount and amount > 0 else None,
        "rationale": "Friendly Tip, nie porada inwestycyjna.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PFM dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    mimetypes.add_type("image/svg+xml", ".svg")
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard is running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
