"""Rule layer that converts model scores into PFM recommendations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .features import DISCRETIONARY_CATEGORIES


CATEGORY_LABELS = {
    "housing": "mieszkanie",
    "utilities": "rachunki i media",
    "groceries": "zakupy spozywcze",
    "dining": "restauracje i jedzenie poza domem",
    "transport": "transport",
    "healthcare": "zdrowie",
    "insurance": "ubezpieczenia",
    "taxes": "podatki i oplaty publiczne",
    "education": "edukacja",
    "shopping": "zakupy detaliczne",
    "entertainment": "rozrywka",
    "subscriptions": "subskrypcje",
    "travel": "podroze",
    "pets": "zwierzeta",
    "gifts_donations": "prezenty i darowizny",
    "debt_payments": "splaty zadluzenia",
    "other": "pozostale wydatki",
}


@dataclass
class Recommendation:
    priority: str
    kind: str
    title: str
    message: str
    category: Optional[str] = None
    suggested_amount: Optional[float] = None
    rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def generate_recommendations(
    profile: Dict[str, float],
    risk_probability: float,
    decision_threshold: float = 0.5,
    max_items: int = 6,
) -> List[Dict[str, object]]:
    """Generate practical PFM recommendations from user features and risk score."""

    outflow = _get(profile, "outflow_amount")
    inflow = _get(profile, "inflow_amount")
    net_cashflow = _get(profile, "net_cashflow_amount")
    monthly_surplus = _get(profile, "monthly_net_cashflow_estimate")
    monthly_outflow = _get(profile, "monthly_outflow_estimate")
    cashflow_margin = _get(profile, "cashflow_margin")
    discretionary_outflow = _get(profile, "discretionary_outflow")
    outflow_to_inflow = _get(profile, "outflow_to_inflow_ratio")

    recommendations: List[Recommendation] = []

    if risk_probability >= max(0.65, decision_threshold):
        amount = _positive_amount(max(abs(net_cashflow), discretionary_outflow * 0.2, outflow * 0.05))
        recommendations.append(
            Recommendation(
                priority="critical",
                kind="liquidity_warning",
                title="Wysokie ryzyko utraty plynnosci",
                message=(
                    "Model widzi sygnaly zblizone do niewyplacalnosci. "
                    "Wstrzymaj wydatki uznaniowe i skup sie na dodatnim przeplywie gotowki."
                ),
                suggested_amount=amount,
                rationale=f"Prawdopodobienstwo ryzyka: {risk_probability:.1%}.",
            )
        )
    elif risk_probability >= max(0.30, decision_threshold * 0.75):
        amount = _positive_amount(max(discretionary_outflow * 0.15, outflow * 0.03))
        recommendations.append(
            Recommendation(
                priority="high",
                kind="liquidity_warning",
                title="Rosnace ryzyko problemow z plynnoscia",
                message=(
                    "Ustaw miesieczny limit wydatkow uznaniowych i przesun nadwyzke "
                    "na bufor bezpieczenstwa."
                ),
                suggested_amount=amount,
                rationale=f"Prawdopodobienstwo ryzyka: {risk_probability:.1%}.",
            )
        )

    if net_cashflow < 0:
        recommendations.append(
            Recommendation(
                priority="high",
                kind="budget_reduction",
                title="Zatrzymaj ujemny przeplyw gotowki",
                message=(
                    "Wydatki przewyzszaja wplywy w analizowanym okresie. "
                    "Najpierw zetnij kategorie elastyczne, zanim zwiekszy sie saldo zadluzenia."
                ),
                suggested_amount=_positive_amount(abs(net_cashflow)),
                rationale=f"Luka przeplywow: {_money(abs(net_cashflow))}.",
            )
        )
    elif inflow > 0 and (cashflow_margin < 0.05 or outflow_to_inflow > 0.95):
        recommendations.append(
            Recommendation(
                priority="medium",
                kind="cashflow_margin",
                title="Bardzo mala poduszka w miesiecznym budzecie",
                message=(
                    "Jestes blisko poziomu, w ktorym pojedynczy wiekszy wydatek moze "
                    "zaburzyc budzet. Zarezerwuj czesc wplywow na bufor."
                ),
                suggested_amount=_positive_amount(max(inflow * 0.05, monthly_outflow * 0.03)),
                rationale=f"Margines gotowkowy: {cashflow_margin:.1%}.",
            )
        )

    category, category_amount = _largest_discretionary_category(profile)
    if category and category_amount > 0:
        cut = _positive_amount(category_amount * (0.2 if risk_probability >= 0.30 else 0.1))
        recommendations.append(
            Recommendation(
                priority="medium" if risk_probability < 0.45 else "high",
                kind="category_saving",
                title=f"Ogranicz: {CATEGORY_LABELS.get(category, category)}",
                message=(
                    "To najwieksza elastyczna kategoria w profilu uzytkownika. "
                    "Ustaw limit i przenies zaoszczedzona kwote na bufor."
                ),
                category=category,
                suggested_amount=cut,
                rationale=f"Wydatki w kategorii: {_money(category_amount)}.",
            )
        )

    subscriptions = _get(profile, "pfm_spend__subscriptions")
    if subscriptions > max(50.0, outflow * 0.03):
        recommendations.append(
            Recommendation(
                priority="low",
                kind="subscription_review",
                title="Przejrzyj subskrypcje",
                message=(
                    "Subskrypcje sa dobrym kandydatem do szybkiej optymalizacji, "
                    "bo zwykle nie wymagaja zmiany podstawowego stylu zycia."
                ),
                category="subscriptions",
                suggested_amount=_positive_amount(subscriptions * 0.25),
                rationale=f"Suma subskrypcji: {_money(subscriptions)}.",
            )
        )

    if risk_probability < 0.25 and monthly_surplus > 100:
        reserve_amount = _positive_amount(monthly_surplus * 0.5)
        recommendations.append(
            Recommendation(
                priority="medium",
                kind="savings_or_investment",
                title="Zagospodaruj dodatnia nadwyzke",
                message=(
                    "Przekieruj czesc nadwyzki na poduszke finansowa. Gdy bufor jest juz "
                    "zbudowany, rozwaz lokate albo obligacje skarbowe jako konserwatywna czesc portfela."
                ),
                suggested_amount=reserve_amount,
                rationale=f"Szacowana miesieczna nadwyzka: {_money(monthly_surplus)}.",
            )
        )

    if risk_probability < 0.15 and monthly_surplus > 0 and monthly_outflow > 0:
        target_buffer = monthly_outflow * 3
        recommendations.append(
            Recommendation(
                priority="low",
                kind="emergency_fund",
                title="Buduj bufor 3 miesiecy wydatkow",
                message=(
                    "Profil wyglada stabilnie, wiec kolejnym celem moze byc pelniejsza "
                    "poduszka finansowa przed zwiekszaniem ryzyka inwestycyjnego."
                ),
                suggested_amount=_positive_amount(min(monthly_surplus * 0.4, target_buffer)),
                rationale=f"Cel bufora: ok. {_money(target_buffer)}.",
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                priority="low",
                kind="monitoring",
                title="Monitoruj budzet bez radykalnych zmian",
                message=(
                    "Model nie widzi silnego sygnalu ryzyka. Utrzymuj dodatni przeplyw "
                    "gotowki i kontroluj najwieksze kategorie wydatkow."
                ),
                rationale=f"Prawdopodobienstwo ryzyka: {risk_probability:.1%}.",
            )
        )

    return [recommendation.to_dict() for recommendation in _deduplicate(recommendations)[:max_items]]


def _get(profile: Dict[str, float], key: str) -> float:
    value = profile.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _largest_discretionary_category(profile: Dict[str, float]) -> tuple[Optional[str], float]:
    candidates = []
    for category in DISCRETIONARY_CATEGORIES:
        amount = _get(profile, "pfm_spend__" + category)
        candidates.append((category, amount))
    category, amount = max(candidates, key=lambda item: item[1])
    if amount <= 0:
        return None, 0.0
    return category, amount


def _positive_amount(value: float) -> Optional[float]:
    if value <= 0:
        return None
    return round(float(value), 2)


def _money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def _deduplicate(recommendations: List[Recommendation]) -> List[Recommendation]:
    seen = set()
    unique = []
    for recommendation in recommendations:
        key = (recommendation.kind, recommendation.category)
        if key in seen:
            continue
        seen.add(key)
        unique.append(recommendation)
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    return sorted(unique, key=lambda item: priority_order.get(item.priority, 9))
