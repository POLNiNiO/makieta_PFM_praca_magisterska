const state = {
  profile: null,
  categories: [],
  metrics: {},
  disclaimer: "",
};

const moneyFormatter = new Intl.NumberFormat("pl-PL", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const percentFormatter = new Intl.NumberFormat("pl-PL", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const els = {};

document.addEventListener("DOMContentLoaded", async () => {
  cacheElements();
  bindEvents();
  setToday();
  await loadBootstrap();
});

function cacheElements() {
  [
    "modelTask",
    "metricAuc",
    "metricRecall",
    "metricThreshold",
    "refreshButton",
    "resetButton",
    "transactionForm",
    "transactionType",
    "categoryForm",
    "assetForm",
    "liabilityForm",
    "categoryOptions",
    "riskGauge",
    "riskValue",
    "riskLabel",
    "riskBadge",
    "riskSummary",
    "cashflowChart",
    "monthlyInflow",
    "monthlyOutflow",
    "monthlyNet",
    "inflowCaption",
    "outflowCaption",
    "marginCaption",
    "recommendationList",
    "disclaimer",
    "categoryList",
    "categoryTotal",
    "transactionTable",
    "assetsTotal",
    "liabilitiesTotal",
    "netWorth",
    "bufferMonths",
    "assetList",
    "liabilityList",
    "toast",
  ].forEach((id) => {
    els[id] = document.getElementById(id);
  });
}

function bindEvents() {
  els.refreshButton.addEventListener("click", loadProfile);
  els.transactionType.addEventListener("change", renderCategoryOptions);
  els.resetButton.addEventListener("click", async () => {
    const confirmed = window.confirm("Wyczyścić wszystkie wpisane dane w lokalnym profilu?");
    if (!confirmed) return;
    await postJson("/api/reset", {});
    showToast("Profil został wyczyszczony.");
    await loadBootstrap();
  });

  bindForm(els.transactionForm, "/api/transactions", "Dodano transakcję.");
  bindForm(els.categoryForm, "/api/categories", "Dodano kategorię.");
  bindForm(els.assetForm, "/api/assets", "Dodano aktywo.");
  bindForm(els.liabilityForm, "/api/liabilities", "Dodano zobowiązanie.");
  bindMoneyInputs();
}

function bindForm(form, endpoint, successMessage) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(form).entries());
    try {
      const profilePayload = await postJson(endpoint, payload);
      form.reset();
      setToday();
      showToast(successMessage);
      if (endpoint === "/api/categories") {
        await loadBootstrap();
      } else {
        renderProfile(profilePayload);
      }
    } catch (error) {
      showToast(error.message);
    }
  });
}

function setToday() {
  const dateInput = els.transactionForm.querySelector('input[name="date"]');
  if (dateInput && !dateInput.value) {
    dateInput.value = new Date().toISOString().slice(0, 10);
  }
}

function bindMoneyInputs() {
  document.querySelectorAll("[data-money]").forEach((input) => {
    input.addEventListener("focus", () => {
      input.value = normalizeMoneyForEditing(input.value);
      input.select();
    });
    input.addEventListener("blur", () => {
      const value = parseMoney(input.value);
      input.value = value === null ? "" : formatMoneyInput(value);
    });
  });
}

async function loadBootstrap() {
  setLoading(true);
  try {
    const response = await fetch("/api/bootstrap");
    if (!response.ok) throw new Error("Nie udało się wczytać dashboardu.");
    const payload = await response.json();
    state.metrics = payload.metrics || {};
    state.categories = payload.categories || [];
    state.disclaimer = payload.disclaimer || "";
    renderMetrics();
    renderCategoryOptions();
    renderProfile(payload.profile);
  } catch (error) {
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

async function loadProfile() {
  setLoading(true);
  try {
    const response = await fetch("/api/profile");
    if (!response.ok) throw new Error("Nie udało się odświeżyć profilu.");
    renderProfile(await response.json());
  } catch (error) {
    showToast(error.message);
  } finally {
    setLoading(false);
  }
}

async function postJson(endpoint, payload) {
  const response = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(result.error || "Operacja nie powiodła się.");
  return result;
}

async function deleteItem(collection, id) {
  const response = await fetch(`/api/${collection}/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  const result = await response.json().catch(() => ({}));
  if (!response.ok) {
    showToast(result.error || "Nie udało się usunąć wpisu.");
    return;
  }
  renderProfile(result);
}

function renderMetrics() {
  els.modelTask.textContent = formatTaskName(state.metrics.task);
  els.metricAuc.textContent = number(state.metrics.roc_auc, 3);
  els.metricRecall.textContent = percent(state.metrics.recall_risk);
  els.metricThreshold.textContent = number(state.metrics.decision_threshold, 3);
}

function renderCategoryOptions() {
  const type = els.transactionType?.value || "expense";
  const categories = state.categories.filter((category) => {
    const scope = category.scope || "both";
    return scope === type || scope === "both";
  });
  els.categoryOptions.innerHTML = categories
    .map((category) => `<option value="${escapeHtml(category.label)}"></option>`)
    .join("");
}

function renderProfile(payload) {
  state.profile = payload;
  const data = payload?.data || {};
  const analysis = payload?.analysis || {};
  const kpis = analysis.kpis || {};
  const signals = analysis.signals || {};

  renderRisk(analysis);
  renderKpis(kpis, signals);
  renderRecommendations(analysis.recommendations || []);
  renderCategories(analysis.categories || []);
  renderTransactions(data.transactions || []);
  renderBalance(data.assets || [], data.liabilities || [], kpis);
}

function renderRisk(analysis) {
  const hasScore = analysis.risk_probability !== null && analysis.risk_probability !== undefined;
  const risk = hasScore ? clamp(Number(analysis.risk_probability), 0, 1) : 0;
  const info = getRiskInfo(risk, Boolean(analysis.risk_label), hasScore);
  els.riskValue.textContent = hasScore ? percent(risk, 0) : "-";
  els.riskLabel.textContent = hasScore ? "ryzyko" : "model";
  els.riskGauge.style.background = `conic-gradient(${info.color} ${risk * 360}deg, var(--line) 0deg)`;
  els.riskBadge.className = `badge ${info.level}`;
  els.riskBadge.textContent = info.label;
  els.riskSummary.textContent = analysis.model_message || info.summary;
  renderCashflowChart(analysis.kpis || {});
}

function renderKpis(kpis, signals) {
  els.monthlyInflow.textContent = money(kpis.monthly_inflow);
  els.monthlyOutflow.textContent = money(kpis.monthly_outflow);
  els.monthlyNet.textContent = money(kpis.monthly_net);
  els.inflowCaption.textContent = `${money(kpis.total_inflow)} wpisanych wpływów`;
  els.outflowCaption.textContent = `${money(kpis.total_outflow)} wpisanych wydatków`;
  els.marginCaption.textContent = `Margines: ${percent(signals.cashflow_margin)}`;
  els.assetsTotal.textContent = money(kpis.assets_total);
  els.liabilitiesTotal.textContent = money(kpis.liabilities_total);
  els.netWorth.textContent = money(kpis.net_worth);
  els.bufferMonths.textContent = `${number(kpis.buffer_months, 1)} mies.`;
}

function renderRecommendations(items) {
  els.disclaimer.textContent = state.disclaimer;
  if (!items.length) {
    els.recommendationList.innerHTML = emptyState("Dodaj dane, aby otrzymać tipy.");
    return;
  }
  els.recommendationList.innerHTML = items
    .map((item) => {
      const amount = item.suggested_amount
        ? `<div class="recommendation-amount">${money(item.suggested_amount)}</div>`
        : "";
      const icon = item.priority === "critical" || item.priority === "high" ? "icon-alert" : "icon-wallet";
      return `
        <article class="recommendation ${escapeHtml(item.priority || "low")}">
          <div class="recommendation-icon"><svg aria-hidden="true"><use href="#${icon}"></use></svg></div>
          <div>
            <h3>${escapeHtml(item.title || "")}</h3>
            <p>${escapeHtml(item.message || "")}</p>
            ${item.rationale ? `<small>${escapeHtml(item.rationale)}</small>` : ""}
          </div>
          ${amount}
        </article>
      `;
    })
    .join("");
}

function renderCategories(categories) {
  const visible = categories.filter((category) => category.amount > 0).slice(0, 8);
  const total = visible.reduce((sum, category) => sum + Number(category.amount || 0), 0);
  els.categoryTotal.textContent = money(total);
  if (!visible.length) {
    els.categoryList.innerHTML = emptyState("Brak wydatków do pokazania.");
    return;
  }
  const maxAmount = Math.max(...visible.map((category) => Number(category.amount || 0)), 1);
  els.categoryList.innerHTML = visible
    .map((category) => {
      const width = clamp((Number(category.amount) / maxAmount) * 100, 4, 100);
      return `
        <div class="category-row">
          <div class="category-meta">
            <strong>${escapeHtml(category.label)}</strong>
            <span>${money(category.amount)} · ${percent(category.share)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
        </div>
      `;
    })
    .join("");
}

function renderTransactions(transactions) {
  if (!transactions.length) {
    els.transactionTable.innerHTML = `<tr><td colspan="5">${emptyText("Brak transakcji.")}</td></tr>`;
    return;
  }
  els.transactionTable.innerHTML = transactions
    .slice()
    .reverse()
    .map((transaction) => {
      const sign = transaction.type === "income" ? "+" : "-";
      return `
        <tr>
          <td>${escapeHtml(transaction.date)}</td>
          <td>${transaction.type === "income" ? "Przychód" : "Wydatek"}</td>
          <td>${escapeHtml(transaction.category)}</td>
          <td class="${transaction.type === "income" ? "positive" : "negative"}">${sign}${money(transaction.amount)}</td>
          <td><button class="table-action" data-delete="transactions" data-id="${transaction.id}" aria-label="Usuń"><svg><use href="#icon-trash"></use></svg></button></td>
        </tr>
      `;
    })
    .join("");
  bindDeleteButtons(els.transactionTable);
}

function renderBalance(assets, liabilities) {
  els.assetList.innerHTML = assets.length
    ? assets.map((asset) => listItem("assets", asset.id, asset.name, asset.type, asset.amount)).join("")
    : emptyState("Brak aktywów.");
  els.liabilityList.innerHTML = liabilities.length
    ? liabilities
        .map((liability) =>
          listItem(
            "liabilities",
            liability.id,
            liability.name,
            `${liability.type} · rata ${money(liability.monthly_payment)} · ${number(liability.interest_rate, 2)}%`,
            liability.balance,
          ),
        )
        .join("")
    : emptyState("Brak zobowiązań.");
  bindDeleteButtons(els.assetList);
  bindDeleteButtons(els.liabilityList);
}

function listItem(collection, id, title, subtitle, amount) {
  return `
    <div class="stack-item">
      <div>
        <strong>${escapeHtml(title)}</strong>
        <span>${escapeHtml(subtitle || "")}</span>
      </div>
      <div class="stack-side">
        <strong>${money(amount)}</strong>
        <button class="table-action" data-delete="${collection}" data-id="${id}" aria-label="Usuń"><svg><use href="#icon-trash"></use></svg></button>
      </div>
    </div>
  `;
}

function bindDeleteButtons(container) {
  container.querySelectorAll("[data-delete]").forEach((button) => {
    button.addEventListener("click", () => deleteItem(button.dataset.delete, button.dataset.id));
  });
}

function renderCashflowChart(kpis) {
  const inflow = Number(kpis.monthly_inflow || 0);
  const outflow = Number(kpis.monthly_outflow || 0);
  const net = Number(kpis.monthly_net || 0);
  const maxAbs = Math.max(Math.abs(inflow), Math.abs(outflow), Math.abs(net), 1);
  const bars = [
    { label: "Wpływy", value: inflow, color: "var(--green)" },
    { label: "Wydatki", value: outflow, color: "var(--amber)" },
    { label: "Netto", value: net, color: net >= 0 ? "var(--teal)" : "var(--red)" },
  ];
  els.cashflowChart.innerHTML = `
    <svg viewBox="0 0 420 72" role="img" aria-label="Wpływy, wydatki i wynik netto">
      <line x1="16" y1="52" x2="404" y2="52" stroke="#dce4e7" />
      ${bars
        .map((bar, index) => {
          const x = 34 + index * 132;
          const height = Math.max((Math.abs(bar.value) / maxAbs) * 44, 3);
          const y = bar.value >= 0 ? 52 - height : 52;
          return `<rect x="${x}" y="${y}" width="58" height="${height}" rx="4" fill="${bar.color}" /><text x="${x}" y="67" fill="#607080" font-size="10">${bar.label}</text>`;
        })
        .join("")}
    </svg>
  `;
}

function getRiskInfo(risk, label, hasScore) {
  if (!hasScore) {
    return {
      level: "neutral",
      label: "Brak scoringu",
      color: "var(--line)",
      summary: "Dodaj transakcje, żeby model mógł ocenić ryzyko.",
    };
  }
  if (risk >= 0.65 || label) {
    return {
      level: "critical",
      label: "Wysokie ryzyko",
      color: "var(--red)",
      summary: "Model widzi podwyższone ryzyko problemów z płynnością.",
    };
  }
  if (risk >= 0.3) {
    return {
      level: "medium",
      label: "Podwyższone ryzyko",
      color: "var(--amber)",
      summary: "Budżet wymaga monitorowania i kontroli większych kategorii.",
    };
  }
  return {
    level: "low",
    label: "Niskie ryzyko",
    color: "var(--teal)",
    summary: "Model nie widzi obecnie silnego sygnału ryzyka.",
  };
}

function formatTaskName(task) {
  if (task === "insolvency_prediction_1months") return "Ryzyko 1 mies.";
  if (task === "insolvency_prediction_3months") return "Ryzyko 3 mies.";
  return task || "Model PFM";
}

function setLoading(isLoading) {
  document.body.classList.toggle("loading", isLoading);
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("visible");
  window.clearTimeout(showToast.timeout);
  showToast.timeout = window.setTimeout(() => els.toast.classList.remove("visible"), 3200);
}

function money(value) {
  const numeric = Number(value || 0);
  return `${moneyFormatter.format(numeric)} zł`;
}

function formatMoneyInput(value) {
  return `${moneyFormatter.format(Number(value || 0))} zł`;
}

function normalizeMoneyForEditing(value) {
  const parsed = parseMoney(value);
  if (parsed === null) return "";
  return parsed.toFixed(2).replace(".", ",");
}

function parseMoney(value) {
  let text = String(value || "").trim().toLowerCase();
  text = text.replaceAll("zł", "").replaceAll("pln", "").replaceAll("\u00a0", " ");
  text = text.replace(/\s+/g, "");
  if (!text) return null;
  if (text.includes(",") && text.includes(".")) {
    text = text.replaceAll(".", "").replace(",", ".");
  } else {
    text = text.replace(",", ".");
  }
  text = text.replace(/[^0-9.-]/g, "");
  if (!text) return null;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? numeric : null;
}

function percent(value, digits = 1) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return new Intl.NumberFormat("pl-PL", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
}

function number(value, digits = 2) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(digits);
}

function clamp(value, min, max) {
  return Math.min(Math.max(Number(value), min), max);
}

function emptyState(text) {
  return `<div class="empty-state">${escapeHtml(text)}</div>`;
}

function emptyText(text) {
  return `<span class="muted">${escapeHtml(text)}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
