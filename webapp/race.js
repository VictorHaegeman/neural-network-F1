const DATA_URL = "data/betting_guide_data.json";
const DEFAULT_POOL = 1000;

const state = {
  data: null,
  race: null,
  selectedDriverCode: null,
  strategy: "safe",
  market: "top10",
  oddsPolicy: "ask",
  slip: new Map(),
  offeredOdds: new Map(),
};

const strategyRules = {
  safe: { minProbability: 0.72, maxRank: 8, stakeMultiplier: 0.8 },
  balanced: { minProbability: 0.58, maxRank: 10, stakeMultiplier: 1 },
  aggressive: { minProbability: 0.4, maxRank: 12, stakeMultiplier: 1.25 },
};

const el = {
  raceSelect: document.querySelector("#raceSelect"),
  bankrollInput: document.querySelector("#bankrollInput"),
  stakeInput: document.querySelector("#stakeInput"),
  confidenceMetric: document.querySelector("#confidenceMetric"),
  bubbleMetric: document.querySelector("#bubbleMetric"),
  weatherMetric: document.querySelector("#weatherMetric"),
  modelMetric: document.querySelector("#modelMetric"),
  circuitImage: document.querySelector("#circuitImage"),
  circuitCredit: document.querySelector("#circuitCredit"),
  detailHeroTitle: document.querySelector("#detailHeroTitle"),
  detailHeroText: document.querySelector("#detailHeroText"),
  circuitLengthMetric: document.querySelector("#circuitLengthMetric"),
  circuitLapsMetric: document.querySelector("#circuitLapsMetric"),
  overtakingMetric: document.querySelector("#overtakingMetric"),
  forecastHeadline: document.querySelector("#forecastHeadline"),
  forecastDetail: document.querySelector("#forecastDetail"),
  chaosHeadline: document.querySelector("#chaosHeadline"),
  chaosDetail: document.querySelector("#chaosDetail"),
  strategyHeadline: document.querySelector("#strategyHeadline"),
  strategyDetail: document.querySelector("#strategyDetail"),
  bettingHeadline: document.querySelector("#bettingHeadline"),
  bettingDetail: document.querySelector("#bettingDetail"),
  historyCount: document.querySelector("#historyCount"),
  recentRaceList: document.querySelector("#recentRaceList"),
  topPerformerList: document.querySelector("#topPerformerList"),
  teamStrategyNote: document.querySelector("#teamStrategyNote"),
  teamStrategyList: document.querySelector("#teamStrategyList"),
  selectedDriverTitle: document.querySelector("#selectedDriverTitle"),
  selectedDriverSignal: document.querySelector("#selectedDriverSignal"),
  selectedDriverStats: document.querySelector("#selectedDriverStats"),
  selectedDriverResults: document.querySelector("#selectedDriverResults"),
  raceDateLabel: document.querySelector("#raceDateLabel"),
  raceTitle: document.querySelector("#raceTitle"),
  predictionTable: document.querySelector("#predictionTable"),
  autoSlipButton: document.querySelector("#autoSlipButton"),
  clearSlipButton: document.querySelector("#clearSlipButton"),
  slipList: document.querySelector("#slipList"),
  poolMetric: document.querySelector("#poolMetric"),
  exposureMetric: document.querySelector("#exposureMetric"),
  potentialReturnMetric: document.querySelector("#potentialReturnMetric"),
  potentialProfitMetric: document.querySelector("#potentialProfitMetric"),
  remainingMetric: document.querySelector("#remainingMetric"),
  settlementGrid: document.querySelector("#settlementGrid"),
  settleButton: document.querySelector("#settleButton"),
  settlementResult: document.querySelector("#settlementResult"),
};

function credits(value) {
  return `${Math.round(value).toLocaleString("fr-FR")} cr`;
}

function percent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function activeStake() {
  const baseStake = Number(el.stakeInput.value || 0);
  const multiplier = strategyRules[state.strategy].stakeMultiplier;
  return Math.max(1, Math.round(baseStake * multiplier));
}

function bankroll() {
  return Math.max(0, Number(el.bankrollInput.value || 0));
}

function driverByCode(code) {
  return state.race.drivers.find((driver) => driver.driver_code === code);
}

function selectedDriver() {
  return driverByCode(state.selectedDriverCode) || state.race.drivers[0];
}

function marketStake(driver) {
  const crowdWeight = Math.max(0.02, Math.pow(driver.top10_probability, 1.55));
  const totalWeight = state.race.drivers.reduce(
    (sum, item) => sum + Math.max(0.02, Math.pow(item.top10_probability, 1.55)),
    0,
  );
  const crowdStake = (DEFAULT_POOL * crowdWeight) / totalWeight;
  const userStake = state.slip.get(driver.driver_code)?.stake || 0;
  return crowdStake + userStake;
}

function poolTotal() {
  let userTotal = 0;
  state.slip.forEach((bet) => {
    userTotal += bet.stake;
  });
  return DEFAULT_POOL + userTotal;
}

function impliedEdge(driver, offeredOdds) {
  if (!offeredOdds || offeredOdds <= 1) return null;
  return driver.top10_probability * offeredOdds - 1;
}

function driverUrl(driverCode) {
  return `driver.html?race=${encodeURIComponent(state.race.race_id)}&driver=${encodeURIComponent(driverCode)}`;
}

function currentProfile() {
  return window.ProfileStore?.load() || {};
}

function setActiveStrategy(strategy) {
  if (!strategyRules[strategy]) return;
  state.strategy = strategy;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.strategy === strategy);
  });
}

function marketDrivers() {
  if (state.market === "value") {
    return state.race.drivers.filter((driver) => {
      const history = driver.circuit_history || {};
      return (
        driver.predicted_rank >= 7 &&
        driver.predicted_rank <= 12 &&
        (driver.top10_probability >= 0.4 ||
          Number(history.top10_rate_pct || 0) >= 45 ||
          Number(history.avg_position_gain || 0) >= 2)
      );
    });
  }
  if (state.market === "outsiders") {
    return state.race.drivers.filter(
      (driver) => driver.predicted_rank > 10 && driver.top10_probability >= 0.18,
    );
  }
  if (state.market === "history") {
    return state.race.drivers.filter((driver) => {
      const history = driver.circuit_history || {};
      return Number(history.starts || 0) > 0 && Number(history.top10_rate_pct || 0) >= 50;
    });
  }
  return state.race.drivers.filter((driver) => driver.predicted_rank <= 10);
}

function setRace(raceId) {
  state.race = state.data.races.find((race) => race.race_id === raceId) || state.data.races[0];
  state.selectedDriverCode = state.race.drivers[0]?.driver_code || null;
  state.slip.clear();
  state.offeredOdds.clear();
  el.settlementResult.textContent = "";
  window.history.replaceState(null, "", `race.html?race=${encodeURIComponent(state.race.race_id)}`);
  render();
}

function renderRaceOptions() {
  el.raceSelect.innerHTML = state.data.races
    .map(
      (race) =>
        `<option value="${race.race_id}">${race.round}. ${race.grand_prix} - ${race.race_date}</option>`,
    )
    .join("");
}

function renderSummary() {
  const nn = state.data.metadata.neural_network || {};
  const holdout = nn.best_holdout || {};
  const profile = state.race.intelligence?.circuit_profile || {};
  el.confidenceMetric.textContent = percent(state.race.average_top10_probability_pct);
  el.bubbleMetric.textContent = `${state.race.bubble_gap_pct >= 0 ? "+" : ""}${percent(
    state.race.bubble_gap_pct,
  )}`;
  el.weatherMetric.textContent = `${state.race.weather_condition || "Unknown"} / ${percent(
    state.race.rainfall_percentage,
  )}`;
  el.modelMetric.textContent = holdout.race_precision_at_10
    ? Number(holdout.race_precision_at_10).toFixed(3)
    : "--";
  el.raceDateLabel.textContent = `${state.race.race_date} - ${state.race.circuit_name}`;
  el.raceTitle.textContent = state.race.grand_prix;
  if (state.race.circuit_image_path) {
    el.circuitImage.src = state.race.circuit_image_path;
  }
  el.circuitImage.alt = `Plan du ${state.race.circuit_name}`;
  const attribution = state.race.circuit_image_attribution || {};
  el.circuitCredit.href = attribution.source_url || "#";
  el.circuitCredit.textContent = attribution.author
    ? `Map: ${attribution.author} (${attribution.license || "license"})`
    : "Circuit map credit";
  el.detailHeroTitle.textContent = state.race.grand_prix;
  el.detailHeroText.textContent = `${state.race.circuit_name} - ${state.race.weather_condition}, pluie ${percent(
    state.race.rainfall_percentage,
  )}, chaos ${profile.chaos_label || "Unknown"}.`;
  el.circuitLengthMetric.textContent = profile.circuit_length_km
    ? `${Number(profile.circuit_length_km).toFixed(3)} km`
    : "--";
  el.circuitLapsMetric.textContent = profile.total_laps || "--";
  el.overtakingMetric.textContent = profile.overtaking_difficulty || "--";
}

function renderIntelligence() {
  const intelligence = state.race.intelligence || {};
  const profile = intelligence.circuit_profile || {};
  const strategy = intelligence.strategy || {};
  const angles = intelligence.betting_angles || {};
  const safetyCarRate =
    profile.model_safety_car_rate_previous_3_pct ?? profile.safety_car_race_rate_pct ?? 0;
  const disruption =
    profile.model_disruption_score_previous_3 ?? profile.avg_disruption_score ?? 0;

  el.forecastHeadline.textContent = `${state.race.weather_condition || "Unknown"} - ${
    state.race.air_temp_mean
  }°C`;
  el.forecastDetail.textContent = `Pluie ${percent(
    state.race.rainfall_percentage,
  )}, wet race ${percent(state.race.wet_race_probability)}, source ${
    state.race.prediction_weather_source || "unknown"
  }`;

  el.chaosHeadline.textContent = `${profile.chaos_label || "Unknown"} - SC ${percent(
    safetyCarRate,
  )}`;
  el.chaosDetail.textContent = `${profile.races_analyzed || 0} courses analysees, disruption ${Number(
    disruption,
  ).toFixed(1)}, wet historique ${percent(profile.wet_race_rate_pct || 0)}`;

  el.strategyHeadline.textContent = `${strategy.expected_stops_label || "--"} / ${
    strategy.compound_bias || "--"
  }`;
  el.strategyDetail.textContent = `${strategy.base_plan || "--"} Fenetre pit: ${
    strategy.pit_window || "--"
  }. Sensibilite SC: ${strategy.safety_car_sensitivity || "--"}`;

  const safeCodes = (angles.safe || []).map((driver) => driver.driver_code).join(", ");
  const valueCodes = (angles.value_watch || []).map((driver) => driver.driver_code).join(", ");
  el.bettingHeadline.textContent = safeCodes ? `Safe: ${safeCodes}` : "Attendre la grille";
  el.bettingDetail.textContent =
    (angles.warnings || [])[0] ||
    (valueCodes ? `Value watch: ${valueCodes}` : "Pas de warning majeur sur ce dossier.");

  el.historyCount.textContent = `${profile.races_analyzed || 0} courses`;
  el.recentRaceList.innerHTML = (profile.recent_races || [])
    .map(
      (race) => `
        <div class="compact-row">
          <strong>${race.season} - ${race.winner_code || "--"}</strong>
          <span>${race.weather_condition}, SC ${race.safety_car_count}, disruption ${race.race_disruption_score}</span>
        </div>
      `,
    )
    .join("");

  el.topPerformerList.innerHTML = (intelligence.top_circuit_performers || [])
    .map((item) => {
      const history = item.history || {};
      return `
        <button class="compact-row compact-button" data-driver-detail="${item.driver_code}">
          <strong>${item.driver_code} - ${history.top10_rate_pct || 0}% top 10</strong>
          <span>${history.starts || 0} departs, avg P${history.avg_finish || "--"}, ${history.signal || ""}</span>
        </button>
      `;
    })
    .join("");

  el.teamStrategyNote.textContent = strategy.pit_window || "--";
  el.teamStrategyList.innerHTML = (strategy.team_tendencies || [])
    .map(
      (team) => `
        <div class="compact-row">
          <strong>${team.constructor_name}</strong>
          <span>${team.avg_top10_probability_pct}% top 10 moyen, ${team.avg_stints_previous_3} stints, ${team.compound_bias}</span>
        </div>
      `,
    )
    .join("");

  const driver = selectedDriver();
  if (!driver) return;
  const history = driver.circuit_history || {};
  el.selectedDriverTitle.textContent = `${driver.driver_code} - ${driver.driver_name}`;
  el.selectedDriverSignal.textContent = history.signal || "--";
  el.selectedDriverStats.innerHTML = `
    <div><span>Departs</span><strong>${history.starts || 0}</strong></div>
    <div><span>Top 10 ici</span><strong>${percent(history.top10_rate_pct || 0)}</strong></div>
    <div><span>Meilleur</span><strong>P${history.best_finish || "--"}</strong></div>
    <div><span>Gain moyen</span><strong>${Number(history.avg_position_gain || 0).toFixed(1)}</strong></div>
    <div><span>Podiums</span><strong>${history.podiums || 0}</strong></div>
    <div><span>Facteur chaos</span><strong>${percent(history.chaos_dependency_pct || 0)}</strong></div>
  `;
  const recentResults =
    (history.recent_results || [])
      .map(
        (result) => `
          <div class="compact-row">
            <strong>${result.season}: P${result.finish} depuis P${result.grid}</strong>
            <span>${result.weather_condition}, SC ${result.safety_car_count}, disruption ${result.race_disruption_score}</span>
          </div>
        `,
      )
      .join("") || `<div class="empty-state">Aucun historique direct sur ce circuit.</div>`;
  el.selectedDriverResults.innerHTML = `
    ${recentResults}
    <a class="row-link full-row-link" href="${driverUrl(driver.driver_code)}">Ouvrir la fiche pilote</a>
  `;
}

function avatarMarkup(driver) {
  if (driver.headshot_path) {
    return `<img class="avatar" src="${driver.headshot_path}" alt="" />`;
  }
  return `<div class="avatar avatar-fallback">${driver.driver_code.slice(0, 3)}</div>`;
}

function renderPredictionTable() {
  const rows = marketDrivers();
  const profile = currentProfile();
  el.predictionTable.innerHTML = rows
    .map((driver) => {
      const offered = state.offeredOdds.get(driver.driver_code) || "";
      const edge = impliedEdge(driver, Number(offered));
      const edgeClass = edge === null ? "" : edge >= 0 ? "edge-positive" : "edge-negative";
      const edgeText = edge === null ? "--" : `${edge >= 0 ? "+" : ""}${percent(edge * 100)}`;
      const tagClass = driver.confidence_label.toLowerCase();
      const selectedClass = driver.driver_code === state.selectedDriverCode ? "selected-row" : "";
      const favoriteDriver = driver.driver_code === profile.favoriteDriverCode ? "favorite-row" : "";

      return `
        <tr class="${selectedClass} ${favoriteDriver}" data-driver-detail="${driver.driver_code}">
          <td class="rank-cell">#${driver.predicted_rank}</td>
          <td>
            <div class="driver-cell">
              ${avatarMarkup(driver)}
              <div>
                <span class="driver-name">${driver.driver_name}</span>
                <span class="team-name">${driver.constructor_name} - grille P${driver.grid}</span>
              </div>
            </div>
          </td>
          <td class="prob-cell">
            <strong>${percent(driver.top10_probability_pct)}</strong>
            <span class="tag ${tagClass}">${driver.confidence_label}</span>
            ${favoriteDriver ? `<span class="tag favorite">Favori</span>` : ""}
            <div class="prob-track"><div class="prob-fill" style="width:${driver.top10_probability_pct}%"></div></div>
          </td>
          <td>${driver.fair_decimal_odds ? driver.fair_decimal_odds.toFixed(2) : "--"}</td>
          <td>
            <input
              class="odds-input"
              type="number"
              min="1.01"
              step="0.01"
              value="${offered}"
              data-odds="${driver.driver_code}"
              placeholder="ex 2.10"
            />
          </td>
          <td class="${edgeClass}">${edgeText}</td>
          <td><button class="add-button" data-add="${driver.driver_code}">Ajouter</button></td>
        </tr>
      `;
    })
    .join("");
}

function renderSlip() {
  if (state.slip.size === 0) {
    el.slipList.innerHTML = `<div class="empty-state">Aucun ticket actif</div>`;
  } else {
    el.slipList.innerHTML = Array.from(state.slip.values())
      .map((bet) => {
        const driver = driverByCode(bet.driver_code);
        const oddsLabel = bet.offeredOdds
          ? `${bet.oddsSource === "pmu" ? "cote PMU" : "cote modele"} ${Number(bet.offeredOdds).toFixed(2)}`
          : "pari-mutuel";
        return `
          <div class="slip-item">
            <div>
              <span class="slip-title">${driver.driver_code} top 10</span>
              <span class="slip-subtitle">${credits(bet.stake)} - ${oddsLabel} - retour ${credits(
                bet.offeredOdds ? bet.stake * bet.offeredOdds : 0,
              )}</span>
            </div>
            <input type="number" min="1" step="1" value="${bet.stake}" data-stake="${bet.driver_code}" />
            <button class="remove-button" data-remove="${bet.driver_code}">x</button>
          </div>
        `;
      })
      .join("");
  }

  let exposure = 0;
  let potentialReturn = 0;
  state.slip.forEach((bet) => {
    exposure += bet.stake;
    potentialReturn += bet.offeredOdds ? bet.stake * bet.offeredOdds : 0;
  });

  el.poolMetric.textContent = credits(poolTotal());
  el.exposureMetric.textContent = credits(exposure);
  el.potentialReturnMetric.textContent = potentialReturn ? credits(potentialReturn) : "--";
  el.potentialProfitMetric.textContent = potentialReturn ? credits(potentialReturn - exposure) : "--";
  el.remainingMetric.textContent = credits(bankroll() - exposure);
}

function renderSettlement() {
  const firstTen = new Set(state.race.drivers.slice(0, 10).map((driver) => driver.driver_code));
  el.settlementGrid.innerHTML = state.race.drivers
    .map(
      (driver) => `
        <label class="result-toggle">
          <input type="checkbox" data-result="${driver.driver_code}" ${firstTen.has(driver.driver_code) ? "checked" : ""} />
          ${driver.driver_code}
        </label>
      `,
    )
    .join("");
}

function render() {
  renderSummary();
  renderIntelligence();
  renderPredictionTable();
  renderSlip();
  renderSettlement();
}

function addBet(driverCode) {
  const driver = driverByCode(driverCode);
  if (!driver) return;
  const enteredOdds = Number(state.offeredOdds.get(driverCode) || 0);
  const offeredOdds = enteredOdds > 1 ? enteredOdds : driver.fair_decimal_odds;
  state.slip.set(driverCode, {
    driver_code: driverCode,
    stake: activeStake(),
    offeredOdds: offeredOdds > 1 ? offeredOdds : null,
    oddsSource: enteredOdds > 1 ? "pmu" : "model",
  });
  renderSlip();
}

function autoSlip() {
  state.slip.clear();
  const rules = strategyRules[state.strategy];
  state.race.drivers
    .filter(
      (driver) =>
        driver.top10_probability >= rules.minProbability && driver.predicted_rank <= rules.maxRank,
    )
    .slice(0, 10)
    .forEach((driver) => addBet(driver.driver_code));
  renderSlip();
}

function settlePool() {
  const checked = Array.from(document.querySelectorAll("[data-result]:checked")).map(
    (input) => input.dataset.result,
  );
  const winners = new Set(checked);
  const winningMarketStake = state.race.drivers.reduce((sum, driver) => {
    return winners.has(driver.driver_code) ? sum + marketStake(driver) : sum;
  }, 0);

  let userPayout = 0;
  let userStake = 0;
  state.slip.forEach((bet) => {
    userStake += bet.stake;
    if (winners.has(bet.driver_code) && winningMarketStake > 0) {
      userPayout += (bet.stake / winningMarketStake) * poolTotal();
    }
  });

  const profit = userPayout - userStake;
  const resultClass = profit >= 0 ? "edge-positive" : "edge-negative";
  el.settlementResult.innerHTML = `
    Retour: <strong>${credits(userPayout)}</strong> -
    P/L: <strong class="${resultClass}">${profit >= 0 ? "+" : ""}${credits(profit)}</strong>
  `;
}

function bindEvents() {
  el.raceSelect.addEventListener("change", (event) => setRace(event.target.value));
  el.bankrollInput.addEventListener("input", renderSlip);
  el.stakeInput.addEventListener("input", renderSlip);
  el.autoSlipButton.addEventListener("click", autoSlip);
  el.clearSlipButton.addEventListener("click", () => {
    state.slip.clear();
    renderSlip();
  });
  el.settleButton.addEventListener("click", settlePool);

  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => {
      setActiveStrategy(button.dataset.strategy);
      renderSlip();
    });
  });

  document.querySelectorAll(".market-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".market-tab").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.market = button.dataset.market;
      renderPredictionTable();
    });
  });

  document.querySelectorAll(".policy-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".policy-button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.oddsPolicy = button.dataset.oddsPolicy;
    });
  });

  document.body.addEventListener("input", (event) => {
    const stakeCode = event.target.dataset?.stake;
    if (stakeCode && state.slip.has(stakeCode)) {
      const bet = state.slip.get(stakeCode);
      bet.stake = Math.max(1, Number(event.target.value || 0));
      state.slip.set(stakeCode, bet);
      renderSlip();
    }
  });

  document.body.addEventListener("change", (event) => {
    const oddsCode = event.target.dataset?.odds;
    if (oddsCode) {
      state.offeredOdds.set(oddsCode, event.target.value);
      if (state.slip.has(oddsCode)) {
        const bet = state.slip.get(oddsCode);
        const offeredOdds = Number(event.target.value || 0);
        bet.offeredOdds = offeredOdds > 1 ? offeredOdds : null;
        bet.oddsSource = offeredOdds > 1 ? "pmu" : bet.oddsSource;
        state.slip.set(oddsCode, bet);
        renderSlip();
      }
      renderPredictionTable();
    }
  });

  document.body.addEventListener("click", (event) => {
    const addCode = event.target.dataset?.add;
    const removeCode = event.target.dataset?.remove;
    if (addCode) {
      addBet(addCode);
      return;
    }
    if (removeCode) {
      state.slip.delete(removeCode);
      renderSlip();
      return;
    }
    const detailTarget = event.target.closest("[data-driver-detail]");
    if (detailTarget) {
      state.selectedDriverCode = detailTarget.dataset.driverDetail;
      renderIntelligence();
      renderPredictionTable();
    }
  });
}

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Unable to load ${DATA_URL}`);
  }
  state.data = await response.json();
  const params = new URLSearchParams(window.location.search);
  const requestedRace = params.get("race");
  const profile = currentProfile();
  state.race =
    state.data.races.find((race) => race.race_id === requestedRace) ||
    state.data.races.find((race) => race.race_id === profile.favoriteRaceId) ||
    state.data.races[0];
  state.selectedDriverCode =
    state.race.drivers.find((driver) => driver.driver_code === profile.favoriteDriverCode)?.driver_code ||
    state.race.drivers[0]?.driver_code ||
    null;
  if (profile.bankroll) {
    el.bankrollInput.value = profile.bankroll;
  }
  renderRaceOptions();
  el.raceSelect.value = state.race.race_id;
  bindEvents();
  setActiveStrategy(profile.riskProfile || state.strategy);
  render();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><h1>Erreur de chargement</h1><p>${error.message}</p></main>`;
});
