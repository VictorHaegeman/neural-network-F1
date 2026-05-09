const DATA_URL = "/webapp/data/betting_guide_data.json";

const state = {
  data: null,
  race: null,
  driver: null,
};

const el = {
  raceSelect: document.querySelector("#raceSelect"),
  driverSelect: document.querySelector("#driverSelect"),
  raceBackLink: document.querySelector("#raceBackLink"),
  driverHeroAvatar: document.querySelector("#driverHeroAvatar"),
  driverContextLabel: document.querySelector("#driverContextLabel"),
  driverDetailName: document.querySelector("#driverDetailName"),
  driverDetailTeam: document.querySelector("#driverDetailTeam"),
  driverDetailSignal: document.querySelector("#driverDetailSignal"),
  driverDecisionText: document.querySelector("#driverDecisionText"),
  driverDetailStats: document.querySelector("#driverDetailStats"),
  driverHistoryCount: document.querySelector("#driverHistoryCount"),
  driverDetailResults: document.querySelector("#driverDetailResults"),
  raceContextLabel: document.querySelector("#raceContextLabel"),
  raceContextList: document.querySelector("#raceContextList"),
  marketComparisonList: document.querySelector("#marketComparisonList"),
};

function percent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function raceUrl(raceId) {
  return `/webapp/race.html?race=${encodeURIComponent(raceId)}`;
}

function driverUrl(raceId, driverCode) {
  return `/webapp/driver.html?race=${encodeURIComponent(raceId)}&driver=${encodeURIComponent(driverCode)}`;
}

function avatarMarkup(driver) {
  if (driver.headshot_path) {
    return `<img class="avatar" src="${driver.headshot_path}" alt="" />`;
  }
  return `<div class="avatar avatar-fallback">${driver.driver_code.slice(0, 3)}</div>`;
}

function setRace(raceId, preferredDriverCode = null) {
  state.race = state.data.races.find((race) => race.race_id === raceId) || state.data.races[0];
  const driverCode = preferredDriverCode || state.driver?.driver_code;
  state.driver = state.race.drivers.find((driver) => driver.driver_code === driverCode) || state.race.drivers[0];
  renderDriverOptions();
  render();
  window.history.replaceState(null, "", driverUrl(state.race.race_id, state.driver.driver_code));
}

function setDriver(driverCode) {
  state.driver = state.race.drivers.find((driver) => driver.driver_code === driverCode) || state.race.drivers[0];
  render();
  window.history.replaceState(null, "", driverUrl(state.race.race_id, state.driver.driver_code));
}

function renderRaceOptions() {
  el.raceSelect.innerHTML = state.data.races
    .map(
      (race) =>
        `<option value="${race.race_id}">${race.round}. ${race.grand_prix} - ${race.race_date}</option>`,
    )
    .join("");
}

function renderDriverOptions() {
  el.driverSelect.innerHTML = state.race.drivers
    .map(
      (driver) =>
        `<option value="${driver.driver_code}">#${driver.predicted_rank} ${driver.driver_code} - ${driver.driver_name}</option>`,
    )
    .join("");
}

function renderHero(history) {
  el.driverHeroAvatar.innerHTML = avatarMarkup(state.driver);
  el.driverContextLabel.textContent = `${state.race.grand_prix} - ${state.race.race_date}`;
  el.driverDetailName.textContent = `${state.driver.driver_name} (${state.driver.driver_code})`;
  el.driverDetailTeam.textContent = `${state.driver.constructor_name} - grille P${state.driver.grid} - rang modele #${state.driver.predicted_rank}`;
  el.driverDetailSignal.textContent = history.signal || "Pas de signal historique.";
  el.driverDecisionText.textContent = `${state.driver.recommendation}. Le modele donne ${percent(
    state.driver.top10_probability_pct,
  )} de probabilite top 10, soit une cote juste de ${
    state.driver.fair_decimal_odds ? state.driver.fair_decimal_odds.toFixed(2) : "--"
  }.`;
}

function renderStats(history) {
  el.driverDetailStats.innerHTML = `
    <div><span>Proba top 10</span><strong>${percent(state.driver.top10_probability_pct)}</strong></div>
    <div><span>Cote juste</span><strong>${state.driver.fair_decimal_odds ? state.driver.fair_decimal_odds.toFixed(2) : "--"}</strong></div>
    <div><span>Reco</span><strong>${state.driver.recommendation}</strong></div>
    <div><span>Top 10 circuit</span><strong>${percent(history.top10_rate_pct || 0)}</strong></div>
    <div><span>Departs ici</span><strong>${history.starts || 0}</strong></div>
    <div><span>Finish moyen</span><strong>P${history.avg_finish || "--"}</strong></div>
    <div><span>Meilleur ici</span><strong>P${history.best_finish || "--"}</strong></div>
    <div><span>Gain grille</span><strong>${Number(history.avg_position_gain || 0).toFixed(1)}</strong></div>
    <div><span>DNF ici</span><strong>${percent(history.dnf_rate_pct || 0)}</strong></div>
    <div><span>Chaos dependance</span><strong>${percent(history.chaos_dependency_pct || 0)}</strong></div>
    <div><span>Forme top 10</span><strong>${percent((state.driver.top10_rate_previous_5 || 0) * 100)}</strong></div>
    <div><span>Avg last 5</span><strong>P${state.driver.avg_finish_position_previous_5 || "--"}</strong></div>
  `;
}

function renderHistory(history) {
  el.driverHistoryCount.textContent = `${history.starts || 0} departs`;
  el.driverDetailResults.innerHTML = (history.recent_results || [])
    .map(
      (result) => `
        <div class="compact-row">
          <strong>${result.season}: P${result.finish} depuis P${result.grid}</strong>
          <span>${result.points} pts, ${result.weather_condition}, SC ${result.safety_car_count}, disruption ${result.race_disruption_score}</span>
        </div>
      `,
    )
    .join("") || `<div class="empty-state">Aucun historique direct sur ce circuit.</div>`;
}

function renderRaceContext() {
  const profile = state.race.intelligence?.circuit_profile || {};
  const strategy = state.race.intelligence?.strategy || {};
  el.raceContextLabel.textContent = profile.chaos_label || "--";
  el.raceContextList.innerHTML = `
    <div class="compact-row"><strong>Meteo</strong><span>${state.race.weather_condition}, pluie ${percent(
      state.race.rainfall_percentage,
    )}, wet race ${percent(state.race.wet_race_probability)}</span></div>
    <div class="compact-row"><strong>Safety car</strong><span>${percent(
      profile.model_safety_car_rate_previous_3_pct ?? profile.safety_car_race_rate_pct ?? 0,
    )} sur le signal recent, disruption ${profile.model_disruption_score_previous_3 ?? profile.avg_disruption_score ?? "--"}</span></div>
    <div class="compact-row"><strong>Strategie</strong><span>${strategy.expected_stops_label || "--"}, ${
      strategy.compound_bias || "--"
    }, fenetre ${strategy.pit_window || "--"}</span></div>
  `;
}

function renderMarketComparison() {
  const neighbors = state.race.drivers
    .filter((driver) => Math.abs(driver.predicted_rank - state.driver.predicted_rank) <= 2)
    .map(
      (driver) => `
        <div class="compact-row ${driver.driver_code === state.driver.driver_code ? "highlight-row" : ""}">
          <strong>#${driver.predicted_rank} ${driver.driver_code} - ${percent(driver.top10_probability_pct)}</strong>
          <span>Cote juste ${driver.fair_decimal_odds ? driver.fair_decimal_odds.toFixed(2) : "--"} - ${driver.confidence_label}</span>
        </div>
      `,
    )
    .join("");
  el.marketComparisonList.innerHTML = neighbors;
}

function render() {
  const history = state.driver.circuit_history || {};
  el.raceSelect.value = state.race.race_id;
  el.driverSelect.value = state.driver.driver_code;
  el.raceBackLink.href = raceUrl(state.race.race_id);
  renderHero(history);
  renderStats(history);
  renderHistory(history);
  renderRaceContext();
  renderMarketComparison();
}

function bindEvents() {
  el.raceSelect.addEventListener("change", (event) => setRace(event.target.value));
  el.driverSelect.addEventListener("change", (event) => setDriver(event.target.value));
}

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Unable to load ${DATA_URL}`);
  }
  state.data = await response.json();
  const params = new URLSearchParams(window.location.search);
  const requestedRace = params.get("race");
  const requestedDriver = params.get("driver");
  state.race = state.data.races.find((race) => race.race_id === requestedRace) || state.data.races[0];
  state.driver =
    state.race.drivers.find((driver) => driver.driver_code === requestedDriver) || state.race.drivers[0];
  renderRaceOptions();
  renderDriverOptions();
  bindEvents();
  render();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><h1>Erreur de chargement</h1><p>${error.message}</p></main>`;
});
