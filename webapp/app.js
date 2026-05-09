const DATA_URL = "/webapp/data/betting_guide_data.json";

const state = {
  data: null,
  race: null,
};

const el = {
  raceSelect: document.querySelector("#raceSelect"),
  raceCards: document.querySelector("#raceCards"),
  raceDetailLink: document.querySelector("#raceDetailLink"),
  panelDetailLink: document.querySelector("#panelDetailLink"),
  confidenceMetric: document.querySelector("#confidenceMetric"),
  bubbleMetric: document.querySelector("#bubbleMetric"),
  weatherMetric: document.querySelector("#weatherMetric"),
  modelMetric: document.querySelector("#modelMetric"),
  raceDateLabel: document.querySelector("#raceDateLabel"),
  raceTitle: document.querySelector("#raceTitle"),
  predictionTable: document.querySelector("#predictionTable"),
};

function percent(value) {
  return `${Number(value).toFixed(1)}%`;
}

function detailUrl(raceId) {
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

function setRace(raceId) {
  state.race = state.data.races.find((race) => race.race_id === raceId) || state.data.races[0];
  el.raceSelect.value = state.race.race_id;
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
  el.raceDetailLink.href = detailUrl(state.race.race_id);
  el.panelDetailLink.href = detailUrl(state.race.race_id);
}

function renderRaceCards() {
  el.raceCards.innerHTML = state.data.races
    .map((race) => {
      const active = race.race_id === state.race.race_id ? "active-card" : "";
      const topDrivers = race.drivers
        .slice(0, 3)
        .map((driver) => `${driver.driver_code} ${percent(driver.top10_probability_pct)}`)
        .join(" / ");
      const profile = race.intelligence?.circuit_profile || {};
      return `
        <button class="scoreboard-race-card ${active}" data-race-card="${race.race_id}">
          <span class="summary-label">${race.race_date}</span>
          <strong>${race.grand_prix}</strong>
          <span>${topDrivers}</span>
          <span>${race.weather_condition} - chaos ${profile.chaos_label || "Unknown"}</span>
        </button>
      `;
    })
    .join("");
}

function renderPredictionTable() {
  el.predictionTable.innerHTML = state.race.drivers
    .map((driver) => {
      const history = driver.circuit_history || {};
      return `
        <tr>
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
            <span class="tag ${driver.confidence_label.toLowerCase()}">${driver.confidence_label}</span>
            <div class="prob-track"><div class="prob-fill" style="width:${driver.top10_probability_pct}%"></div></div>
          </td>
          <td>${driver.fair_decimal_odds ? driver.fair_decimal_odds.toFixed(2) : "--"}</td>
          <td>
            <strong>${history.starts || 0} departs</strong>
            <span class="table-muted">${history.top10_rate_pct || 0}% top 10 ici</span>
          </td>
          <td><a class="row-link" href="${driverUrl(state.race.race_id, driver.driver_code)}">Voir pilote</a></td>
        </tr>
      `;
    })
    .join("");
}

function render() {
  renderSummary();
  renderRaceCards();
  renderPredictionTable();
}

function bindEvents() {
  el.raceSelect.addEventListener("change", (event) => setRace(event.target.value));
  document.body.addEventListener("click", (event) => {
    const card = event.target.closest("[data-race-card]");
    if (card) {
      setRace(card.dataset.raceCard);
    }
  });
}

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Unable to load ${DATA_URL}`);
  }
  state.data = await response.json();
  state.race = state.data.races[0];
  renderRaceOptions();
  bindEvents();
  render();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><h1>Erreur de chargement</h1><p>${error.message}</p></main>`;
});
