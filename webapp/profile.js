const DATA_URL = "data/betting_guide_data.json";

const state = {
  data: null,
  profile: null,
};

const el = {
  form: document.querySelector("#profileForm"),
  displayNameInput: document.querySelector("#displayNameInput"),
  favoriteDriverSelect: document.querySelector("#favoriteDriverSelect"),
  favoriteRaceSelect: document.querySelector("#favoriteRaceSelect"),
  riskProfileSelect: document.querySelector("#riskProfileSelect"),
  bankrollInput: document.querySelector("#bankrollInput"),
  resetProfileButton: document.querySelector("#resetProfileButton"),
  saveStatus: document.querySelector("#saveStatus"),
  profileAvatar: document.querySelector("#profileAvatar"),
  profileName: document.querySelector("#profileName"),
  profileSummaryList: document.querySelector("#profileSummaryList"),
  raceLink: document.querySelector("#raceLink"),
  driverLink: document.querySelector("#driverLink"),
};

function raceUrl(raceId) {
  return `race.html?race=${encodeURIComponent(raceId)}`;
}

function driverUrl(raceId, driverCode) {
  return `driver.html?race=${encodeURIComponent(raceId)}&driver=${encodeURIComponent(driverCode)}`;
}

function allDrivers() {
  const drivers = new Map();
  state.data.races.forEach((race) => {
    race.drivers.forEach((driver) => {
      if (!drivers.has(driver.driver_code)) {
        drivers.set(driver.driver_code, driver);
      }
    });
  });
  return Array.from(drivers.values()).sort((a, b) => a.driver_name.localeCompare(b.driver_name));
}

function driverByCode(code) {
  return allDrivers().find((driver) => driver.driver_code === code) || allDrivers()[0];
}

function raceById(raceId) {
  return state.data.races.find((race) => race.race_id === raceId) || state.data.races[0];
}

function avatarMarkup(driver) {
  if (driver?.headshot_path) {
    return `<img class="avatar" src="${driver.headshot_path}" alt="" />`;
  }
  return `<div class="avatar avatar-fallback">${(driver?.driver_code || "F1").slice(0, 3)}</div>`;
}

function renderOptions() {
  el.favoriteRaceSelect.innerHTML = state.data.races
    .map((race) => `<option value="${race.race_id}">${race.round}. ${race.grand_prix}</option>`)
    .join("");

  el.favoriteDriverSelect.innerHTML = allDrivers()
    .map(
      (driver) =>
        `<option value="${driver.driver_code}">${driver.driver_code} - ${driver.driver_name}</option>`,
    )
    .join("");
}

function renderForm() {
  const race = raceById(state.profile.favoriteRaceId);
  const driver = driverByCode(state.profile.favoriteDriverCode);
  el.displayNameInput.value = state.profile.displayName || "";
  el.favoriteRaceSelect.value = race.race_id;
  el.favoriteDriverSelect.value = driver.driver_code;
  el.riskProfileSelect.value = state.profile.riskProfile || "balanced";
  el.bankrollInput.value = state.profile.bankroll || 1000;
}

function renderSummary() {
  const race = raceById(state.profile.favoriteRaceId);
  const driver = driverByCode(state.profile.favoriteDriverCode);
  const riskLabels = {
    safe: "Safe",
    balanced: "Equilibre",
    aggressive: "Agressif",
  };

  el.profileAvatar.innerHTML = avatarMarkup(driver);
  el.profileName.textContent = state.profile.displayName || "Invite";
  el.profileSummaryList.innerHTML = `
    <div class="compact-row"><strong>${driver.driver_code} - ${driver.driver_name}</strong><span>Pilote favori</span></div>
    <div class="compact-row"><strong>${race.grand_prix}</strong><span>GP favori</span></div>
    <div class="compact-row"><strong>${riskLabels[state.profile.riskProfile] || "Equilibre"}</strong><span>Style de ticket</span></div>
    <div class="compact-row"><strong>${Number(state.profile.bankroll || 1000).toLocaleString("fr-FR")} cr</strong><span>Bankroll fictive</span></div>
  `;
  el.raceLink.href = raceUrl(race.race_id);
  el.driverLink.href = driverUrl(race.race_id, driver.driver_code);
}

function render() {
  renderForm();
  renderSummary();
}

function saveFromForm() {
  state.profile = window.ProfileStore.save({
    displayName: el.displayNameInput.value.trim(),
    favoriteDriverCode: el.favoriteDriverSelect.value,
    favoriteRaceId: el.favoriteRaceSelect.value,
    riskProfile: el.riskProfileSelect.value,
    bankroll: Math.max(10, Number(el.bankrollInput.value || 1000)),
  });
  el.saveStatus.textContent = "Profil sauvegarde";
  renderSummary();
}

function bindEvents() {
  el.form.addEventListener("submit", (event) => {
    event.preventDefault();
    saveFromForm();
  });

  el.resetProfileButton.addEventListener("click", () => {
    window.ProfileStore.clear();
    state.profile = window.ProfileStore.load();
    el.saveStatus.textContent = "Profil efface";
    render();
  });
}

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Unable to load ${DATA_URL}`);
  }
  state.data = await response.json();
  state.profile = window.ProfileStore.load();
  if (!state.profile.favoriteRaceId) {
    state.profile.favoriteRaceId = state.data.races[0]?.race_id || "";
  }
  if (!state.profile.favoriteDriverCode) {
    state.profile.favoriteDriverCode = state.data.races[0]?.drivers[0]?.driver_code || "";
  }
  renderOptions();
  bindEvents();
  render();
}

init().catch((error) => {
  document.body.innerHTML = `<main class="app-shell"><h1>Erreur de chargement</h1><p>${error.message}</p></main>`;
});
