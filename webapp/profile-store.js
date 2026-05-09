(function () {
  const STORAGE_KEY = "f1_neural_profile_v1";
  const DEFAULT_PROFILE = {
    displayName: "",
    favoriteDriverCode: "",
    favoriteRaceId: "",
    riskProfile: "balanced",
    bankroll: 1000,
    createdAt: "",
    updatedAt: "",
  };

  function load() {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return { ...DEFAULT_PROFILE };
      return { ...DEFAULT_PROFILE, ...JSON.parse(raw) };
    } catch {
      return { ...DEFAULT_PROFILE };
    }
  }

  function save(profile) {
    const existing = load();
    const now = new Date().toISOString();
    const next = {
      ...DEFAULT_PROFILE,
      ...existing,
      ...profile,
      createdAt: existing.createdAt || now,
      updatedAt: now,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    window.dispatchEvent(new CustomEvent("f1-profile-updated", { detail: next }));
    return next;
  }

  function clear() {
    window.localStorage.removeItem(STORAGE_KEY);
    window.dispatchEvent(new CustomEvent("f1-profile-updated", { detail: { ...DEFAULT_PROFILE } }));
  }

  function hasProfile() {
    return Boolean(window.localStorage.getItem(STORAGE_KEY));
  }

  window.ProfileStore = {
    load,
    save,
    clear,
    hasProfile,
  };
})();
