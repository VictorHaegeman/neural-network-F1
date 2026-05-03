from __future__ import annotations

import argparse
import re
import time
from collections import defaultdict, deque
from datetime import date
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd


RAW_PATH = Path("data/raw")
RACE_RESULTS_PATH = RAW_PATH / "race_results.csv"
CACHE_PATH = Path(".fastf1_cache")
FASTF1_RACE_CONTROL_PATH = RAW_PATH / "fastf1_race_control.csv"
RACE_CONTROL_HISTORY_PATH = RAW_PATH / "race_control_history.csv"

SUPPORTED_START_YEAR = 2018
ROLLING_WINDOW = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch FastF1 race-control messages and pre-race history features.")
    parser.add_argument("--start-year", type=int, default=SUPPORTED_START_YEAR)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_PATH)
    parser.add_argument("--race-results", type=Path, default=RACE_RESULTS_PATH)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).upper()


def safe_mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows if float(row.get(key, 0.0)) >= 0]
    return float(sum(values) / len(values)) if values else 0.0


def safe_rate(rows: list[dict[str, Any]], key: str) -> float:
    values = [1 if float(row.get(key, 0.0)) > 0 else 0 for row in rows]
    return float(sum(values) / len(values)) if values else 0.0


def get_races(race_results: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    races = (
        race_results[["race_id", "season", "round", "grand_prix", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .copy()
    )
    races = races[races["season"].between(max(start_year, SUPPORTED_START_YEAR), end_year)]
    return races


def contains_any(text: pd.Series, patterns: list[str]) -> pd.Series:
    combined = "|".join(patterns)
    return text.str.contains(combined, regex=True, na=False)


def summarize_messages(race_id: str, messages: pd.DataFrame) -> dict[str, Any]:
    if messages is None or messages.empty:
        return {
            "race_id": race_id,
            "fastf1_race_control_available": 0,
        }

    data = messages.copy()
    text = data.get("Message", pd.Series(dtype=object)).map(safe_text)
    category = data.get("Category", pd.Series(dtype=object)).map(safe_text)
    flag = data.get("Flag", pd.Series(dtype=object)).map(safe_text)
    status = data.get("Status", pd.Series(dtype=object)).map(safe_text)

    safety_car = contains_any(text, [r"\bSAFETY CAR\b"]) & ~contains_any(text, [r"VIRTUAL SAFETY CAR"])
    safety_car_deployed = safety_car & contains_any(text, [r"DEPLOYED", r"SC DEPLOYED"])
    virtual_safety_car = contains_any(text, [r"VIRTUAL SAFETY CAR", r"\bVSC\b"])
    virtual_safety_car_deployed = virtual_safety_car & contains_any(text, [r"DEPLOYED"])

    red_flag = (flag == "RED") | contains_any(text, [r"RED FLAG"])
    yellow_flag = ((flag == "YELLOW") | contains_any(text, [r"\bYELLOW\b"])) & ~contains_any(text, [r"CLEAR"])
    double_yellow = (flag == "DOUBLE YELLOW") | contains_any(text, [r"DOUBLE YELLOW"])
    black_flag = flag.str.contains("BLACK", na=False) | contains_any(text, [r"BLACK FLAG"])

    track_limits = contains_any(text, [r"TRACK LIMIT", r"TIME .* DELETED"])
    investigation = contains_any(text, [r"INVESTIGAT", r"NOTED", r"REVIEWED"])
    penalty = contains_any(text, [r"PENALTY", r"DRIVE THROUGH", r"STOP AND GO", r"TIME PENALTY"])
    incident = (category == "CAREVENT") | contains_any(
        text,
        [r"INCIDENT", r"COLLISION", r"SPUN", r"STOPPED", r"CRASH", r"CONTACT", r"OFF TRACK"],
    )
    drs_disabled = (category == "DRS") & ((status == "DISABLED") | contains_any(text, [r"DRS DISABLED"]))
    deleted_lap = contains_any(text, [r"TIME .* DELETED"])
    slippery_track = contains_any(text, [r"SLIPPERY"])
    clear = (flag == "CLEAR") | contains_any(text, [r"\bCLEAR\b"])

    disruption_score = (
        int(safety_car_deployed.sum()) * 4
        + int(virtual_safety_car_deployed.sum()) * 3
        + int(red_flag.sum()) * 5
        + int(yellow_flag.sum()) * 1
        + int(double_yellow.sum()) * 2
        + int(incident.sum()) * 1
    )

    return {
        "race_id": race_id,
        "fastf1_race_control_available": 1,
        "fastf1_race_control_messages_count": int(len(data)),
        "fastf1_safety_car_count": int(safety_car_deployed.sum()),
        "fastf1_virtual_safety_car_count": int(virtual_safety_car_deployed.sum()),
        "fastf1_red_flag_count": int(red_flag.sum()),
        "fastf1_yellow_flag_count": int(yellow_flag.sum()),
        "fastf1_double_yellow_count": int(double_yellow.sum()),
        "fastf1_black_flag_count": int(black_flag.sum()),
        "fastf1_track_limits_count": int(track_limits.sum()),
        "fastf1_investigation_count": int(investigation.sum()),
        "fastf1_penalty_count": int(penalty.sum()),
        "fastf1_incident_count": int(incident.sum()),
        "fastf1_drs_disabled_count": int(drs_disabled.sum()),
        "fastf1_deleted_lap_count": int(deleted_lap.sum()),
        "fastf1_slippery_track_count": int(slippery_track.sum()),
        "fastf1_clear_count": int(clear.sum()),
        "fastf1_race_disruption_score": int(disruption_score),
    }


def fetch_race_control(season: int, round_number: int, race_id: str) -> dict[str, Any]:
    session = fastf1.get_session(season, round_number, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=True)
    return summarize_messages(race_id, session.race_control_messages)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def combine_existing(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        combined = new.copy()
    elif new.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, new], ignore_index=True, sort=False)

    if combined.empty:
        return combined

    return (
        combined.drop_duplicates(keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )


def build_history_features(race_results: pd.DataFrame, race_control: pd.DataFrame) -> pd.DataFrame:
    race_lookup = (
        race_results[["race_id", "season", "round", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
    )
    control_lookup = {
        str(row["race_id"]): row.to_dict()
        for _, row in race_control.iterrows()
        if int(float(row.get("fastf1_race_control_available", 0))) == 1
    }

    circuit_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    season_history: dict[int, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    rows: list[dict[str, Any]] = []

    for _, race in race_lookup.iterrows():
        race_id = str(race["race_id"])
        season = int(race["season"])
        circuit_id = str(race["circuit_id"])
        previous_circuit = list(circuit_history[circuit_id])
        previous_season = list(season_history[season])

        row = {
            "race_id": race_id,
            "race_control_history_available": int(bool(previous_circuit or previous_season)),
            "circuit_race_control_races_previous_3": len(previous_circuit),
            "circuit_safety_car_rate_previous_3": safe_rate(previous_circuit, "fastf1_safety_car_count"),
            "circuit_vsc_rate_previous_3": safe_rate(previous_circuit, "fastf1_virtual_safety_car_count"),
            "circuit_red_flag_rate_previous_3": safe_rate(previous_circuit, "fastf1_red_flag_count"),
            "circuit_yellow_flags_avg_previous_3": safe_mean(previous_circuit, "fastf1_yellow_flag_count"),
            "circuit_incidents_avg_previous_3": safe_mean(previous_circuit, "fastf1_incident_count"),
            "circuit_penalties_avg_previous_3": safe_mean(previous_circuit, "fastf1_penalty_count"),
            "circuit_track_limits_avg_previous_3": safe_mean(previous_circuit, "fastf1_track_limits_count"),
            "circuit_race_disruption_score_avg_previous_3": safe_mean(
                previous_circuit,
                "fastf1_race_disruption_score",
            ),
            "season_race_control_races_previous_3": len(previous_season),
            "season_safety_car_rate_previous_3": safe_rate(previous_season, "fastf1_safety_car_count"),
            "season_vsc_rate_previous_3": safe_rate(previous_season, "fastf1_virtual_safety_car_count"),
            "season_red_flag_rate_previous_3": safe_rate(previous_season, "fastf1_red_flag_count"),
            "season_race_disruption_score_avg_previous_3": safe_mean(
                previous_season,
                "fastf1_race_disruption_score",
            ),
        }
        rows.append(row)

        current = control_lookup.get(race_id)
        if current:
            circuit_history[circuit_id].append(current)
            season_history[season].append(current)

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if not args.race_results.exists():
        raise FileNotFoundError(f"Missing race results: {args.race_results}")

    race_results = pd.read_csv(args.race_results)
    races = get_races(race_results, args.start_year, args.end_year)

    existing = load_existing(FASTF1_RACE_CONTROL_PATH) if args.incremental else pd.DataFrame()
    existing_races = set(existing.get("race_id", pd.Series(dtype=str)).astype(str))
    if args.incremental and not args.force:
        before_count = len(races)
        races = races[~races["race_id"].astype(str).isin(existing_races)].copy()
        print(f"Incremental mode skipped {before_count - len(races)} already fetched races")

    if args.max_races is not None:
        races = races.head(args.max_races)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))
    fastf1.set_log_level("WARNING")

    rows: list[dict[str, Any]] = []
    for index, race in enumerate(races.itertuples(index=False), start=1):
        print(f"FastF1 race-control {index}/{len(races)}: {race.season} round {race.round} - {race.grand_prix}")
        try:
            rows.append(fetch_race_control(int(race.season), int(race.round), str(race.race_id)))
        except Exception as exc:
            print(f"Skipped {race.season} round {race.round}: {exc}")
            rows.append(
                {
                    "race_id": str(race.race_id),
                    "fastf1_race_control_available": 0,
                }
            )
        time.sleep(args.sleep)

    race_control = combine_existing(existing, pd.DataFrame(rows), ["race_id"])
    history = build_history_features(race_results, race_control)

    FASTF1_RACE_CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    race_control.to_csv(FASTF1_RACE_CONTROL_PATH, index=False)
    history.to_csv(RACE_CONTROL_HISTORY_PATH, index=False)

    available = int(race_control.get("fastf1_race_control_available", pd.Series(dtype=float)).fillna(0).sum())
    print(f"Wrote {FASTF1_RACE_CONTROL_PATH}: {race_control.shape[0]} rows, {race_control.shape[1]} columns")
    print(f"Wrote {RACE_CONTROL_HISTORY_PATH}: {history.shape[0]} rows, {history.shape[1]} columns")
    print(f"FastF1 race-control available: {available}/{len(race_control)} races")


if __name__ == "__main__":
    main()
