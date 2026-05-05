from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_PATH = Path("data/raw")
RAW_PATH.mkdir(parents=True, exist_ok=True)

DEFAULT_START_YEAR = 2011
DEFAULT_END_YEAR = date.today().year

CIRCUIT_LENGTH_KM = {
    "albert_park": 5.278,
    "americas": 5.513,
    "bahrain": 5.412,
    "baku": 6.003,
    "catalunya": 4.657,
    "hockenheimring": 4.574,
    "hungaroring": 4.381,
    "imola": 4.909,
    "interlagos": 4.309,
    "istanbul": 5.338,
    "jeddah": 6.174,
    "losail": 5.419,
    "marina_bay": 4.94,
    "miami": 5.412,
    "monaco": 3.337,
    "monza": 5.793,
    "mugello": 5.245,
    "nurburgring": 5.148,
    "portimao": 4.653,
    "red_bull_ring": 4.318,
    "ricard": 5.842,
    "rodriguez": 4.304,
    "shanghai": 5.451,
    "silverstone": 5.891,
    "sochi": 5.848,
    "spa": 7.004,
    "suzuka": 5.807,
    "vegas": 6.201,
    "villeneuve": 4.361,
    "yas_marina": 5.281,
    "zandvoort": 4.259,
}

STREET_CIRCUITS = {
    "albert_park",
    "baku",
    "jeddah",
    "marina_bay",
    "miami",
    "monaco",
    "vegas",
    "villeneuve",
}

HARD_TO_OVERTAKE = {"monaco", "hungaroring", "zandvoort", "marina_bay", "imola"}
EASY_TO_OVERTAKE = {"spa", "monza", "bahrain", "shanghai", "baku", "jeddah", "americas"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch F1 raw data from the Jolpica API.")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between API calls.")
    parser.add_argument(
        "--skip-pit-stops",
        action="store_true",
        help="Skip real pit-stop fetching and keep V0 placeholder pit-stop features.",
    )
    parser.add_argument(
        "--pit-stop-workers",
        type=int,
        default=4,
        help="Parallel workers used for pit-stop race fetching.",
    )
    return parser.parse_args()


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: list[float], default: float = 0.0) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return default
    return float(sum(clean) / len(clean))


def best_or_default(values: list[int], default: int = 0) -> int:
    clean = [value for value in values if value]
    return min(clean) if clean else default


def worst_or_default(values: list[int], default: int = 0) -> int:
    clean = [value for value in values if value]
    return max(clean) if clean else default


def race_id(season: int, round_number: int) -> str:
    return f"{season}_{round_number:02d}"


def time_to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("+"):
        text = text[1:]
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[-3]) if len(parts) == 3 else 0
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return None


def is_classified(status: str) -> bool:
    status_text = str(status).lower()
    return status_text == "finished" or status_text.startswith("+")


def is_dns(status: str, grid: int) -> bool:
    status_text = str(status).lower()
    return "did not start" in status_text or status_text == "dns"


def is_disqualified(status: str) -> bool:
    return "disqualified" in str(status).lower()


def is_accident_dnf(status: str) -> bool:
    status_text = str(status).lower()
    markers = ["accident", "collision", "spun", "damage", "crash"]
    return any(marker in status_text for marker in markers)


def is_mechanical_dnf(status: str) -> bool:
    status_text = str(status).lower()
    markers = [
        "engine",
        "gearbox",
        "hydraulics",
        "electrical",
        "power unit",
        "transmission",
        "brakes",
        "suspension",
        "clutch",
        "fuel",
        "oil",
        "water",
        "overheating",
        "mechanical",
        "battery",
        "turbo",
        "exhaust",
    ]
    return any(marker in status_text for marker in markers)


def jolpica_get(session: requests.Session, endpoint: str, sleep_seconds: float) -> dict[str, Any]:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    last_error: Exception | None = None

    for attempt in range(5):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 429:
                retry_after = as_float(response.headers.get("Retry-After"), default=0.0)
                wait_seconds = retry_after if retry_after > 0 else min(60.0, 5.0 * (attempt + 1))
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            time.sleep(sleep_seconds)
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(30.0, 2.0 * (attempt + 1)))

    raise RuntimeError(f"Unable to fetch {url}") from last_error


def merge_paginated_races(races: list[dict[str, Any]], result_key: str) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for race in races:
        key = (str(race.get("season")), str(race.get("round")))
        if key not in merged:
            merged[key] = dict(race)
            merged[key][result_key] = list(race.get(result_key, []))
        else:
            merged[key][result_key].extend(race.get(result_key, []))

    return sorted(
        merged.values(),
        key=lambda race: (as_int(race.get("season")), as_int(race.get("round"))),
    )


def fetch_races(session: requests.Session, season: int, table: str, sleep_seconds: float) -> list[dict[str, Any]]:
    result_key = "Results" if table == "results" else "QualifyingResults"
    all_races: list[dict[str, Any]] = []
    offset = 0
    limit = 100
    total = None

    while total is None or offset < total:
        payload = jolpica_get(
            session,
            f"{season}/{table}.json?limit={limit}&offset={offset}",
            sleep_seconds,
        )
        mr_data = payload.get("MRData", {})
        total = as_int(mr_data.get("total"))
        response_limit = as_int(mr_data.get("limit"), default=limit)
        races = mr_data.get("RaceTable", {}).get("Races", [])

        if not races:
            break

        all_races.extend(races)
        offset += response_limit

    return merge_paginated_races(all_races, result_key)


def fetch_single_race_records(
    session: requests.Session,
    season: int,
    round_number: int,
    table: str,
    result_key: str,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    offset = 0
    limit = 100
    total = None
    merged_race: dict[str, Any] | None = None

    while total is None or offset < total:
        payload = jolpica_get(
            session,
            f"{season}/{round_number}/{table}.json?limit={limit}&offset={offset}",
            sleep_seconds,
        )
        mr_data = payload.get("MRData", {})
        total = as_int(mr_data.get("total"))
        response_limit = as_int(mr_data.get("limit"), default=limit)
        races = mr_data.get("RaceTable", {}).get("Races", [])

        if not races:
            break

        race = races[0]
        if merged_race is None:
            merged_race = dict(race)
            merged_race[result_key] = list(race.get(result_key, []))
        else:
            merged_race[result_key].extend(race.get(result_key, []))

        offset += response_limit

    return merged_race


def fetch_pit_stop_races(
    result_races: list[dict[str, Any]],
    sleep_seconds: float,
    workers: int,
) -> list[dict[str, Any]]:
    races: list[dict[str, Any]] = []
    race_refs = [
        (as_int(race.get("season")), as_int(race.get("round")))
        for race in result_races
    ]

    def fetch_one(race_ref: tuple[int, int]) -> dict[str, Any] | None:
        season, round_number = race_ref
        with requests.Session() as worker_session:
            return fetch_single_race_records(
                session=worker_session,
                season=season,
                round_number=round_number,
                table="pitstops",
                result_key="PitStops",
                sleep_seconds=sleep_seconds,
            )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(fetch_one, race_ref) for race_ref in race_refs]

        for index, future in enumerate(as_completed(futures), start=1):
            pit_stop_race = future.result()
            if pit_stop_race is not None:
                races.append(pit_stop_race)

            if index % 25 == 0:
                print(f"Fetched pit stops for {index}/{len(result_races)} races")

    return sorted(
        races,
        key=lambda race: (as_int(race.get("season")), as_int(race.get("round"))),
    )


def rank_entities(points: dict[str, float], wins: dict[str, int], entities: set[str]) -> dict[str, int]:
    if not entities or not any(points.get(entity, 0.0) for entity in entities):
        return {entity: 0 for entity in entities}

    ordered = sorted(
        entities,
        key=lambda entity: (-points.get(entity, 0.0), -wins.get(entity, 0), entity),
    )
    return {entity: index + 1 for index, entity in enumerate(ordered)}


def track_type(circuit_id: str) -> str:
    if circuit_id in STREET_CIRCUITS:
        return "street"
    return "permanent"


def overtaking_difficulty(circuit_id: str) -> str:
    if circuit_id in HARD_TO_OVERTAKE:
        return "hard"
    if circuit_id in EASY_TO_OVERTAKE:
        return "easy"
    return "medium"


def parse_result_races(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for race in races:
        season = as_int(race.get("season"))
        round_number = as_int(race.get("round"))
        circuit = race.get("Circuit", {})
        location = circuit.get("Location", {})
        current_race_id = race_id(season, round_number)

        for result in race.get("Results", []):
            driver = result.get("Driver", {})
            constructor = result.get("Constructor", {})
            fastest_lap = result.get("FastestLap", {}) or {}
            fastest_lap_time = (fastest_lap.get("Time") or {}).get("time")
            fastest_lap_speed = (fastest_lap.get("AverageSpeed") or {}).get("speed")
            final_position = as_int(result.get("position"))

            rows.append(
                {
                    "race_id": current_race_id,
                    "season": season,
                    "round": round_number,
                    "grand_prix": race.get("raceName"),
                    "race_date": race.get("date"),
                    "circuit_id": circuit.get("circuitId"),
                    "circuit_name": circuit.get("circuitName"),
                    "country": location.get("country"),
                    "locality": location.get("locality"),
                    "driver_id": driver.get("driverId"),
                    "driver_code": driver.get("code"),
                    "driver_name": f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip(),
                    "driver_nationality": driver.get("nationality"),
                    "driver_date_of_birth": driver.get("dateOfBirth"),
                    "constructor_id": constructor.get("constructorId"),
                    "constructor_name": constructor.get("name"),
                    "constructor_nationality": constructor.get("nationality"),
                    "grid": as_int(result.get("grid")),
                    "final_position": final_position,
                    "position_text": result.get("positionText"),
                    "points": as_float(result.get("points")),
                    "laps": as_int(result.get("laps")),
                    "status": result.get("status"),
                    "fastest_lap_rank": as_int(fastest_lap.get("rank")),
                    "fastest_lap_number": as_int(fastest_lap.get("lap")),
                    "fastest_lap_time": fastest_lap_time,
                    "fastest_lap_seconds": time_to_seconds(fastest_lap_time),
                    "fastest_lap_avg_speed_kph": as_float(fastest_lap_speed),
                    "top10_finish": int(1 <= final_position <= 10),
                }
            )

    return rows


def parse_qualifying_races(races: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for race in races:
        season = as_int(race.get("season"))
        round_number = as_int(race.get("round"))
        current_race_id = race_id(season, round_number)
        race_rows: list[dict[str, Any]] = []

        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            q1_seconds = time_to_seconds(result.get("Q1"))
            q2_seconds = time_to_seconds(result.get("Q2"))
            q3_seconds = time_to_seconds(result.get("Q3"))
            valid_times = [value for value in [q1_seconds, q2_seconds, q3_seconds] if value is not None]
            best_seconds = min(valid_times) if valid_times else None

            race_rows.append(
                {
                    "race_id": current_race_id,
                    "driver_id": driver.get("driverId"),
                    "qualifying_position": as_int(result.get("position")),
                    "q1_seconds": q1_seconds,
                    "q2_seconds": q2_seconds,
                    "q3_seconds": q3_seconds,
                    "best_qualifying_seconds": best_seconds,
                    "reached_q2": int(q2_seconds is not None),
                    "reached_q3": int(q3_seconds is not None),
                }
            )

        pole_time = min(
            [row["best_qualifying_seconds"] for row in race_rows if row["best_qualifying_seconds"] is not None],
            default=None,
        )
        for row in race_rows:
            if pole_time is None or row["best_qualifying_seconds"] is None:
                row["qualifying_gap_to_pole_seconds"] = None
            else:
                row["qualifying_gap_to_pole_seconds"] = row["best_qualifying_seconds"] - pole_time
            rows.append(row)

    return rows


def parse_pit_stop_races(races: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for race in races:
        season = as_int(race.get("season"))
        round_number = as_int(race.get("round"))
        current_race_id = race_id(season, round_number)

        for pit_stop in race.get("PitStops", []):
            rows.append(
                {
                    "race_id": current_race_id,
                    "driver_id": pit_stop.get("driverId"),
                    "pit_lap": as_int(pit_stop.get("lap")),
                    "pit_stop_number": as_int(pit_stop.get("stop")),
                    "pit_duration_seconds": as_float(pit_stop.get("duration")),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "race_id",
                "driver_id",
                "pit_lap",
                "pit_stop_number",
                "pit_duration_seconds",
            ]
        )

    return pd.DataFrame(rows)


def summarize_current_pit_stops(pit_stop_events: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "race_id",
        "driver_id",
        "current_race_pit_stops",
        "current_avg_pit_stop_time",
        "current_best_pit_stop_time",
        "current_worst_pit_stop_time",
        "current_total_pit_time",
        "current_pit_stop_rank",
    ]

    if pit_stop_events.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        pit_stop_events.groupby(["race_id", "driver_id"], as_index=False)
        .agg(
            current_race_pit_stops=("pit_stop_number", "max"),
            current_avg_pit_stop_time=("pit_duration_seconds", "mean"),
            current_best_pit_stop_time=("pit_duration_seconds", "min"),
            current_worst_pit_stop_time=("pit_duration_seconds", "max"),
            current_total_pit_time=("pit_duration_seconds", "sum"),
        )
        .fillna(0)
    )
    summary["current_pit_stop_rank"] = summary.groupby("race_id")["current_total_pit_time"].rank(
        method="average",
        ascending=True,
    )
    return summary[columns]


def previous_values(history: deque[dict[str, Any]], key: str) -> list[Any]:
    return [entry.get(key) for entry in history]


def build_feature_tables(
    race_results: pd.DataFrame,
    pit_stop_summary: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    sorted_results = race_results.sort_values(["season", "round", "final_position", "driver_id"]).copy()
    if pit_stop_summary is None:
        pit_stop_summary = pd.DataFrame()
    pit_lookup = {
        (str(row["race_id"]), str(row["driver_id"])): row.to_dict()
        for _, row in pit_stop_summary.iterrows()
    }

    driver_points: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    driver_wins: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    constructor_points: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    constructor_wins: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    season_drivers: dict[int, set[str]] = defaultdict(set)
    season_constructors: dict[int, set[str]] = defaultdict(set)
    driver_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=5))
    speed_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=3))
    pit_history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=3))

    driver_standings_rows: list[dict[str, Any]] = []
    constructor_standings_rows: list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []
    pit_rows: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []

    for _, race_group in sorted_results.groupby(["season", "round"], sort=True):
        season = as_int(race_group["season"].iloc[0])
        race = race_group["race_id"].iloc[0]
        current_drivers = set(race_group["driver_id"].astype(str))
        current_constructors = set(race_group["constructor_id"].astype(str))
        season_drivers[season].update(current_drivers)
        season_constructors[season].update(current_constructors)

        driver_rank = rank_entities(driver_points[season], driver_wins[season], season_drivers[season])
        constructor_rank = rank_entities(
            constructor_points[season],
            constructor_wins[season],
            season_constructors[season],
        )

        for _, result in race_group.iterrows():
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            history = driver_history[driver_id]
            speed_window = speed_history[driver_id]
            pit_window = pit_history[driver_id]

            previous_positions = [as_int(value) for value in previous_values(history, "final_position")]
            previous_grids = [as_int(value) for value in previous_values(history, "grid")]
            previous_laps = [as_int(value) for value in previous_values(history, "laps")]
            previous_points = [as_float(value) for value in previous_values(history, "points")]
            previous_top10 = [as_int(value) for value in previous_values(history, "top10_finish")]
            previous_dnfs = [as_int(value) for value in previous_values(history, "dnf")]
            previous_dns = [as_int(value) for value in previous_values(history, "dns")]
            previous_mechanical = [as_int(value) for value in previous_values(history, "mechanical_dnf")]
            previous_accident = [as_int(value) for value in previous_values(history, "accident_dnf")]
            previous_disq = [as_int(value) for value in previous_values(history, "disqualified")]

            race_count = len(history)
            top10_count = sum(previous_top10)
            dnf_count = sum(previous_dnfs)
            dns_count = sum(previous_dns)
            disq_count = sum(previous_disq)

            driver_standings_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "driver_position_before_race": driver_rank.get(driver_id, 0),
                    "driver_points_before_race": driver_points[season].get(driver_id, 0.0),
                    "driver_wins_before_race": driver_wins[season].get(driver_id, 0),
                    "driver_performance_score": driver_points[season].get(driver_id, 0.0)
                    + 10 * driver_wins[season].get(driver_id, 0),
                }
            )

            constructor_standings_rows.append(
                {
                    "race_id": race,
                    "constructor_id": constructor_id,
                    "constructor_position_before_race": constructor_rank.get(constructor_id, 0),
                    "constructor_points_before_race": constructor_points[season].get(constructor_id, 0.0),
                    "constructor_wins_before_race": constructor_wins[season].get(constructor_id, 0),
                    "constructor_performance_score": constructor_points[season].get(constructor_id, 0.0)
                    + 10 * constructor_wins[season].get(constructor_id, 0),
                }
            )

            form_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "races_count_previous_5": race_count,
                    "top10_finishes_previous_5": top10_count,
                    "top10_rate_previous_5": top10_count / race_count if race_count else 0.0,
                    "points_previous_5": sum(previous_points),
                    "avg_points_previous_5": mean(previous_points),
                    "points_finishes_previous_5": sum(1 for value in previous_points if value > 0),
                    "avg_finish_position_previous_5": mean(previous_positions),
                    "avg_grid_position_previous_5": mean(previous_grids),
                    "best_finish_previous_5": best_or_default(previous_positions),
                    "worst_finish_previous_5": worst_or_default(previous_positions),
                    "avg_laps_completed_previous_5": mean(previous_laps),
                    "dnf_previous_5": dnf_count,
                }
            )

            finish_count = race_count - dnf_count - dns_count - disq_count
            reliability_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "dnf_previous_5": dnf_count,
                    "dns_previous_5": dns_count,
                    "mechanical_dnf_previous_5": sum(previous_mechanical),
                    "accident_dnf_previous_5": sum(previous_accident),
                    "disqualified_previous_5": disq_count,
                    "finish_rate_previous_5": finish_count / race_count if race_count else 1.0,
                    "reliability_score": (finish_count / race_count if race_count else 1.0)
                    - 0.1 * sum(previous_mechanical)
                    - 0.05 * sum(previous_accident),
                }
            )

            previous_lap_seconds = [
                entry["fastest_lap_seconds"]
                for entry in speed_window
                if entry.get("fastest_lap_seconds") is not None
            ]
            previous_lap_ranks = [
                entry["fastest_lap_rank"] for entry in speed_window if as_int(entry.get("fastest_lap_rank")) > 0
            ]
            previous_speeds = [
                entry["fastest_lap_avg_speed_kph"]
                for entry in speed_window
                if as_float(entry.get("fastest_lap_avg_speed_kph")) > 0
            ]
            previous_completed_laps = [
                entry["laps"] for entry in speed_window if as_int(entry.get("laps")) > 0
            ]
            previous_pit_races = list(pit_window)
            previous_pit_avg_times = [
                as_float(entry.get("current_avg_pit_stop_time"))
                for entry in previous_pit_races
                if as_float(entry.get("current_avg_pit_stop_time")) > 0
            ]
            previous_pit_best_times = [
                as_float(entry.get("current_best_pit_stop_time"))
                for entry in previous_pit_races
                if as_float(entry.get("current_best_pit_stop_time")) > 0
            ]
            previous_pit_worst_times = [
                as_float(entry.get("current_worst_pit_stop_time"))
                for entry in previous_pit_races
                if as_float(entry.get("current_worst_pit_stop_time")) > 0
            ]
            previous_total_pit_times = [
                as_float(entry.get("current_total_pit_time"))
                for entry in previous_pit_races
                if as_float(entry.get("current_total_pit_time")) > 0
            ]
            previous_pit_ranks = [
                as_float(entry.get("current_pit_stop_rank"))
                for entry in previous_pit_races
                if as_float(entry.get("current_pit_stop_rank")) > 0
            ]
            previous_pit_counts = [
                as_int(entry.get("current_race_pit_stops"))
                for entry in previous_pit_races
                if as_int(entry.get("current_race_pit_stops")) > 0
            ]

            lap_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "races_with_lap_data_previous_3": len(previous_lap_seconds),
                    "avg_lap_time_previous_3_races": mean(previous_lap_seconds),
                    "best_lap_time_previous_3_races": min(previous_lap_seconds) if previous_lap_seconds else 0.0,
                    "lap_time_std_previous_3_races": pd.Series(previous_lap_seconds).std(ddof=0)
                    if previous_lap_seconds
                    else 0.0,
                    "avg_race_pace_rank_previous_3_races": mean(previous_lap_ranks),
                    "avg_completed_laps_previous_3_races": mean(previous_completed_laps),
                }
            )

            pit_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "races_with_pit_data_previous_3": len(previous_pit_races),
                    "avg_pit_stop_time_previous_3_races": mean(previous_pit_avg_times),
                    "best_pit_stop_time_previous_3_races": min(previous_pit_best_times)
                    if previous_pit_best_times
                    else 0.0,
                    "worst_pit_stop_time_previous_3_races": max(previous_pit_worst_times)
                    if previous_pit_worst_times
                    else 0.0,
                    "total_pit_stops_previous_3_races": sum(previous_pit_counts),
                    "avg_total_pit_time_previous_3_races": mean(previous_total_pit_times),
                    "avg_pit_stop_rank_previous_3_races": mean(previous_pit_ranks),
                }
            )

            telemetry_rows.append(
                {
                    "race_id": race,
                    "driver_id": driver_id,
                    "telemetry_races_previous_3": len(previous_speeds),
                    "avg_speed_previous_3_races": mean(previous_speeds),
                    "max_speed_previous_3_races": max(previous_speeds) if previous_speeds else 0.0,
                    "avg_throttle_previous_3_races": 0.0,
                    "avg_brake_previous_3_races": 0.0,
                    "avg_drs_previous_3_races": 0.0,
                    "drs_usage_rate_previous_3_races": 0.0,
                    "avg_distance_previous_3_races": mean(previous_completed_laps),
                }
            )

        for _, result in race_group.iterrows():
            driver_id = str(result["driver_id"])
            constructor_id = str(result["constructor_id"])
            points = as_float(result["points"])
            status = str(result["status"])
            grid = as_int(result["grid"])
            final_position = as_int(result["final_position"])
            dnf = int(not is_classified(status) and not is_dns(status, grid) and not is_disqualified(status))
            dns = int(is_dns(status, grid))
            disqualified = int(is_disqualified(status))
            accident_dnf = int(dnf and is_accident_dnf(status))
            mechanical_dnf = int(dnf and is_mechanical_dnf(status))

            driver_points[season][driver_id] += points
            constructor_points[season][constructor_id] += points
            if final_position == 1:
                driver_wins[season][driver_id] += 1
                constructor_wins[season][constructor_id] += 1

            history_entry = {
                "final_position": final_position,
                "grid": grid,
                "points": points,
                "laps": as_int(result["laps"]),
                "top10_finish": as_int(result["top10_finish"]),
                "dnf": dnf,
                "dns": dns,
                "mechanical_dnf": mechanical_dnf,
                "accident_dnf": accident_dnf,
                "disqualified": disqualified,
                "fastest_lap_seconds": result.get("fastest_lap_seconds"),
                "fastest_lap_rank": result.get("fastest_lap_rank"),
                "fastest_lap_avg_speed_kph": result.get("fastest_lap_avg_speed_kph"),
            }
            driver_history[driver_id].append(history_entry)
            speed_history[driver_id].append(history_entry)
            pit_history[driver_id].append(
                pit_lookup.get(
                    (str(result["race_id"]), driver_id),
                    {
                        "current_race_pit_stops": 0,
                        "current_avg_pit_stop_time": 0.0,
                        "current_best_pit_stop_time": 0.0,
                        "current_worst_pit_stop_time": 0.0,
                        "current_total_pit_time": 0.0,
                        "current_pit_stop_rank": 0.0,
                    },
                )
            )

    return {
        "driver_standings": pd.DataFrame(driver_standings_rows),
        "constructor_standings": pd.DataFrame(constructor_standings_rows).drop_duplicates(
            ["race_id", "constructor_id"]
        ),
        "form_data": pd.DataFrame(form_rows),
        "reliability_data": pd.DataFrame(reliability_rows),
        "lap_times": pd.DataFrame(lap_rows),
        "pit_stops": pd.DataFrame(pit_rows),
        "telemetry_data": pd.DataFrame(telemetry_rows),
    }


def build_driver_info(race_results: pd.DataFrame) -> pd.DataFrame:
    grouped = race_results.groupby("driver_id", dropna=False)
    rows = []

    for driver_id, group in grouped:
        seasons = sorted(group["season"].dropna().astype(int).unique())
        first_row = group.sort_values(["season", "round"]).iloc[0]
        birth_date = first_row.get("driver_date_of_birth")
        age_at_2026 = None
        if isinstance(birth_date, str) and birth_date:
            age_at_2026 = 2026 - datetime.strptime(birth_date, "%Y-%m-%d").year

        rows.append(
            {
                "driver_id": driver_id,
                "nationality": first_row.get("driver_nationality"),
                "date_of_birth": birth_date,
                "first_season_in_dataset": seasons[0],
                "last_season_in_dataset": seasons[-1],
                "number_of_seasons_in_dataset": len(seasons),
                "age_at_2026": age_at_2026,
                "rookie_in_dataset": int(len(seasons) == 1),
            }
        )

    return pd.DataFrame(rows)


def build_team_info(race_results: pd.DataFrame) -> pd.DataFrame:
    grouped = race_results.groupby("constructor_id", dropna=False)
    rows = []

    for constructor_id, group in grouped:
        seasons = sorted(group["season"].dropna().astype(int).unique())
        first_row = group.sort_values(["season", "round"]).iloc[0]
        rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_nationality": first_row.get("constructor_nationality"),
                "first_season_in_dataset": seasons[0],
                "last_season_in_dataset": seasons[-1],
                "number_of_seasons_in_dataset": len(seasons),
                "team_experience_score": len(seasons) / max(1, DEFAULT_END_YEAR - DEFAULT_START_YEAR + 1),
            }
        )

    return pd.DataFrame(rows)


def build_circuit_info(race_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for race, group in race_results.groupby("race_id", sort=True):
        first_row = group.iloc[0]
        circuit_id = str(first_row.get("circuit_id"))
        total_laps = int(group["laps"].max())
        circuit_length = CIRCUIT_LENGTH_KM.get(circuit_id, 0.0)

        rows.append(
            {
                "race_id": race,
                "circuit_length_km": circuit_length,
                "total_laps": total_laps,
                "race_distance_km": circuit_length * total_laps if circuit_length else 0.0,
                "track_type": track_type(circuit_id),
                "overtaking_difficulty": overtaking_difficulty(circuit_id),
                "street_circuit": int(circuit_id in STREET_CIRCUITS),
            }
        )

    return pd.DataFrame(rows)


def build_weather_data(race_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for race in sorted(race_results["race_id"].unique()):
        rows.append(
            {
                "race_id": race,
                "air_temp_mean": 20.0,
                "air_temp_min": 20.0,
                "air_temp_max": 20.0,
                "track_temp_mean": 30.0,
                "track_temp_min": 30.0,
                "track_temp_max": 30.0,
                "humidity_mean": 50.0,
                "pressure_mean": 1013.0,
                "wind_speed_mean": 0.0,
                "wind_direction_mean": 0.0,
                "rainfall": 0.0,
                "rainfall_percentage": 0.0,
                "wet_race": 0,
                "weather_condition": "Unknown",
                "weather_data_available": 0,
            }
        )

    return pd.DataFrame(rows)


def build_race_control_data(race_results: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for race, group in race_results.groupby("race_id", sort=True):
        dnf_count = 0
        incident_count = 0
        classified_count = 0

        for _, result in group.iterrows():
            status = str(result["status"])
            grid = as_int(result["grid"])
            if is_classified(status):
                classified_count += 1
            elif not is_dns(status, grid):
                dnf_count += 1
            if is_accident_dnf(status):
                incident_count += 1

        rows.append(
            {
                "race_id": race,
                "safety_car_count": 0,
                "virtual_safety_car_count": 0,
                "red_flag_count": 0,
                "yellow_flag_count": 0,
                "double_yellow_count": 0,
                "black_flag_count": 0,
                "track_limits_count": 0,
                "investigation_count": 0,
                "penalty_count": 0,
                "incident_count": incident_count,
                "race_control_messages_count": 0,
                "race_control_data_available": 0,
                "total_dnf_count": dnf_count,
                "classified_driver_count": classified_count,
                "race_disruption_score": dnf_count + incident_count,
            }
        )

    return pd.DataFrame(rows)


def write_csv(name: str, df: pd.DataFrame) -> None:
    path = RAW_PATH / name
    df.to_csv(path, index=False)
    print(f"Wrote {path}: {len(df)} rows, {len(df.columns)} columns")


def main() -> None:
    args = parse_args()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be greater than or equal to --start-year")

    session = requests.Session()
    all_result_races: list[dict[str, Any]] = []
    all_qualifying_races: list[dict[str, Any]] = []

    for season in range(args.start_year, args.end_year + 1):
        result_races = fetch_races(session, season, "results", args.sleep)
        qualifying_races = fetch_races(session, season, "qualifying", args.sleep)
        print(
            f"Fetched {season}: {len(result_races)} result races, "
            f"{len(qualifying_races)} qualifying races"
        )
        all_result_races.extend(result_races)
        all_qualifying_races.extend(qualifying_races)

    if args.skip_pit_stops:
        pit_stop_races: list[dict[str, Any]] = []
    else:
        print(f"Fetching pit stops for {len(all_result_races)} races...")
        pit_stop_races = fetch_pit_stop_races(
            all_result_races,
            args.sleep,
            workers=args.pit_stop_workers,
        )

    result_rows = parse_result_races(all_result_races)
    if not result_rows:
        raise RuntimeError("No race results were fetched. Check the API or the selected year range.")

    race_results = pd.DataFrame(result_rows)
    qualifying_results = pd.DataFrame(parse_qualifying_races(all_qualifying_races))
    pit_stop_events = parse_pit_stop_races(pit_stop_races)
    pit_stop_summary = summarize_current_pit_stops(pit_stop_events)
    feature_tables = build_feature_tables(race_results, pit_stop_summary=pit_stop_summary)

    write_csv("race_results.csv", race_results)
    write_csv("qualifying_results.csv", qualifying_results)
    write_csv("driver_info.csv", build_driver_info(race_results))
    write_csv("team_info.csv", build_team_info(race_results))
    write_csv("circuit_info.csv", build_circuit_info(race_results))
    write_csv("constructor_standings.csv", feature_tables["constructor_standings"])
    write_csv("driver_standings.csv", feature_tables["driver_standings"])
    write_csv("form_data.csv", feature_tables["form_data"])
    write_csv("reliability_data.csv", feature_tables["reliability_data"])
    write_csv("lap_times.csv", feature_tables["lap_times"])
    write_csv("pit_stops.csv", feature_tables["pit_stops"])
    write_csv("pit_stop_events.csv", pit_stop_events)
    write_csv("weather_data.csv", build_weather_data(race_results))
    write_csv("race_control_messages.csv", build_race_control_data(race_results))
    write_csv("telemetry_data.csv", feature_tables["telemetry_data"])


if __name__ == "__main__":
    main()
