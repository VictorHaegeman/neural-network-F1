from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from generate_raw_data import (
    CIRCUIT_LENGTH_KM,
    EASY_TO_OVERTAKE,
    HARD_TO_OVERTAKE,
    STREET_CIRCUITS,
    is_accident_dnf,
    is_classified,
    is_disqualified,
    is_dns,
    is_mechanical_dnf,
    time_to_seconds,
)
from train_model import DATA_PATH, MODEL_PATH, build_features


JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OUTPUT_DIR = Path("outputs/predictions")

LEAKAGE_DEFAULTS: dict[str, Any] = {
    "final_position": 0,
    "position_text": "PRED",
    "points": 0.0,
    "laps": 0,
    "status": "Pre-race prediction",
    "fastest_lap_rank": 0,
    "fastest_lap_number": 0,
    "fastest_lap_time": "Unknown",
    "fastest_lap_seconds": 0.0,
    "fastest_lap_avg_speed_kph": 0.0,
    "top10_finish": 0,
    "safety_car_count": 0,
    "virtual_safety_car_count": 0,
    "red_flag_count": 0,
    "yellow_flag_count": 0,
    "double_yellow_count": 0,
    "black_flag_count": 0,
    "track_limits_count": 0,
    "investigation_count": 0,
    "penalty_count": 0,
    "incident_count": 0,
    "race_control_messages_count": 0,
    "race_control_data_available": 0,
    "total_dnf_count": 0,
    "classified_driver_count": 0,
    "race_disruption_score": 0,
    "fastf1_race_control_available": 0,
    "fastf1_race_control_messages_count": 0,
    "fastf1_safety_car_count": 0,
    "fastf1_virtual_safety_car_count": 0,
    "fastf1_red_flag_count": 0,
    "fastf1_yellow_flag_count": 0,
    "fastf1_double_yellow_count": 0,
    "fastf1_black_flag_count": 0,
    "fastf1_track_limits_count": 0,
    "fastf1_investigation_count": 0,
    "fastf1_penalty_count": 0,
    "fastf1_incident_count": 0,
    "fastf1_drs_disabled_count": 0,
    "fastf1_deleted_lap_count": 0,
    "fastf1_slippery_track_count": 0,
    "fastf1_clear_count": 0,
    "fastf1_race_disruption_score": 0,
}

RACE_CONTROL_HISTORY_COLUMNS = [
    "race_control_history_available",
    "circuit_race_control_races_previous_3",
    "circuit_safety_car_rate_previous_3",
    "circuit_vsc_rate_previous_3",
    "circuit_red_flag_rate_previous_3",
    "circuit_yellow_flags_avg_previous_3",
    "circuit_incidents_avg_previous_3",
    "circuit_penalties_avg_previous_3",
    "circuit_track_limits_avg_previous_3",
    "circuit_race_disruption_score_avg_previous_3",
    "season_race_control_races_previous_3",
    "season_safety_car_rate_previous_3",
    "season_vsc_rate_previous_3",
    "season_red_flag_rate_previous_3",
    "season_race_disruption_score_avg_previous_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict top-10 finishers for upcoming F1 races using pre-race features."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--position-model",
        type=Path,
        default=None,
        help="Optional finish-position model used to add predicted finishing order.",
    )
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--count", type=int, default=4, help="Number of upcoming races to score.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--current-date",
        type=str,
        default=date.today().isoformat(),
        help="Date used to select upcoming races, format YYYY-MM-DD.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Disable live Open-Meteo forecast calls and use historical circuit weather fallback.",
    )
    return parser.parse_args()


def as_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value) or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_mean(values: Any, default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.mean())


def safe_min(values: Any, default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.min())


def safe_max(values: Any, default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return default
    return float(series.max())


def slugify(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return text.strip("_") or "race"


def jolpica_get(session: requests.Session, endpoint: str) -> dict[str, Any]:
    response = session.get(f"{JOLPICA_BASE_URL}/{endpoint.lstrip('/')}", timeout=30)
    response.raise_for_status()
    return response.json()


def race_id(season: int, round_number: int) -> str:
    return f"{season}_{round_number:02d}"


def fetch_schedule(session: requests.Session, season: int) -> list[dict[str, Any]]:
    payload = jolpica_get(session, f"{season}/races.json?limit=100")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    return sorted(races, key=lambda item: as_int(item.get("round")))


def race_to_record(race: dict[str, Any]) -> dict[str, Any]:
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})
    season = as_int(race.get("season"))
    round_number = as_int(race.get("round"))
    race_date = str(race.get("date", ""))

    return {
        "race_id": race_id(season, round_number),
        "season": season,
        "round": round_number,
        "grand_prix": race.get("raceName", "Unknown Grand Prix"),
        "race_date": race_date,
        "race_time_utc": race.get("time", ""),
        "circuit_id": circuit.get("circuitId", "unknown"),
        "circuit_name": circuit.get("circuitName", "Unknown Circuit"),
        "country": location.get("country", "Unknown"),
        "locality": location.get("locality", "Unknown"),
        "latitude": as_float(location.get("lat"), np.nan),
        "longitude": as_float(location.get("long"), np.nan),
    }


def choose_upcoming_races(
    schedule: list[dict[str, Any]],
    df: pd.DataFrame,
    season: int,
    current_date: date,
    count: int,
) -> list[dict[str, Any]]:
    completed_round = 0
    season_rows = df[df["season"] == season]
    if not season_rows.empty:
        completed_round = int(season_rows["round"].max())

    records = [race_to_record(race) for race in schedule]
    upcoming = [
        race
        for race in records
        if race["round"] > completed_round
        or datetime.strptime(race["race_date"], "%Y-%m-%d").date() >= current_date
    ]
    upcoming = sorted(upcoming, key=lambda item: (item["season"], item["round"]))
    return upcoming[:count]


def fetch_qualifying(session: requests.Session, season: int, round_number: int) -> dict[str, dict[str, Any]]:
    payload = jolpica_get(session, f"{season}/{round_number}/qualifying.json?limit=100")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return {}

    rows: list[dict[str, Any]] = []
    for result in races[0].get("QualifyingResults", []):
        driver = result.get("Driver", {})
        q_times = [
            time_to_seconds(result.get("Q1")),
            time_to_seconds(result.get("Q2")),
            time_to_seconds(result.get("Q3")),
        ]
        clean_times = [value for value in q_times if value is not None]
        best_time = min(clean_times) if clean_times else None
        rows.append(
            {
                "driver_id": driver.get("driverId"),
                "qualifying_position": as_int(result.get("position")),
                "q1_seconds": q_times[0],
                "q2_seconds": q_times[1],
                "q3_seconds": q_times[2],
                "best_qualifying_seconds": best_time,
                "reached_q2": int(q_times[1] is not None),
                "reached_q3": int(q_times[2] is not None),
            }
        )

    pole_time = min(
        [row["best_qualifying_seconds"] for row in rows if row["best_qualifying_seconds"] is not None],
        default=None,
    )
    for row in rows:
        if pole_time is None or row["best_qualifying_seconds"] is None:
            row["qualifying_gap_to_pole_seconds"] = 0.0
        else:
            row["qualifying_gap_to_pole_seconds"] = row["best_qualifying_seconds"] - pole_time

    return {str(row["driver_id"]): row for row in rows if row.get("driver_id")}


def circuit_defaults(circuit_id: str) -> dict[str, Any]:
    length = CIRCUIT_LENGTH_KM.get(circuit_id, 5.0)
    if circuit_id in HARD_TO_OVERTAKE:
        overtaking = "hard"
    elif circuit_id in EASY_TO_OVERTAKE:
        overtaking = "easy"
    else:
        overtaking = "medium"

    return {
        "circuit_length_km": length,
        "total_laps": int(round(305 / length)) if length else 60,
        "race_distance_km": float(length * int(round(305 / length))) if length else 305.0,
        "track_type": "street" if circuit_id in STREET_CIRCUITS else "permanent",
        "overtaking_difficulty": overtaking,
        "street_circuit": int(circuit_id in STREET_CIRCUITS),
    }


def circuit_features(df: pd.DataFrame, circuit_id: str) -> dict[str, Any]:
    columns = [
        "circuit_length_km",
        "total_laps",
        "race_distance_km",
        "track_type",
        "overtaking_difficulty",
        "street_circuit",
    ]
    history = df[df["circuit_id"] == circuit_id]
    defaults = circuit_defaults(circuit_id)

    if history.empty:
        return defaults

    latest = history.sort_values(["season", "round"]).iloc[-1]
    return {column: latest.get(column, defaults[column]) for column in columns}


def historical_weather(df: pd.DataFrame, circuit_id: str) -> tuple[dict[str, Any], str]:
    weather_columns = [
        "air_temp_mean",
        "air_temp_min",
        "air_temp_max",
        "track_temp_mean",
        "track_temp_min",
        "track_temp_max",
        "humidity_mean",
        "pressure_mean",
        "wind_speed_mean",
        "wind_direction_mean",
        "rainfall",
        "rainfall_percentage",
        "wet_race",
        "weather_condition",
        "weather_data_available",
        "fastf1_weather_available",
        "fastf1_air_temp_mean",
        "fastf1_air_temp_min",
        "fastf1_air_temp_max",
        "fastf1_track_temp_mean",
        "fastf1_track_temp_min",
        "fastf1_track_temp_max",
        "fastf1_humidity_mean",
        "fastf1_pressure_mean",
        "fastf1_wind_speed_mean",
        "fastf1_wind_direction_mean",
        "fastf1_rainfall_rate",
        "fastf1_wet_race",
    ]
    history = df[df["circuit_id"] == circuit_id]
    if history.empty:
        return (
            {
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
                "fastf1_weather_available": 0,
                "fastf1_air_temp_mean": 0.0,
                "fastf1_air_temp_min": 0.0,
                "fastf1_air_temp_max": 0.0,
                "fastf1_track_temp_mean": 0.0,
                "fastf1_track_temp_min": 0.0,
                "fastf1_track_temp_max": 0.0,
                "fastf1_humidity_mean": 0.0,
                "fastf1_pressure_mean": 0.0,
                "fastf1_wind_speed_mean": 0.0,
                "fastf1_wind_direction_mean": 0.0,
                "fastf1_rainfall_rate": 0.0,
                "fastf1_wet_race": 0,
            },
            "default_placeholder",
        )

    numeric = history.select_dtypes(include=["number", "bool"])
    row: dict[str, Any] = {}
    for column in weather_columns:
        if column not in history.columns:
            continue
        if column in numeric.columns:
            row[column] = safe_mean(history[column])
        else:
            mode = history[column].mode(dropna=True)
            row[column] = mode.iloc[0] if not mode.empty else "Unknown"

    row["weather_data_available"] = int(as_float(row.get("weather_data_available"), 0) > 0)
    row["fastf1_weather_available"] = int(as_float(row.get("fastf1_weather_available"), 0) > 0)
    return row, "circuit_history"


def historical_race_control(df: pd.DataFrame, circuit_id: str, season: int) -> dict[str, Any]:
    if "fastf1_race_control_available" not in df.columns:
        return {column: 0.0 for column in RACE_CONTROL_HISTORY_COLUMNS}

    races = (
        df.drop_duplicates("race_id")
        .sort_values(["season", "round"])
        .copy()
    )
    available = pd.to_numeric(
        races["fastf1_race_control_available"],
        errors="coerce",
    ).fillna(0) > 0
    races = races[available]

    circuit_history = races[races["circuit_id"] == circuit_id].tail(3)
    season_history = races[races["season"] == season].tail(3)

    def mean_value(history: pd.DataFrame, column: str) -> float:
        if history.empty or column not in history.columns:
            return 0.0
        return safe_mean(history[column], 0.0)

    def rate_value(history: pd.DataFrame, column: str) -> float:
        if history.empty or column not in history.columns:
            return 0.0
        values = pd.to_numeric(history[column], errors="coerce").fillna(0)
        return float((values > 0).mean()) if len(values) else 0.0

    return {
        "race_control_history_available": int(not circuit_history.empty or not season_history.empty),
        "circuit_race_control_races_previous_3": int(len(circuit_history)),
        "circuit_safety_car_rate_previous_3": rate_value(circuit_history, "fastf1_safety_car_count"),
        "circuit_vsc_rate_previous_3": rate_value(circuit_history, "fastf1_virtual_safety_car_count"),
        "circuit_red_flag_rate_previous_3": rate_value(circuit_history, "fastf1_red_flag_count"),
        "circuit_yellow_flags_avg_previous_3": mean_value(circuit_history, "fastf1_yellow_flag_count"),
        "circuit_incidents_avg_previous_3": mean_value(circuit_history, "fastf1_incident_count"),
        "circuit_penalties_avg_previous_3": mean_value(circuit_history, "fastf1_penalty_count"),
        "circuit_track_limits_avg_previous_3": mean_value(circuit_history, "fastf1_track_limits_count"),
        "circuit_race_disruption_score_avg_previous_3": mean_value(
            circuit_history,
            "fastf1_race_disruption_score",
        ),
        "season_race_control_races_previous_3": int(len(season_history)),
        "season_safety_car_rate_previous_3": rate_value(season_history, "fastf1_safety_car_count"),
        "season_vsc_rate_previous_3": rate_value(season_history, "fastf1_virtual_safety_car_count"),
        "season_red_flag_rate_previous_3": rate_value(season_history, "fastf1_red_flag_count"),
        "season_race_disruption_score_avg_previous_3": mean_value(
            season_history,
            "fastf1_race_disruption_score",
        ),
    }


def fetch_weather_forecast(race: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    if pd.isna(race.get("latitude")) or pd.isna(race.get("longitude")):
        return None, "missing_coordinates"

    params = {
        "latitude": race["latitude"],
        "longitude": race["longitude"],
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation_probability",
                "precipitation",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_probability_max",
                "wind_speed_10m_max",
            ]
        ),
        "start_date": race["race_date"],
        "end_date": race["race_date"],
        "timezone": "UTC",
    }
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly.get("time"):
        return None, "forecast_unavailable"

    temperature = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    pressure = hourly.get("surface_pressure", [])
    wind_speed = hourly.get("wind_speed_10m", [])
    wind_direction = hourly.get("wind_direction_10m", [])
    rain_probability = hourly.get("precipitation_probability", [])
    precipitation = hourly.get("precipitation", [])

    rain_rate = safe_mean(precipitation)
    rain_probability_max = safe_max(rain_probability)
    air_mean = safe_mean(temperature, 20.0)
    air_min = safe_min(temperature, air_mean)
    air_max = safe_max(temperature, air_mean)
    wet_race = int(rain_rate > 0.05 or rain_probability_max >= 50)

    forecast = {
        "air_temp_mean": air_mean,
        "air_temp_min": air_min,
        "air_temp_max": air_max,
        "track_temp_mean": air_mean + 10.0,
        "track_temp_min": air_min + 8.0,
        "track_temp_max": air_max + 12.0,
        "humidity_mean": safe_mean(humidity, 50.0),
        "pressure_mean": safe_mean(pressure, 1013.0),
        "wind_speed_mean": safe_mean(wind_speed, 0.0),
        "wind_direction_mean": safe_mean(wind_direction, 0.0),
        "rainfall": rain_rate,
        "rainfall_percentage": rain_probability_max,
        "wet_race": wet_race,
        "weather_condition": "Forecast rain risk" if wet_race else "Forecast dry",
        "weather_data_available": 1,
        "fastf1_weather_available": 0,
        "fastf1_air_temp_mean": air_mean,
        "fastf1_air_temp_min": air_min,
        "fastf1_air_temp_max": air_max,
        "fastf1_track_temp_mean": air_mean + 10.0,
        "fastf1_track_temp_min": air_min + 8.0,
        "fastf1_track_temp_max": air_max + 12.0,
        "fastf1_humidity_mean": safe_mean(humidity, 50.0),
        "fastf1_pressure_mean": safe_mean(pressure, 1013.0),
        "fastf1_wind_speed_mean": safe_mean(wind_speed, 0.0),
        "fastf1_wind_direction_mean": safe_mean(wind_direction, 0.0),
        "fastf1_rainfall_rate": rain_rate,
        "fastf1_wet_race": wet_race,
    }
    return forecast, "open_meteo_forecast"


def active_driver_rows(df: pd.DataFrame, season: int) -> pd.DataFrame:
    season_rows = df[df["season"] == season]
    if season_rows.empty:
        season_rows = df[df["season"] == df["season"].max()]

    latest_round = int(season_rows["round"].max())
    latest_race = season_rows[season_rows["round"] == latest_round].copy()
    latest_race = latest_race.sort_values(["constructor_name", "driver_name"])
    return latest_race.drop_duplicates("driver_id", keep="last")


def standings_after_latest_race(df: pd.DataFrame, season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    season_rows = df[df["season"] == season].copy()
    if season_rows.empty:
        season_rows = df[df["season"] == df["season"].max()].copy()

    driver = (
        season_rows.groupby("driver_id", as_index=False)
        .agg(
            driver_points_before_race=("points", "sum"),
            driver_wins_before_race=("final_position", lambda values: int((pd.to_numeric(values) == 1).sum())),
            avg_finish=("final_position", "mean"),
        )
        .sort_values(
            ["driver_points_before_race", "driver_wins_before_race", "avg_finish"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    driver["driver_position_before_race"] = driver.index + 1
    max_points = max(float(driver["driver_points_before_race"].max()), 1.0)
    driver["driver_performance_score"] = driver["driver_points_before_race"] / max_points

    constructor = (
        season_rows.groupby("constructor_id", as_index=False)
        .agg(
            constructor_points_before_race=("points", "sum"),
            constructor_wins_before_race=("final_position", lambda values: int((pd.to_numeric(values) == 1).sum())),
            avg_finish=("final_position", "mean"),
        )
        .sort_values(
            ["constructor_points_before_race", "constructor_wins_before_race", "avg_finish"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    constructor["constructor_position_before_race"] = constructor.index + 1
    max_constructor_points = max(float(constructor["constructor_points_before_race"].max()), 1.0)
    constructor["constructor_performance_score"] = (
        constructor["constructor_points_before_race"] / max_constructor_points
    )
    return driver, constructor


def recent_driver_features(history: pd.DataFrame, driver_id: str) -> dict[str, Any]:
    driver_history = history[history["driver_id"] == driver_id].sort_values(["season", "round"]).tail(5)
    if driver_history.empty:
        return {}

    final_position = pd.to_numeric(driver_history["final_position"], errors="coerce").replace(0, np.nan)
    grid = pd.to_numeric(driver_history["grid"], errors="coerce").replace(0, np.nan)
    laps = pd.to_numeric(driver_history["laps"], errors="coerce").replace(0, np.nan)
    points = pd.to_numeric(driver_history["points"], errors="coerce").fillna(0)
    top10 = pd.to_numeric(driver_history["top10_finish"], errors="coerce").fillna(0)

    classified = [
        is_classified(row["status"])
        for _, row in driver_history.iterrows()
    ]
    dns_count = int(
        sum(is_dns(row["status"], as_int(row.get("grid"))) for _, row in driver_history.iterrows())
    )
    mechanical_count = int(sum(is_mechanical_dnf(row["status"]) for _, row in driver_history.iterrows()))
    accident_count = int(sum(is_accident_dnf(row["status"]) for _, row in driver_history.iterrows()))
    disqualified_count = int(sum(is_disqualified(row["status"]) for _, row in driver_history.iterrows()))
    dnf_count = int(len(classified) - sum(classified))
    finish_rate = float(sum(classified) / max(1, len(classified)))

    return {
        "races_count_previous_5": int(len(driver_history)),
        "top10_finishes_previous_5": int(top10.sum()),
        "top10_rate_previous_5": float(top10.mean()),
        "points_previous_5": float(points.sum()),
        "avg_points_previous_5": float(points.mean()),
        "points_finishes_previous_5": int((points > 0).sum()),
        "avg_finish_position_previous_5": float(final_position.mean()) if not final_position.dropna().empty else 20.0,
        "avg_grid_position_previous_5": float(grid.mean()) if not grid.dropna().empty else 20.0,
        "best_finish_previous_5": int(final_position.min()) if not final_position.dropna().empty else 20,
        "worst_finish_previous_5": int(final_position.max()) if not final_position.dropna().empty else 20,
        "avg_laps_completed_previous_5": float(laps.mean()) if not laps.dropna().empty else 0.0,
        "form_dnf_previous_5": dnf_count,
        "dnf_previous_5": dnf_count,
        "dns_previous_5": dns_count,
        "mechanical_dnf_previous_5": mechanical_count,
        "accident_dnf_previous_5": accident_count,
        "disqualified_previous_5": disqualified_count,
        "finish_rate_previous_5": finish_rate,
        "reliability_score": finish_rate,
    }


def estimated_qualifying_order(rows: pd.DataFrame) -> dict[str, int]:
    score = (
        pd.to_numeric(rows.get("driver_points_before_race", 0), errors="coerce").fillna(0) * 1.0
        + pd.to_numeric(rows.get("constructor_points_before_race", 0), errors="coerce").fillna(0) * 0.45
        + pd.to_numeric(rows.get("top10_rate_previous_5", 0), errors="coerce").fillna(0) * 35.0
        - pd.to_numeric(rows.get("avg_finish_position_previous_5", 20), errors="coerce").fillna(20) * 0.4
    )
    ranked = rows.assign(_estimated_score=score).sort_values(
        ["_estimated_score", "driver_name"], ascending=[False, True]
    )
    return {str(driver_id): rank for rank, driver_id in enumerate(ranked["driver_id"], start=1)}


def apply_qualifying_features(
    race_rows: pd.DataFrame,
    qualifying: dict[str, dict[str, Any]],
) -> tuple[pd.DataFrame, str]:
    race_rows = race_rows.copy()
    estimated_order = estimated_qualifying_order(race_rows)
    has_actual_qualifying = bool(qualifying)

    for index, row in race_rows.iterrows():
        driver_id = str(row["driver_id"])
        estimated_position = estimated_order.get(driver_id, len(race_rows))
        actual = qualifying.get(driver_id)

        if actual:
            position = as_int(actual.get("qualifying_position"), estimated_position)
            race_rows.at[index, "qualifying_position"] = position
            race_rows.at[index, "grid"] = position
            for column in [
                "q1_seconds",
                "q2_seconds",
                "q3_seconds",
                "best_qualifying_seconds",
                "qualifying_gap_to_pole_seconds",
                "reached_q2",
                "reached_q3",
            ]:
                race_rows.at[index, column] = actual.get(column, row.get(column, 0))
        else:
            race_rows.at[index, "qualifying_position"] = estimated_position
            race_rows.at[index, "grid"] = estimated_position
            base_lap = safe_mean(race_rows.get("best_qualifying_seconds", pd.Series(dtype=float)), 90.0)
            gap = max(0, estimated_position - 1) * 0.18
            race_rows.at[index, "q1_seconds"] = base_lap + gap
            race_rows.at[index, "q2_seconds"] = base_lap + gap if estimated_position <= 15 else 0.0
            race_rows.at[index, "q3_seconds"] = base_lap + gap if estimated_position <= 10 else 0.0
            race_rows.at[index, "best_qualifying_seconds"] = base_lap + gap
            race_rows.at[index, "qualifying_gap_to_pole_seconds"] = gap
            race_rows.at[index, "reached_q2"] = int(estimated_position <= 15)
            race_rows.at[index, "reached_q3"] = int(estimated_position <= 10)

    source = "jolpica_actual_qualifying" if has_actual_qualifying else "estimated_strength_order"
    return race_rows, source


def build_upcoming_race_rows(
    df: pd.DataFrame,
    race: dict[str, Any],
    active_rows: pd.DataFrame,
    driver_standings: pd.DataFrame,
    constructor_standings: pd.DataFrame,
    weather: dict[str, Any],
    weather_source: str,
) -> pd.DataFrame:
    rows = active_rows.copy()
    race_date = datetime.strptime(race["race_date"], "%Y-%m-%d")

    for column, value in race.items():
        if column in rows.columns:
            rows[column] = value

    for column, value in {
        "race_id": race["race_id"],
        "season": race["season"],
        "round": race["round"],
        "grand_prix": race["grand_prix"],
        "race_date": race["race_date"],
        "circuit_id": race["circuit_id"],
        "circuit_name": race["circuit_name"],
        "country": race["country"],
        "locality": race["locality"],
        "race_month": race_date.month,
        "race_day": race_date.day,
    }.items():
        rows[column] = value

    for column, value in circuit_features(df, race["circuit_id"]).items():
        rows[column] = value

    for column, value in LEAKAGE_DEFAULTS.items():
        if column in rows.columns:
            rows[column] = value

    for column, value in weather.items():
        if column in rows.columns:
            rows[column] = value

    for column, value in historical_race_control(df, race["circuit_id"], race["season"]).items():
        if column in rows.columns:
            rows[column] = value

    rows = rows.drop(
        columns=[
            column
            for column in [
                "driver_position_before_race",
                "driver_points_before_race",
                "driver_wins_before_race",
                "driver_performance_score",
            ]
            if column in rows.columns
        ]
    ).merge(driver_standings.drop(columns=["avg_finish"], errors="ignore"), on="driver_id", how="left")

    rows = rows.drop(
        columns=[
            column
            for column in [
                "constructor_position_before_race",
                "constructor_points_before_race",
                "constructor_wins_before_race",
                "constructor_performance_score",
            ]
            if column in rows.columns
        ]
    ).merge(constructor_standings.drop(columns=["avg_finish"], errors="ignore"), on="constructor_id", how="left")

    recent_features = [
        recent_driver_features(df, str(driver_id))
        for driver_id in rows["driver_id"]
    ]
    recent_df = pd.DataFrame(recent_features)
    for column in recent_df.columns:
        rows[column] = recent_df[column].values

    rows["prediction_weather_source"] = weather_source
    return rows


def output_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "race_id",
        "season",
        "round",
        "grand_prix",
        "race_date",
        "circuit_name",
        "driver_code",
        "driver_name",
        "constructor_name",
        "grid",
        "qualifying_position",
        "top10_probability",
        "top10_probability_rank",
        "predicted_top10_by_probability",
        "predicted_rank",
        "predicted_top10",
        "predicted_finish_position_raw",
        "predicted_finish_rank",
        "predicted_top10_position",
        "driver_points_before_race",
        "constructor_points_before_race",
        "top10_rate_previous_5",
        "avg_finish_position_previous_5",
        "air_temp_mean",
        "rainfall_percentage",
        "wet_race",
        "weather_condition",
        "fastf1_avg_tyre_life_previous_3",
        "fastf1_stint_count_previous_3",
        "fastf1_soft_lap_rate_previous_3",
        "fastf1_medium_lap_rate_previous_3",
        "fastf1_hard_lap_rate_previous_3",
        "circuit_safety_car_rate_previous_3",
        "circuit_race_disruption_score_avg_previous_3",
        "prediction_weather_source",
        "prediction_qualifying_source",
    ]
    return [column for column in preferred if column in df.columns]


def main() -> None:
    args = parse_args()
    current_date = datetime.strptime(args.current_date, "%Y-%m-%d").date()

    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing model: {args.model}. Run scripts/train_model.py first.")
    if args.position_model is not None and not args.position_model.exists():
        raise FileNotFoundError(
            f"Missing position model: {args.position_model}. Run scripts/train_position_model.py first."
        )

    df = pd.read_csv(args.data)
    model = joblib.load(args.model)
    position_model = joblib.load(args.position_model) if args.position_model is not None else None

    session = requests.Session()
    schedule = fetch_schedule(session, args.season)
    upcoming_races = choose_upcoming_races(schedule, df, args.season, current_date, args.count)
    if not upcoming_races:
        raise ValueError(f"No upcoming races found for season {args.season}.")

    active_rows = active_driver_rows(df, args.season)
    driver_standings, constructor_standings = standings_after_latest_race(df, args.season)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined_outputs: list[pd.DataFrame] = []
    notes: list[dict[str, Any]] = []

    for race in upcoming_races:
        fallback_weather, fallback_weather_source = historical_weather(df, race["circuit_id"])
        if args.no_weather:
            weather, weather_source = fallback_weather, fallback_weather_source
        else:
            try:
                forecast_weather, forecast_source = fetch_weather_forecast(race)
                weather = forecast_weather or fallback_weather
                weather_source = forecast_source if forecast_weather else fallback_weather_source
            except requests.RequestException:
                weather, weather_source = fallback_weather, fallback_weather_source

        race_rows = build_upcoming_race_rows(
            df=df,
            race=race,
            active_rows=active_rows,
            driver_standings=driver_standings,
            constructor_standings=constructor_standings,
            weather=weather,
            weather_source=weather_source,
        )
        qualifying = fetch_qualifying(session, race["season"], race["round"])
        race_rows, qualifying_source = apply_qualifying_features(race_rows, qualifying)
        race_rows["prediction_qualifying_source"] = qualifying_source

        missing_columns = [column for column in df.columns if column not in race_rows.columns]
        if missing_columns:
            raise ValueError(f"Upcoming race rows are missing training columns: {missing_columns}")

        probabilities = model.predict_proba(build_features(race_rows[df.columns]))[:, 1]
        race_rows["top10_probability"] = probabilities
        race_rows = race_rows.sort_values("top10_probability", ascending=False).reset_index(drop=True)
        race_rows["predicted_rank"] = race_rows.index + 1
        race_rows["predicted_top10"] = (race_rows["predicted_rank"] <= args.top_n).astype(int)

        if position_model is not None:
            race_rows["top10_probability_rank"] = race_rows["predicted_rank"]
            race_rows["predicted_top10_by_probability"] = race_rows["predicted_top10"]
            raw_positions = np.clip(position_model.predict(build_features(race_rows[df.columns])), 1, len(race_rows))
            race_rows["predicted_finish_position_raw"] = raw_positions
            race_rows["predicted_finish_rank"] = (
                race_rows["predicted_finish_position_raw"].rank(method="first", ascending=True).astype(int)
            )
            race_rows["predicted_top10_position"] = race_rows["predicted_finish_rank"].where(
                race_rows["predicted_finish_rank"] <= args.top_n,
                0,
            )
            race_rows["predicted_rank"] = race_rows["predicted_finish_rank"]
            race_rows["predicted_top10"] = (race_rows["predicted_finish_rank"] <= args.top_n).astype(int)
            race_rows = race_rows.sort_values("predicted_finish_rank").reset_index(drop=True)

        readable = race_rows[output_columns(race_rows)].copy()
        readable_path = (
            args.output_dir
            / f"upcoming_top10_predictions_{race['season']}_round{race['round']:02d}_{slugify(race['grand_prix'])}.csv"
        )
        readable.to_csv(readable_path, index=False)
        combined_outputs.append(readable)

        notes.append(
            {
                "race_id": race["race_id"],
                "grand_prix": race["grand_prix"],
                "race_date": race["race_date"],
                "weather_source": weather_source,
                "qualifying_source": qualifying_source,
                "output": str(readable_path),
            }
        )

        print(f"\n{race['grand_prix']} ({race['race_date']})")
        print(f"Weather source: {weather_source}")
        print(f"Qualifying source: {qualifying_source}")
        print(readable.head(args.top_n).to_string(index=False))

    combined = pd.concat(combined_outputs, ignore_index=True)
    combined_path = args.output_dir / "upcoming_top10_predictions.csv"
    notes_path = args.output_dir / "upcoming_prediction_notes.json"
    combined.to_csv(combined_path, index=False)
    with notes_path.open("w", encoding="utf-8") as file:
        json.dump(notes, file, indent=2)

    print(f"\nCombined predictions written to {combined_path}")
    print(f"Prediction notes written to {notes_path}")


if __name__ == "__main__":
    main()
