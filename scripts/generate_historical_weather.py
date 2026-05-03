from __future__ import annotations

import argparse
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_PATH = Path("data/raw")
RACE_RESULTS_PATH = RAW_PATH / "race_results.csv"
OUTPUT_PATH = RAW_PATH / "weather_data.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch historical race-day weather from Open-Meteo Archive.")
    parser.add_argument("--start-year", type=int, default=2011)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--race-results", type=Path, default=RACE_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
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


def safe_mean(values: list[Any], default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else default


def safe_min(values: list[Any], default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.min()) if not series.empty else default


def safe_max(values: list[Any], default: float = 0.0) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.max()) if not series.empty else default


def race_id(season: int, round_number: int) -> str:
    return f"{season}_{round_number:02d}"


def jolpica_get(session: requests.Session, endpoint: str) -> dict[str, Any]:
    response = session.get(f"{JOLPICA_BASE_URL}/{endpoint.lstrip('/')}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_schedule(session: requests.Session, start_year: int, end_year: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season in range(start_year, end_year + 1):
        payload = jolpica_get(session, f"{season}/races.json?limit=100")
        races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            circuit = race.get("Circuit", {})
            location = circuit.get("Location", {})
            rows.append(
                {
                    "race_id": race_id(as_int(race.get("season")), as_int(race.get("round"))),
                    "season": as_int(race.get("season")),
                    "round": as_int(race.get("round")),
                    "grand_prix": race.get("raceName", "Unknown Grand Prix"),
                    "race_date": race.get("date"),
                    "circuit_id": circuit.get("circuitId", "unknown"),
                    "latitude": as_float(location.get("lat")),
                    "longitude": as_float(location.get("long")),
                }
            )
        print(f"Loaded Jolpica schedule {season}: {len(races)} races")
    return pd.DataFrame(rows)


def weather_condition(rainfall: float, rainfall_percentage: float) -> str:
    if rainfall > 4:
        return "Rain"
    if rainfall > 0.2:
        return "Light rain"
    if rainfall_percentage > 0:
        return "Dry with recorded precipitation nearby"
    return "Dry"


def fetch_open_meteo_weather(
    session: requests.Session,
    race: pd.Series,
) -> dict[str, Any]:
    params = {
        "latitude": float(race["latitude"]),
        "longitude": float(race["longitude"]),
        "start_date": str(race["race_date"]),
        "end_date": str(race["race_date"]),
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "rain",
            ]
        ),
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "precipitation_hours",
                "wind_speed_10m_max",
            ]
        ),
        "timezone": "UTC",
    }
    response = session.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    daily = payload.get("daily", {})

    air_temp = hourly.get("temperature_2m", [])
    humidity = hourly.get("relative_humidity_2m", [])
    pressure = hourly.get("surface_pressure", [])
    wind_speed = hourly.get("wind_speed_10m", [])
    wind_direction = hourly.get("wind_direction_10m", [])
    precipitation = hourly.get("precipitation", [])
    precipitation_series = pd.to_numeric(pd.Series(precipitation), errors="coerce").fillna(0)

    air_mean = safe_mean(air_temp, 20.0)
    air_min = safe_min(air_temp, air_mean)
    air_max = safe_max(air_temp, air_mean)
    hourly_rainfall_total = float(precipitation_series.sum())
    rainfall = safe_mean(daily.get("precipitation_sum", []), hourly_rainfall_total)
    rain_hours = safe_mean(
        daily.get("precipitation_hours", []),
        float((precipitation_series > 0).sum()),
    )
    rainfall_percentage = min(100.0, max(0.0, rain_hours / 24.0 * 100.0))

    return {
        "race_id": race["race_id"],
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
        "rainfall": rainfall,
        "rainfall_percentage": rainfall_percentage,
        "wet_race": int(rainfall > 0.05 or rain_hours > 0),
        "weather_condition": weather_condition(rainfall, rainfall_percentage),
        "weather_data_available": 1,
    }


def placeholder_weather(race_id_value: str) -> dict[str, Any]:
    return {
        "race_id": race_id_value,
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


def main() -> None:
    args = parse_args()
    if not args.race_results.exists():
        raise FileNotFoundError(f"Missing race results: {args.race_results}")

    race_results = pd.read_csv(args.race_results)
    race_ids = set(race_results["race_id"].astype(str).unique())

    session = requests.Session()
    schedule = fetch_schedule(session, args.start_year, args.end_year)
    schedule = schedule[schedule["race_id"].isin(race_ids)].sort_values(["season", "round"])

    rows: list[dict[str, Any]] = []
    total = len(schedule)
    for index, race in enumerate(schedule.itertuples(index=False), start=1):
        race_series = pd.Series(race._asdict())
        label = f"{int(race.season)} round {int(race.round)} - {race.grand_prix}"
        print(f"Open-Meteo {index}/{total}: {label}")
        try:
            rows.append(fetch_open_meteo_weather(session, race_series))
        except Exception as exc:
            print(f"Skipped weather for {race.race_id}: {exc}")
            rows.append(placeholder_weather(str(race.race_id)))
        time.sleep(args.sleep)

    weather = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(args.output, index=False)

    available = int(weather["weather_data_available"].sum()) if not weather.empty else 0
    print(f"Wrote {args.output}: {weather.shape[0]} rows, {weather.shape[1]} columns")
    print(f"Weather available: {available}/{len(weather)} races")


if __name__ == "__main__":
    main()
