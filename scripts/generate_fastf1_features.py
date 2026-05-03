from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd


RAW_PATH = Path("data/raw")
RACE_RESULTS_PATH = RAW_PATH / "race_results.csv"
CACHE_PATH = Path(".fastf1_cache")
FASTF1_WEATHER_PATH = RAW_PATH / "fastf1_weather.csv"
FASTF1_LAP_SUMMARIES_PATH = RAW_PATH / "fastf1_lap_summaries.csv"
FASTF1_DRIVER_FORM_PATH = RAW_PATH / "fastf1_driver_form.csv"

ROLLING_WINDOW = 3
SUPPORTED_START_YEAR = 2018
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optional FastF1 feature tables.")
    parser.add_argument("--start-year", type=int, default=SUPPORTED_START_YEAR)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--max-races", type=int, default=None)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_PATH)
    parser.add_argument("--race-results", type=Path, default=RACE_RESULTS_PATH)
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Keep existing FastF1 rows and fetch only missing races in the requested range.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --incremental, refetch races even if they already exist locally.",
    )
    return parser.parse_args()


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else 0.0


def safe_min(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.min()) if not numeric.empty else 0.0


def safe_max(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else 0.0


def timedelta_seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def get_existing_races(race_results: pd.DataFrame, start_year: int, end_year: int | None) -> pd.DataFrame:
    races = (
        race_results[["race_id", "season", "round", "grand_prix"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .copy()
    )
    races = races[races["season"] >= max(start_year, SUPPORTED_START_YEAR)]
    if end_year is not None:
        races = races[races["season"] <= end_year]
    return races


def build_driver_code_lookup(race_results: pd.DataFrame) -> dict[tuple[str, str], str]:
    lookup = {}
    for _, row in race_results.iterrows():
        driver_code = str(row.get("driver_code", "")).upper()
        if driver_code and driver_code != "NAN":
            lookup[(str(row["race_id"]), driver_code)] = str(row["driver_id"])
    return lookup


def summarize_weather(race_id: str, weather: pd.DataFrame) -> dict[str, Any]:
    if weather is None or weather.empty:
        return {
            "race_id": race_id,
            "fastf1_weather_available": 0,
        }

    rainfall = pd.to_numeric(weather.get("Rainfall", pd.Series(dtype=float)), errors="coerce").fillna(0)
    return {
        "race_id": race_id,
        "fastf1_weather_available": 1,
        "fastf1_air_temp_mean": safe_mean(weather.get("AirTemp", pd.Series(dtype=float))),
        "fastf1_air_temp_min": safe_min(weather.get("AirTemp", pd.Series(dtype=float))),
        "fastf1_air_temp_max": safe_max(weather.get("AirTemp", pd.Series(dtype=float))),
        "fastf1_track_temp_mean": safe_mean(weather.get("TrackTemp", pd.Series(dtype=float))),
        "fastf1_track_temp_min": safe_min(weather.get("TrackTemp", pd.Series(dtype=float))),
        "fastf1_track_temp_max": safe_max(weather.get("TrackTemp", pd.Series(dtype=float))),
        "fastf1_humidity_mean": safe_mean(weather.get("Humidity", pd.Series(dtype=float))),
        "fastf1_pressure_mean": safe_mean(weather.get("Pressure", pd.Series(dtype=float))),
        "fastf1_wind_speed_mean": safe_mean(weather.get("WindSpeed", pd.Series(dtype=float))),
        "fastf1_wind_direction_mean": safe_mean(weather.get("WindDirection", pd.Series(dtype=float))),
        "fastf1_rainfall_rate": float(rainfall.mean()) if len(rainfall) else 0.0,
        "fastf1_wet_race": int((rainfall > 0).any()),
    }


def summarize_laps(race_id: str, laps: pd.DataFrame, driver_lookup: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    if laps is None or laps.empty:
        return []

    laps = laps.copy()
    laps["Driver"] = laps["Driver"].astype(str).str.upper()
    laps["lap_time_seconds"] = timedelta_seconds(laps["LapTime"])
    laps = laps[laps["lap_time_seconds"].notna()].copy()

    if "Deleted" in laps.columns:
        deleted = laps["Deleted"].astype("boolean").fillna(False)
        laps = laps[~deleted]
    if "FastF1Generated" in laps.columns:
        laps = laps[~laps["FastF1Generated"].fillna(False).astype(bool)]

    rows: list[dict[str, Any]] = []
    for driver_code, group in laps.groupby("Driver"):
        driver_id = driver_lookup.get((race_id, driver_code))
        if not driver_id:
            continue

        compound = group.get("Compound", pd.Series(dtype=object)).astype(str).str.upper()
        position = pd.to_numeric(group.get("Position", pd.Series(dtype=float)), errors="coerce")
        tyre_life = pd.to_numeric(group.get("TyreLife", pd.Series(dtype=float)), errors="coerce")

        row: dict[str, Any] = {
            "race_id": race_id,
            "driver_id": driver_id,
            "fastf1_current_lap_data_available": 1,
            "fastf1_current_lap_count": int(len(group)),
            "fastf1_current_accurate_lap_count": int(group.get("IsAccurate", pd.Series(dtype=bool)).fillna(False).sum()),
            "fastf1_current_avg_lap_time": safe_mean(group["lap_time_seconds"]),
            "fastf1_current_best_lap_time": safe_min(group["lap_time_seconds"]),
            "fastf1_current_lap_time_std": float(group["lap_time_seconds"].std(ddof=0) or 0.0),
            "fastf1_current_avg_position": float(position.mean()) if not position.dropna().empty else 0.0,
            "fastf1_current_best_position": float(position.min()) if not position.dropna().empty else 0.0,
            "fastf1_current_worst_position": float(position.max()) if not position.dropna().empty else 0.0,
            "fastf1_current_avg_tyre_life": float(tyre_life.mean()) if not tyre_life.dropna().empty else 0.0,
            "fastf1_current_stint_count": int(pd.to_numeric(group.get("Stint", pd.Series(dtype=float)), errors="coerce").nunique()),
            "fastf1_current_compound_count": int(compound.replace("NAN", np.nan).dropna().nunique()),
            "fastf1_current_wet_lap_rate": float(compound.isin(["INTERMEDIATE", "WET"]).mean()) if len(compound) else 0.0,
        }

        for compound_name in COMPOUNDS:
            row[f"fastf1_current_{compound_name.lower()}_lap_rate"] = (
                float((compound == compound_name).mean()) if len(compound) else 0.0
            )

        for speed_col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
            if speed_col in group.columns:
                row[f"fastf1_current_{speed_col.lower()}_mean"] = safe_mean(group[speed_col])

        rows.append(row)

    return rows


def fetch_fastf1_race(
    season: int,
    round_number: int,
    race_id: str,
    driver_lookup: dict[tuple[str, str], str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = fastf1.get_session(season, round_number, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    weather_row = summarize_weather(race_id, session.weather_data)
    lap_rows = summarize_laps(race_id, session.laps, driver_lookup)
    return weather_row, lap_rows


def rolling_mean(history: deque[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in history if float(row.get(key, 0.0)) > 0]
    return float(sum(values) / len(values)) if values else 0.0


def rolling_min(history: deque[dict[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in history if float(row.get(key, 0.0)) > 0]
    return float(min(values)) if values else 0.0


def build_driver_form(race_results: pd.DataFrame, lap_summaries: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row["race_id"]), str(row["driver_id"])): row.to_dict()
        for _, row in lap_summaries.iterrows()
    }
    history: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    rows: list[dict[str, Any]] = []

    base = (
        race_results[["race_id", "season", "round", "driver_id"]]
        .drop_duplicates()
        .sort_values(["season", "round", "driver_id"])
    )

    for _, row in base.iterrows():
        race_id = str(row["race_id"])
        driver_id = str(row["driver_id"])
        driver_history = history[driver_id]

        feature_row: dict[str, Any] = {
            "race_id": race_id,
            "driver_id": driver_id,
            "fastf1_races_with_lap_data_previous_3": len(driver_history),
            "fastf1_avg_lap_time_previous_3": rolling_mean(driver_history, "fastf1_current_avg_lap_time"),
            "fastf1_best_lap_time_previous_3": rolling_min(driver_history, "fastf1_current_best_lap_time"),
            "fastf1_lap_time_std_previous_3": rolling_mean(driver_history, "fastf1_current_lap_time_std"),
            "fastf1_avg_position_previous_3": rolling_mean(driver_history, "fastf1_current_avg_position"),
            "fastf1_best_position_previous_3": rolling_min(driver_history, "fastf1_current_best_position"),
            "fastf1_avg_tyre_life_previous_3": rolling_mean(driver_history, "fastf1_current_avg_tyre_life"),
            "fastf1_stint_count_previous_3": rolling_mean(driver_history, "fastf1_current_stint_count"),
            "fastf1_compound_count_previous_3": rolling_mean(driver_history, "fastf1_current_compound_count"),
            "fastf1_wet_lap_rate_previous_3": rolling_mean(driver_history, "fastf1_current_wet_lap_rate"),
        }

        for compound_name in COMPOUNDS:
            feature_row[f"fastf1_{compound_name.lower()}_lap_rate_previous_3"] = rolling_mean(
                driver_history,
                f"fastf1_current_{compound_name.lower()}_lap_rate",
            )

        for speed_col in ["speedi1", "speedi2", "speedfl", "speedst"]:
            feature_row[f"fastf1_{speed_col}_mean_previous_3"] = rolling_mean(
                driver_history,
                f"fastf1_current_{speed_col}_mean",
            )

        rows.append(feature_row)

        current = lookup.get((race_id, driver_id))
        if current:
            driver_history.append(current)

    return pd.DataFrame(rows)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)
    print(f"Wrote {path}: {df.shape[0]} rows, {df.shape[1]} columns")


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

    combined = combined.drop_duplicates(keys, keep="last")
    sort_columns = [column for column in ["race_id", "driver_id"] if column in combined.columns]
    if sort_columns:
        combined = combined.sort_values(sort_columns)
    return combined.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.race_results.exists():
        raise FileNotFoundError(f"Missing race results: {args.race_results}")

    race_results = pd.read_csv(args.race_results)
    races = get_existing_races(race_results, args.start_year, args.end_year)

    existing_weather = load_existing(FASTF1_WEATHER_PATH) if args.incremental else pd.DataFrame()
    existing_laps = load_existing(FASTF1_LAP_SUMMARIES_PATH) if args.incremental else pd.DataFrame()
    existing_race_ids = set(existing_weather.get("race_id", pd.Series(dtype=str)).astype(str))
    if args.incremental and not args.force:
        before_count = len(races)
        races = races[~races["race_id"].astype(str).isin(existing_race_ids)].copy()
        print(f"Incremental mode skipped {before_count - len(races)} already fetched races")

    if args.max_races is not None:
        races = races.head(args.max_races)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))
    fastf1.set_log_level("WARNING")

    driver_lookup = build_driver_code_lookup(race_results)
    weather_rows: list[dict[str, Any]] = []
    lap_rows: list[dict[str, Any]] = []

    for index, race in enumerate(races.itertuples(index=False), start=1):
        print(f"FastF1 {index}/{len(races)}: {race.season} round {race.round} - {race.grand_prix}")
        try:
            weather_row, current_lap_rows = fetch_fastf1_race(
                season=int(race.season),
                round_number=int(race.round),
                race_id=str(race.race_id),
                driver_lookup=driver_lookup,
            )
            weather_rows.append(weather_row)
            lap_rows.extend(current_lap_rows)
        except Exception as exc:
            print(f"Skipped {race.season} round {race.round}: {exc}")
            weather_rows.append(
                {
                    "race_id": str(race.race_id),
                    "fastf1_weather_available": 0,
                }
            )

    weather_df = combine_existing(existing_weather, pd.DataFrame(weather_rows), ["race_id"])
    lap_summaries_df = combine_existing(existing_laps, pd.DataFrame(lap_rows), ["race_id", "driver_id"])
    driver_form_df = build_driver_form(race_results, lap_summaries_df)

    write_csv(FASTF1_WEATHER_PATH, weather_df)
    write_csv(FASTF1_LAP_SUMMARIES_PATH, lap_summaries_df)
    write_csv(FASTF1_DRIVER_FORM_PATH, driver_form_df)


if __name__ == "__main__":
    main()
