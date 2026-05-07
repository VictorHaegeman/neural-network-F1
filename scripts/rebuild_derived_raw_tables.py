from __future__ import annotations

from pathlib import Path

import pandas as pd

from generate_raw_data import (
    RAW_PATH,
    build_circuit_info,
    build_driver_info,
    build_feature_tables,
    build_race_control_data,
    build_team_info,
    build_weather_data,
    summarize_current_pit_stops,
    write_csv,
)
from import_missing_completed_races import preserve_existing_race_table


def load_csv(name: str) -> pd.DataFrame:
    path = RAW_PATH / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    race_results = load_csv("race_results.csv")
    pit_stop_events = load_csv("pit_stop_events.csv")
    if race_results.empty:
        raise FileNotFoundError("Missing or empty data/raw/race_results.csv")

    pit_stop_summary = summarize_current_pit_stops(pit_stop_events)
    feature_tables = build_feature_tables(race_results, pit_stop_summary=pit_stop_summary)

    write_csv("driver_info.csv", build_driver_info(race_results))
    write_csv("team_info.csv", build_team_info(race_results))
    write_csv("circuit_info.csv", build_circuit_info(race_results))
    write_csv("constructor_standings.csv", feature_tables["constructor_standings"])
    write_csv("driver_standings.csv", feature_tables["driver_standings"])
    write_csv("form_data.csv", feature_tables["form_data"])
    write_csv("reliability_data.csv", feature_tables["reliability_data"])
    write_csv("lap_times.csv", feature_tables["lap_times"])
    write_csv("pit_stops.csv", feature_tables["pit_stops"])
    preserve_existing_race_table("weather_data.csv", build_weather_data(race_results))
    preserve_existing_race_table("race_control_messages.csv", build_race_control_data(race_results))
    write_csv("telemetry_data.csv", feature_tables["telemetry_data"])


if __name__ == "__main__":
    main()
