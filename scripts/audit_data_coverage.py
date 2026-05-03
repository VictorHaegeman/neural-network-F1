from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests


JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_PATH = Path("data/raw")
FINAL_PATH = Path("data/final/f1_top10_model_dataset.csv")
OUTPUT_PATH = Path("outputs")
REPORT_PATH = OUTPUT_PATH / "data_coverage_report.csv"
SUMMARY_PATH = OUTPUT_PATH / "data_coverage_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local F1 data coverage against Jolpica API.")
    parser.add_argument("--season", type=int, default=date.today().year)
    return parser.parse_args()


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def race_id(season: int, round_number: int) -> str:
    return f"{season}_{round_number:02d}"


def jolpica_get(session: requests.Session, endpoint: str) -> dict[str, Any]:
    response = session.get(f"{JOLPICA_BASE_URL}/{endpoint.lstrip('/')}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_race_table(session: requests.Session, season: int, table: str) -> list[dict[str, Any]]:
    payload = jolpica_get(session, f"{season}/{table}.json?limit=1000")
    return payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])


def load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def race_ids_from_races(races: list[dict[str, Any]]) -> set[str]:
    return {
        race_id(as_int(race.get("season")), as_int(race.get("round")))
        for race in races
        if as_int(race.get("season")) and as_int(race.get("round"))
    }


def build_report(season: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    session = requests.Session()
    schedule = fetch_race_table(session, season, "races")
    result_races = fetch_race_table(session, season, "results")
    qualifying_races = fetch_race_table(session, season, "qualifying")

    api_result_ids = race_ids_from_races(result_races)
    api_qualifying_ids = race_ids_from_races(qualifying_races)

    local_results = load_optional_csv(RAW_PATH / "race_results.csv")
    local_qualifying = load_optional_csv(RAW_PATH / "qualifying_results.csv")
    local_fastf1_weather = load_optional_csv(RAW_PATH / "fastf1_weather.csv")
    final_df = load_optional_csv(FINAL_PATH)

    local_result_ids = set(local_results.get("race_id", pd.Series(dtype=str)).astype(str))
    local_qualifying_ids = set(local_qualifying.get("race_id", pd.Series(dtype=str)).astype(str))
    local_fastf1_weather_ids = set(local_fastf1_weather.get("race_id", pd.Series(dtype=str)).astype(str))
    final_ids = set(final_df.get("race_id", pd.Series(dtype=str)).astype(str))

    rows = []
    for race in schedule:
        season_number = as_int(race.get("season"))
        round_number = as_int(race.get("round"))
        current_race_id = race_id(season_number, round_number)
        circuit = race.get("Circuit", {})
        rows.append(
            {
                "race_id": current_race_id,
                "season": season_number,
                "round": round_number,
                "grand_prix": race.get("raceName"),
                "race_date": race.get("date"),
                "circuit_id": circuit.get("circuitId"),
                "api_has_results": current_race_id in api_result_ids,
                "local_has_results": current_race_id in local_result_ids,
                "api_has_qualifying": current_race_id in api_qualifying_ids,
                "local_has_qualifying": current_race_id in local_qualifying_ids,
                "local_has_fastf1_weather": current_race_id in local_fastf1_weather_ids,
                "local_has_final_dataset_rows": current_race_id in final_ids,
            }
        )

    report = pd.DataFrame(rows)
    summary = {
        "season": season,
        "scheduled_races": int(len(report)),
        "api_result_races": int(report["api_has_results"].sum()) if not report.empty else 0,
        "local_result_races": int(report["local_has_results"].sum()) if not report.empty else 0,
        "api_qualifying_races": int(report["api_has_qualifying"].sum()) if not report.empty else 0,
        "local_qualifying_races": int(report["local_has_qualifying"].sum()) if not report.empty else 0,
        "local_fastf1_weather_races": int(report["local_has_fastf1_weather"].sum()) if not report.empty else 0,
        "final_dataset_races": int(report["local_has_final_dataset_rows"].sum()) if not report.empty else 0,
        "missing_local_results_available_in_api": report[
            report["api_has_results"] & ~report["local_has_results"]
        ]["race_id"].tolist(),
        "missing_local_qualifying_available_in_api": report[
            report["api_has_qualifying"] & ~report["local_has_qualifying"]
        ]["race_id"].tolist(),
        "api_results_without_final_rows": report[
            report["api_has_results"] & ~report["local_has_final_dataset_rows"]
        ]["race_id"].tolist(),
    }
    return report, summary


def main() -> None:
    args = parse_args()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    report, summary = build_report(args.season)
    report.to_csv(REPORT_PATH, index=False)
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Data coverage report written to {REPORT_PATH}")
    print(f"Data coverage summary written to {SUMMARY_PATH}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
