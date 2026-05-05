from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from generate_raw_data import (
    RAW_PATH,
    build_circuit_info,
    build_driver_info,
    build_feature_tables,
    build_race_control_data,
    build_team_info,
    build_weather_data,
    fetch_single_race_records,
    parse_pit_stop_races,
    parse_qualifying_races,
    parse_result_races,
    race_id,
    summarize_current_pit_stops,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COVERAGE_SUMMARY = PROJECT_ROOT / "outputs/data_coverage_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incrementally import only completed races missing from local raw CSVs. "
            "Existing race_id rows are skipped unless --force is used."
        )
    )
    parser.add_argument("--season", type=int, default=None, help="Season to import.")
    parser.add_argument("--round", type=int, default=None, help="Specific round to import.")
    parser.add_argument(
        "--coverage-summary",
        type=Path,
        default=DEFAULT_COVERAGE_SUMMARY,
        help="Coverage summary produced by scripts/audit_data_coverage.py.",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Delay between API calls.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing rows for the selected race_id. Use sparingly.",
    )
    parser.add_argument(
        "--skip-pit-stops",
        action="store_true",
        help="Skip pit-stop event import for newly imported race results.",
    )
    return parser.parse_args()


def load_csv(name: str) -> pd.DataFrame:
    path = RAW_PATH / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_race_ref(value: str) -> tuple[int, int]:
    season, round_number = value.split("_", maxsplit=1)
    return int(season), int(round_number)


def candidate_races(args: argparse.Namespace) -> list[tuple[int, int]]:
    if args.season is not None and args.round is not None:
        return [(args.season, args.round)]

    if not args.coverage_summary.exists():
        raise FileNotFoundError(
            f"Coverage summary not found: {args.coverage_summary}. "
            "Run scripts/audit_data_coverage.py first or pass --season and --round."
        )

    summary = json.loads(args.coverage_summary.read_text(encoding="utf-8"))
    race_ids: list[str] = []
    race_ids.extend(summary.get("missing_local_results_available_in_api", []))
    race_ids.extend(summary.get("api_results_without_final_rows", []))

    candidates = sorted({parse_race_ref(value) for value in race_ids})
    if args.season is not None:
        candidates = [race_ref for race_ref in candidates if race_ref[0] == args.season]
    return candidates


def has_race(df: pd.DataFrame, current_race_id: str) -> bool:
    return not df.empty and "race_id" in df.columns and current_race_id in set(df["race_id"].astype(str))


def replace_race_rows(df: pd.DataFrame, new_rows: pd.DataFrame, current_race_id: str) -> pd.DataFrame:
    if new_rows.empty:
        return df
    if df.empty:
        return new_rows
    kept = df[df["race_id"].astype(str) != current_race_id].copy()
    combined = pd.concat([kept, new_rows], ignore_index=True)
    sort_columns = [column for column in ["season", "round", "final_position", "qualifying_position", "driver_id"] if column in combined.columns]
    return combined.sort_values(sort_columns).reset_index(drop=True) if sort_columns else combined


def fetch_optional_race(
    session: requests.Session,
    season: int,
    round_number: int,
    table: str,
    result_key: str,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    try:
        return fetch_single_race_records(
            session=session,
            season=season,
            round_number=round_number,
            table=table,
            result_key=result_key,
            sleep_seconds=sleep_seconds,
        )
    except RuntimeError as exc:
        print(f"Skipped {season}_{round_number:02d} {table}: {exc}")
        return None


def rebuild_derived_tables(race_results: pd.DataFrame, pit_stop_events: pd.DataFrame) -> None:
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


def preserve_existing_race_table(name: str, fallback: pd.DataFrame) -> None:
    existing = load_csv(name)
    if existing.empty or "race_id" not in existing.columns:
        write_csv(name, fallback)
        return

    existing_races = set(existing["race_id"].astype(str))
    missing = fallback[~fallback["race_id"].astype(str).isin(existing_races)].copy()
    combined = pd.concat([existing, missing], ignore_index=True)
    combined = combined.sort_values("race_id").reset_index(drop=True)
    write_csv(name, combined)


def main() -> None:
    args = parse_args()
    candidates = candidate_races(args)
    if not candidates:
        print("No missing completed races to import.")
        return

    race_results = load_csv("race_results.csv")
    qualifying_results = load_csv("qualifying_results.csv")
    pit_stop_events = load_csv("pit_stop_events.csv")
    changed = False

    with requests.Session() as session:
        for season, round_number in candidates:
            current_race_id = race_id(season, round_number)
            print(f"Checking {current_race_id}...")

            if not has_race(race_results, current_race_id) or args.force:
                result_race = fetch_optional_race(
                    session,
                    season,
                    round_number,
                    "results",
                    "Results",
                    args.sleep,
                )
                result_rows = pd.DataFrame(parse_result_races([result_race])) if result_race else pd.DataFrame()
                if result_rows.empty:
                    print(f"No race-result rows imported for {current_race_id}.")
                else:
                    race_results = replace_race_rows(race_results, result_rows, current_race_id)
                    print(f"Imported race results for {current_race_id}: {len(result_rows)} rows.")
                    changed = True
            else:
                print(f"Race results already local for {current_race_id}; skipped.")

            if not has_race(qualifying_results, current_race_id) or args.force:
                qualifying_race = fetch_optional_race(
                    session,
                    season,
                    round_number,
                    "qualifying",
                    "QualifyingResults",
                    args.sleep,
                )
                qualifying_rows = pd.DataFrame(parse_qualifying_races([qualifying_race])) if qualifying_race else pd.DataFrame()
                if qualifying_rows.empty:
                    print(f"No qualifying rows imported for {current_race_id}.")
                else:
                    qualifying_results = replace_race_rows(qualifying_results, qualifying_rows, current_race_id)
                    print(f"Imported qualifying for {current_race_id}: {len(qualifying_rows)} rows.")
                    changed = True
            else:
                print(f"Qualifying already local for {current_race_id}; skipped.")

            if args.skip_pit_stops:
                print(f"Pit stops skipped for {current_race_id}.")
            elif not has_race(pit_stop_events, current_race_id) or args.force:
                pit_stop_race = fetch_optional_race(
                    session,
                    season,
                    round_number,
                    "pitstops",
                    "PitStops",
                    args.sleep,
                )
                pit_stop_rows = parse_pit_stop_races([pit_stop_race]) if pit_stop_race else pd.DataFrame()
                if pit_stop_rows.empty:
                    print(f"No pit-stop rows imported for {current_race_id}.")
                else:
                    pit_stop_events = replace_race_rows(pit_stop_events, pit_stop_rows, current_race_id)
                    print(f"Imported pit stops for {current_race_id}: {len(pit_stop_rows)} rows.")
                    changed = True
            else:
                print(f"Pit stops already local for {current_race_id}; skipped.")

    if not changed:
        print("No local raw files changed.")
        return

    write_csv("race_results.csv", race_results)
    write_csv("qualifying_results.csv", qualifying_results)
    write_csv("pit_stop_events.csv", pit_stop_events)
    rebuild_derived_tables(race_results, pit_stop_events)


if __name__ == "__main__":
    main()
