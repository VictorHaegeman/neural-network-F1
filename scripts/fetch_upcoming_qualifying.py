from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from generate_raw_data import time_to_seconds


JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
RAW_PATH = Path("data/raw")
UPCOMING_OUTPUT_PATH = RAW_PATH / "upcoming_qualifying_results.csv"
QUALIFYING_HISTORY_PATH = RAW_PATH / "qualifying_results.csv"

QUALIFYING_COLUMNS = [
    "race_id",
    "driver_id",
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "best_qualifying_seconds",
    "reached_q2",
    "reached_q3",
    "qualifying_gap_to_pole_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch qualifying results for a specific or next F1 race.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument(
        "--merge-history",
        action="store_true",
        help="Merge fetched rows into data/raw/qualifying_results.csv.",
    )
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


def resolve_race(session: requests.Session, season: int | None, round_number: int | None) -> tuple[int, int]:
    if season is not None and round_number is not None:
        return season, round_number

    payload = jolpica_get(session, "current/next.json")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if races:
        race = races[0]
        return as_int(race.get("season")), as_int(race.get("round"))

    return date.today().year, 1


def fetch_qualifying(session: requests.Session, season: int, round_number: int) -> pd.DataFrame:
    payload = jolpica_get(session, f"{season}/{round_number}/qualifying.json?limit=100")
    races = payload.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    rows: list[dict[str, Any]] = []

    for race in races:
        current_race_id = race_id(as_int(race.get("season")), as_int(race.get("round")))
        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            q1 = time_to_seconds(result.get("Q1"))
            q2 = time_to_seconds(result.get("Q2"))
            q3 = time_to_seconds(result.get("Q3"))
            clean_times = [value for value in [q1, q2, q3] if value is not None]
            best_time = min(clean_times) if clean_times else None
            rows.append(
                {
                    "race_id": current_race_id,
                    "driver_id": driver.get("driverId"),
                    "qualifying_position": as_int(result.get("position")),
                    "q1_seconds": q1,
                    "q2_seconds": q2,
                    "q3_seconds": q3,
                    "best_qualifying_seconds": best_time,
                    "reached_q2": int(q2 is not None),
                    "reached_q3": int(q3 is not None),
                    "qualifying_gap_to_pole_seconds": None,
                }
            )

    df = pd.DataFrame(rows, columns=QUALIFYING_COLUMNS)
    if df.empty:
        return df

    for current_race_id, index in df.groupby("race_id").groups.items():
        pole_time = df.loc[index, "best_qualifying_seconds"].dropna().min()
        if pd.isna(pole_time):
            df.loc[index, "qualifying_gap_to_pole_seconds"] = 0.0
        else:
            df.loc[index, "qualifying_gap_to_pole_seconds"] = (
                df.loc[index, "best_qualifying_seconds"] - float(pole_time)
            )

    return df


def merge_history(fetched: pd.DataFrame) -> None:
    if fetched.empty:
        return

    if QUALIFYING_HISTORY_PATH.exists() and QUALIFYING_HISTORY_PATH.stat().st_size > 0:
        history = pd.read_csv(QUALIFYING_HISTORY_PATH)
    else:
        history = pd.DataFrame(columns=QUALIFYING_COLUMNS)

    combined = pd.concat([history, fetched], ignore_index=True)
    combined = combined.drop_duplicates(["race_id", "driver_id"], keep="last")
    combined = combined.sort_values(["race_id", "qualifying_position", "driver_id"])
    combined.to_csv(QUALIFYING_HISTORY_PATH, index=False)


def main() -> None:
    args = parse_args()
    session = requests.Session()
    season, round_number = resolve_race(session, args.season, args.round)
    fetched = fetch_qualifying(session, season, round_number)

    RAW_PATH.mkdir(parents=True, exist_ok=True)
    fetched.to_csv(UPCOMING_OUTPUT_PATH, index=False)
    if args.merge_history:
        merge_history(fetched)

    print(f"Fetched qualifying rows: {len(fetched)}")
    print(f"Race: {race_id(season, round_number)}")
    print(f"Wrote {UPCOMING_OUTPUT_PATH}")
    if args.merge_history:
        print(f"Merged into {QUALIFYING_HISTORY_PATH}")
    if not fetched.empty:
        print(fetched.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
