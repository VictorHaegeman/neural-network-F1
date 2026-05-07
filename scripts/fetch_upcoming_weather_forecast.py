from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from predict_upcoming_races import (
    DATA_PATH,
    fetch_schedule,
    fetch_weather_forecast,
    historical_weather,
    choose_upcoming_races,
    race_to_record,
)


OUTPUT_PATH = Path("data/raw/upcoming_weather_forecast.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and store pre-race weather forecasts for upcoming races.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--current-date", type=str, default=date.today().isoformat())
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")

    df = pd.read_csv(args.data)
    current_date = datetime.strptime(args.current_date, "%Y-%m-%d").date()
    with requests.Session() as session:
        schedule = fetch_schedule(session, args.season)
    upcoming = choose_upcoming_races(schedule, df, args.season, current_date, args.count)

    fetched_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows: list[dict[str, object]] = []
    for race in upcoming:
        fallback, fallback_source = historical_weather(df, race["circuit_id"])
        try:
            forecast, source = fetch_weather_forecast(race)
        except requests.RequestException as exc:
            forecast, source = None, f"forecast_request_failed:{type(exc).__name__}"

        weather = forecast or fallback
        rows.append(
            {
                "race_id": race["race_id"],
                "season": race["season"],
                "round": race["round"],
                "grand_prix": race["grand_prix"],
                "race_date": race["race_date"],
                "circuit_id": race["circuit_id"],
                "latitude": race["latitude"],
                "longitude": race["longitude"],
                "forecast_source": source if forecast else fallback_source,
                "forecast_available": int(forecast is not None),
                "fetched_at_utc": fetched_at,
                **weather,
            }
        )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {output.shape[0]} rows, {output.shape[1]} columns")
    if not output.empty:
        print(output[["race_id", "grand_prix", "race_date", "forecast_source", "forecast_available"]].to_string(index=False))


if __name__ == "__main__":
    main()
