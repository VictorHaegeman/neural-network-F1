from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train_model import DATA_PATH, MODEL_PATH, build_features


PREDICTIONS_PATH = Path("outputs/predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export readable top-10 predictions for a race.")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--round", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def choose_prediction_race(df: pd.DataFrame, season: int | None, round_number: int | None) -> tuple[int, int]:
    if season is not None and round_number is not None:
        return season, round_number

    filtered = df.copy()
    if season is not None:
        filtered = filtered[filtered["season"] == season]
    if round_number is not None:
        filtered = filtered[filtered["round"] == round_number]

    if filtered.empty:
        raise ValueError("No rows match the requested season/round filters.")

    latest = filtered.sort_values(["season", "round"]).iloc[-1]
    return int(latest["season"]), int(latest["round"])


def prediction_output_path(season: int, round_number: int) -> Path:
    PREDICTIONS_PATH.mkdir(parents=True, exist_ok=True)
    return PREDICTIONS_PATH / f"top10_predictions_{season}_{round_number:02d}.csv"


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Missing dataset: {args.data}")
    if not args.model.exists():
        raise FileNotFoundError(
            f"Missing trained model: {args.model}. Run `python scripts/train_model.py` first."
        )

    df = pd.read_csv(args.data)
    season, round_number = choose_prediction_race(df, args.season, args.round)
    race_df = df[(df["season"] == season) & (df["round"] == round_number)].copy()

    if race_df.empty:
        raise ValueError(f"No race rows found for season={season}, round={round_number}.")

    model = joblib.load(args.model)
    probabilities = model.predict_proba(build_features(race_df))[:, 1]

    output = race_df[
        [
            "race_id",
            "season",
            "round",
            "grand_prix",
            "driver_id",
            "driver_code",
            "driver_name",
            "constructor_name",
            "grid",
            "final_position",
            "top10_finish",
        ]
    ].copy()
    output["top10_probability"] = probabilities
    output = output.sort_values("top10_probability", ascending=False).reset_index(drop=True)
    output["predicted_rank"] = output.index + 1
    output["predicted_top10"] = (output["predicted_rank"] <= args.top_n).astype(int)

    path = args.output or prediction_output_path(season, round_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)

    print(f"Predictions written to {path}")
    print(output.head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
