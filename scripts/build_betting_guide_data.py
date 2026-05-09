from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_PREDICTIONS_PATH = Path("outputs/predictions/neural_network/upcoming_top10_predictions.csv")
DEFAULT_OUTPUT_PATH = Path("webapp/data/betting_guide_data.json")
DEFAULT_HISTORY_PATH = Path("data/final/f1_top10_model_dataset.csv")
NN_SUMMARY_PATH = Path("outputs/neural_network_summary.json")
MODEL_COMPARISON_PATH = Path("outputs/model_comparison.csv")
POSITION_METRICS_PATH = Path("outputs/position_model_metrics.json")
HEADSHOT_DIR = Path("outputs/driver_headshots")
CIRCUIT_ASSETS_BY_NAME = {
    "Circuit Gilles Villeneuve": {
        "path": "/webapp/assets/circuits/circuit_gilles_villeneuve.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Circuit_Gilles_Villeneuve.svg",
        "author": "Will Pittenger, derivative work by cBuckley",
        "license": "CC BY-SA 3.0",
    },
    "Circuit de Monaco": {
        "path": "/webapp/assets/circuits/circuit_de_monaco.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Circuit_Monaco.svg",
        "author": "Rumbin",
        "license": "Public domain",
    },
    "Circuit de Barcelona-Catalunya": {
        "path": "/webapp/assets/circuits/circuit_de_barcelona_catalunya.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:2023_F1_CourseLayout_Spain.svg",
        "author": "ごひょううべこ",
        "license": "CC BY-SA 4.0",
    },
    "Red Bull Ring": {
        "path": "/webapp/assets/circuits/red_bull_ring.svg",
        "source_url": "https://commons.wikimedia.org/wiki/File:Circuit_Red_Bull_Ring.svg",
        "author": "Pitlane02 and Wikimedia contributors",
        "license": "CC BY-SA 3.0",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JSON data for the neural-network betting guide webapp.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--history-data", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else 0.0


def numeric_value(row: pd.Series, column: str, default: float = 0.0) -> float:
    if column not in row:
        return default
    return safe_float(row.get(column), default)


def first_text(row: pd.Series, column: str, default: str = "Unknown") -> str:
    if column not in row:
        return default
    value = row.get(column)
    if value is None or pd.isna(value) or str(value).strip() == "":
        return default
    return str(value).strip()


def fair_decimal_odds(probability: float) -> float | None:
    if probability <= 0:
        return None
    return round(1 / probability, 2)


def confidence_label(probability: float, rank: int) -> str:
    if probability >= 0.78 and rank <= 5:
        return "Core"
    if probability >= 0.6 and rank <= 10:
        return "Strong"
    if probability >= 0.45 and rank <= 12:
        return "Bubble"
    if probability >= 0.25:
        return "Speculative"
    return "Avoid"


def pick_recommendation(probability: float, rank: int, predicted_top10: bool) -> str:
    if predicted_top10 and probability >= 0.72:
        return "Priority top-10"
    if predicted_top10 and probability >= 0.55:
        return "Playable top-10"
    if rank <= 12 and probability >= 0.4:
        return "Watch the odds"
    return "Pass unless price is high"


def headshot_path(driver_code: str) -> str | None:
    candidate = HEADSHOT_DIR / f"{driver_code}.png"
    if candidate.exists():
        return f"/outputs/driver_headshots/{driver_code}.png"
    return None


def completed_history(history_df: pd.DataFrame | None, circuit_name: str, race_date: str) -> pd.DataFrame:
    if history_df is None or history_df.empty or "circuit_name" not in history_df.columns:
        return pd.DataFrame()

    final_position = pd.to_numeric(history_df.get("final_position"), errors="coerce")
    completed = history_df[final_position > 0].copy()
    if completed.empty:
        return completed

    same_circuit = completed["circuit_name"].astype(str).str.casefold() == circuit_name.casefold()
    completed = completed[same_circuit].copy()
    if completed.empty or "race_date" not in completed.columns:
        return completed

    current_date = pd.to_datetime(race_date, errors="coerce")
    completed_dates = pd.to_datetime(completed["race_date"], errors="coerce")
    if not pd.isna(current_date):
        completed = completed[completed_dates < current_date].copy()
    return completed


def unique_race_rows(circuit_rows: pd.DataFrame) -> pd.DataFrame:
    if circuit_rows.empty:
        return circuit_rows
    return (
        circuit_rows.sort_values(["season", "round", "race_id"])
        .drop_duplicates("race_id", keep="first")
        .reset_index(drop=True)
    )


def race_rate(race_rows: pd.DataFrame, column: str) -> float:
    if race_rows.empty or column not in race_rows.columns:
        return 0.0
    values = pd.to_numeric(race_rows[column], errors="coerce").fillna(0)
    return float((values > 0).mean())


def race_mean(race_rows: pd.DataFrame, column: str) -> float:
    if race_rows.empty or column not in race_rows.columns:
        return 0.0
    return float(pd.to_numeric(race_rows[column], errors="coerce").fillna(0).mean())


def combined_race_series(race_rows: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in race_rows.columns]
    if not existing:
        return pd.Series([0.0] * len(race_rows), index=race_rows.index)
    values = race_rows[existing].apply(pd.to_numeric, errors="coerce").fillna(0)
    return values.max(axis=1)


def chaos_label(safety_car_rate: float, disruption_score: float) -> str:
    if safety_car_rate >= 0.65 or disruption_score >= 38:
        return "Chaotique"
    if safety_car_rate >= 0.35 or disruption_score >= 22:
        return "Variable"
    return "Controle"


def build_circuit_profile(circuit_rows: pd.DataFrame) -> dict[str, Any]:
    race_rows = unique_race_rows(circuit_rows)
    if race_rows.empty:
        return {
            "races_analyzed": 0,
            "safety_car_race_rate_pct": 0.0,
            "avg_safety_car_count": 0.0,
            "vsc_race_rate_pct": 0.0,
            "red_flag_race_rate_pct": 0.0,
            "wet_race_rate_pct": 0.0,
            "avg_rainfall_pct": 0.0,
            "avg_air_temp": 0.0,
            "avg_disruption_score": 0.0,
            "avg_dnf_count": 0.0,
            "avg_penalties": 0.0,
            "avg_incidents": 0.0,
            "chaos_label": "Unknown",
            "track_type": "Unknown",
            "overtaking_difficulty": "Unknown",
            "street_circuit": False,
            "circuit_length_km": 0.0,
            "total_laps": 0,
            "recent_races": [],
        }

    safety_car_values = combined_race_series(race_rows, ["safety_car_count", "fastf1_safety_car_count"])
    vsc_values = combined_race_series(
        race_rows,
        ["virtual_safety_car_count", "fastf1_virtual_safety_car_count"],
    )
    red_flag_values = combined_race_series(race_rows, ["red_flag_count", "fastf1_red_flag_count"])
    disruption_values = combined_race_series(
        race_rows,
        ["race_disruption_score", "fastf1_race_disruption_score"],
    )
    safety_car_rate = float((safety_car_values > 0).mean())
    disruption_score = float(disruption_values.mean())

    winners: dict[str, dict[str, str]] = {}
    if "final_position" in circuit_rows.columns:
        winner_rows = circuit_rows[pd.to_numeric(circuit_rows["final_position"], errors="coerce") == 1]
        for _, row in winner_rows.iterrows():
            winners[str(row.get("race_id"))] = {
                "driver_code": first_text(row, "driver_code", ""),
                "driver_name": first_text(row, "driver_name", ""),
            }

    recent_races: list[dict[str, Any]] = []
    for _, row in race_rows.sort_values(["season", "round"], ascending=False).head(5).iterrows():
        race_id = str(row.get("race_id"))
        recent_races.append(
            {
                "season": safe_int(row.get("season")),
                "grand_prix": first_text(row, "grand_prix", ""),
                "winner_code": winners.get(race_id, {}).get("driver_code", ""),
                "winner_name": winners.get(race_id, {}).get("driver_name", ""),
                "weather_condition": first_text(row, "weather_condition", "Unknown"),
                "safety_car_count": safe_int(
                    max(safe_float(row.get("safety_car_count")), safe_float(row.get("fastf1_safety_car_count")))
                ),
                "red_flag_count": safe_int(
                    max(safe_float(row.get("red_flag_count")), safe_float(row.get("fastf1_red_flag_count")))
                ),
                "race_disruption_score": round(
                    max(
                        safe_float(row.get("race_disruption_score")),
                        safe_float(row.get("fastf1_race_disruption_score")),
                    ),
                    1,
                ),
                "wet_race": bool(safe_int(row.get("wet_race"))),
            }
        )

    first = race_rows.iloc[-1]
    return {
        "races_analyzed": int(len(race_rows)),
        "safety_car_race_rate_pct": round(safety_car_rate * 100, 1),
        "avg_safety_car_count": round(float(safety_car_values.mean()), 2),
        "vsc_race_rate_pct": round(float((vsc_values > 0).mean()) * 100, 1),
        "red_flag_race_rate_pct": round(float((red_flag_values > 0).mean()) * 100, 1),
        "wet_race_rate_pct": round(race_rate(race_rows, "wet_race") * 100, 1),
        "avg_rainfall_pct": round(race_mean(race_rows, "rainfall_percentage"), 1),
        "avg_air_temp": round(race_mean(race_rows, "air_temp_mean"), 1),
        "avg_disruption_score": round(disruption_score, 1),
        "avg_dnf_count": round(race_mean(race_rows, "total_dnf_count"), 1),
        "avg_penalties": round(race_mean(race_rows, "penalty_count"), 1),
        "avg_incidents": round(race_mean(race_rows, "incident_count"), 1),
        "chaos_label": chaos_label(safety_car_rate, disruption_score),
        "track_type": first_text(first, "track_type", "Unknown"),
        "overtaking_difficulty": first_text(first, "overtaking_difficulty", "Unknown"),
        "street_circuit": bool(safe_int(first.get("street_circuit"))),
        "circuit_length_km": round(numeric_value(first, "circuit_length_km"), 3),
        "total_laps": safe_int(first.get("total_laps")),
        "recent_races": recent_races,
    }


def dnf_like(status: str) -> bool:
    lower = status.casefold()
    classified_terms = ["finished", "lap", "+", "withdrawn"]
    if any(term in lower for term in classified_terms):
        return False
    return bool(lower and lower != "unknown")


def driver_history_label(
    starts: int,
    top10_rate: float,
    avg_finish: float,
    avg_gain: float,
    chaos_dependency: float,
) -> str:
    if starts == 0:
        return "Pas d'historique direct ici."
    if starts == 1:
        return "Signal utile, mais echantillon tres court."
    if top10_rate >= 0.7 and avg_finish <= 8 and chaos_dependency < 55:
        return "Performance repetee, pas seulement un scenario chanceux."
    if top10_rate >= 0.5 and chaos_dependency >= 55:
        return "Bons resultats, mais souvent dans des courses agitees."
    if avg_gain >= 3:
        return "Bon profil de remontada sur ce circuit."
    if top10_rate <= 0.25:
        return "Historique faible sur ce circuit."
    return "Historique correct, a confirmer avec la qualif et la meteo."


def build_driver_circuit_history(
    driver: dict[str, Any],
    circuit_rows: pd.DataFrame,
    circuit_profile: dict[str, Any],
) -> dict[str, Any]:
    if circuit_rows.empty or "driver_code" not in circuit_rows.columns:
        return {
            "starts": 0,
            "top10_rate_pct": 0.0,
            "avg_finish": None,
            "best_finish": None,
            "avg_grid": None,
            "avg_position_gain": 0.0,
            "podiums": 0,
            "dnf_rate_pct": 0.0,
            "chaos_dependency_pct": 0.0,
            "signal": "Pas d'historique direct ici.",
            "recent_results": [],
        }

    rows = circuit_rows[circuit_rows["driver_code"].astype(str) == driver["driver_code"]].copy()
    if rows.empty:
        return {
            "starts": 0,
            "top10_rate_pct": 0.0,
            "avg_finish": None,
            "best_finish": None,
            "avg_grid": None,
            "avg_position_gain": 0.0,
            "podiums": 0,
            "dnf_rate_pct": 0.0,
            "chaos_dependency_pct": 0.0,
            "signal": "Pas d'historique direct ici.",
            "recent_results": [],
        }

    final_position = pd.to_numeric(rows["final_position"], errors="coerce")
    grid = pd.to_numeric(rows.get("grid"), errors="coerce")
    top10 = pd.to_numeric(rows.get("top10_finish"), errors="coerce").fillna(0)
    starts = int(len(rows))
    top10_rate = float(top10.mean()) if starts else 0.0
    avg_finish = float(final_position.mean()) if not final_position.empty else 0.0
    best_finish = int(final_position.min()) if not final_position.empty else 0
    avg_grid = float(grid.mean()) if not grid.empty else 0.0
    avg_gain = float((grid - final_position).mean()) if not grid.empty else 0.0
    podiums = int((final_position <= 3).sum())

    status = rows.get("status")
    dnf_rate = float(status.astype(str).map(dnf_like).mean()) if status is not None else 0.0

    good_rows = rows[top10 == 1]
    if good_rows.empty:
        chaos_source = pd.to_numeric(rows.get("race_disruption_score"), errors="coerce").fillna(0)
    else:
        chaos_source = pd.to_numeric(good_rows.get("race_disruption_score"), errors="coerce").fillna(0)
    chaos_when_good = float(chaos_source.mean()) if not chaos_source.empty else 0.0
    circuit_disruption = max(1.0, safe_float(circuit_profile.get("avg_disruption_score"), 1.0))
    chaos_dependency = min(100.0, max(0.0, (chaos_when_good / (circuit_disruption * 1.4)) * 70))
    if top10_rate >= 0.7:
        chaos_dependency = max(0.0, chaos_dependency - 12)
    if avg_gain > 2:
        chaos_dependency = max(0.0, chaos_dependency - 8)

    recent_results = []
    for _, row in rows.sort_values(["season", "round"], ascending=False).head(4).iterrows():
        recent_results.append(
            {
                "season": safe_int(row.get("season")),
                "finish": safe_int(row.get("final_position")),
                "grid": safe_int(row.get("grid")),
                "points": round(safe_float(row.get("points")), 1),
                "weather_condition": first_text(row, "weather_condition", "Unknown"),
                "safety_car_count": safe_int(row.get("safety_car_count")),
                "race_disruption_score": round(safe_float(row.get("race_disruption_score")), 1),
            }
        )

    return {
        "starts": starts,
        "top10_rate_pct": round(top10_rate * 100, 1),
        "avg_finish": round(avg_finish, 1),
        "best_finish": best_finish,
        "avg_grid": round(avg_grid, 1),
        "avg_position_gain": round(avg_gain, 1),
        "podiums": podiums,
        "dnf_rate_pct": round(dnf_rate * 100, 1),
        "chaos_dependency_pct": round(chaos_dependency, 1),
        "signal": driver_history_label(starts, top10_rate, avg_finish, avg_gain, chaos_dependency),
        "recent_results": recent_results,
    }


def compound_bias(rows: list[dict[str, Any]]) -> str:
    rates = {
        "Soft": safe_mean([safe_float(row.get("fastf1_soft_lap_rate_previous_3"), math.nan) for row in rows]),
        "Medium": safe_mean([safe_float(row.get("fastf1_medium_lap_rate_previous_3"), math.nan) for row in rows]),
        "Hard": safe_mean([safe_float(row.get("fastf1_hard_lap_rate_previous_3"), math.nan) for row in rows]),
        "Inter": safe_mean([safe_float(row.get("fastf1_intermediate_lap_rate_previous_3"), math.nan) for row in rows]),
    }
    return max(rates, key=rates.get)


def build_strategy_context(
    race: dict[str, Any],
    group: list[dict[str, Any]],
    circuit_profile: dict[str, Any],
) -> dict[str, Any]:
    stint_values = [safe_float(row.get("fastf1_stint_count_previous_3"), math.nan) for row in group]
    tyre_life_values = [safe_float(row.get("fastf1_avg_tyre_life_previous_3"), math.nan) for row in group]
    avg_stints = safe_mean(stint_values)
    avg_tyre_life = safe_mean(tyre_life_values)
    expected_stops = max(1, round(avg_stints - 1)) if avg_stints else 1
    if avg_stints >= 2.45:
        stop_label = "1-2 arrets"
    elif avg_stints >= 1.8:
        stop_label = f"{expected_stops} arret"
    else:
        stop_label = "1 arret prudent"

    total_laps = safe_int(circuit_profile.get("total_laps"))
    if avg_tyre_life and total_laps:
        low = max(6, int(avg_tyre_life * 0.75))
        high = min(total_laps - 1, int(avg_tyre_life * 1.35))
        pit_window = f"tour {low}-{high}"
    else:
        pit_window = "fenetre a ajuster apres qualif"

    wet_probability = safe_float(race.get("wet_race_probability"))
    if wet_probability >= 45:
        base_plan = "Plan flexible: pluie/intermediaires possibles."
    elif str(circuit_profile.get("overtaking_difficulty", "")).casefold() in {"hard", "very hard"}:
        base_plan = "Priorite position piste: undercut important."
    elif safe_float(circuit_profile.get("safety_car_race_rate_pct")) >= 50:
        base_plan = "Garder une fenetre ouverte pour safety car."
    else:
        base_plan = "Plan sec standard, optimiser degradation."

    teams: list[dict[str, Any]] = []
    for constructor in sorted({str(row.get("constructor_name", "")) for row in group}):
        rows = [row for row in group if str(row.get("constructor_name", "")) == constructor]
        teams.append(
            {
                "constructor_name": constructor,
                "avg_top10_probability_pct": round(
                    safe_mean([safe_float(row.get("top10_probability"), math.nan) for row in rows]) * 100,
                    1,
                ),
                "avg_stints_previous_3": round(
                    safe_mean([safe_float(row.get("fastf1_stint_count_previous_3"), math.nan) for row in rows]),
                    2,
                ),
                "avg_tyre_life_previous_3": round(
                    safe_mean([safe_float(row.get("fastf1_avg_tyre_life_previous_3"), math.nan) for row in rows]),
                    1,
                ),
                "compound_bias": compound_bias(rows),
            }
        )

    teams = sorted(teams, key=lambda item: item["avg_top10_probability_pct"], reverse=True)
    return {
        "expected_stops_label": stop_label,
        "avg_stints_previous_3": round(avg_stints, 2),
        "avg_tyre_life_previous_3": round(avg_tyre_life, 1),
        "pit_window": pit_window,
        "compound_bias": compound_bias(group),
        "base_plan": base_plan,
        "safety_car_sensitivity": "Haute" if safe_float(circuit_profile.get("safety_car_race_rate_pct")) >= 50 else "Normale",
        "team_tendencies": teams[:8],
    }


def build_betting_angles(
    drivers: list[dict[str, Any]],
    race: dict[str, Any],
    circuit_profile: dict[str, Any],
) -> dict[str, Any]:
    safe = [
        driver
        for driver in drivers
        if driver["predicted_top10"] and driver["top10_probability"] >= 0.72
    ][:4]
    value_watch = [
        driver
        for driver in drivers
        if 8 <= driver["predicted_rank"] <= 12
        and driver["top10_probability"] >= 0.4
        and (
            driver.get("circuit_history", {}).get("starts", 0) == 0
            or driver.get("circuit_history", {}).get("top10_rate_pct", 0) >= 40
            or driver.get("circuit_history", {}).get("avg_position_gain", 0) >= 2
        )
    ][:4]

    warnings: list[str] = []
    if safe_float(race.get("bubble_gap_pct")) < 5:
        warnings.append("Bubble 10/11 serre: eviter les grosses mises sur les rangs 9-12.")
    if safe_float(race.get("wet_race_probability")) >= 45:
        warnings.append("Meteo humide: probabilites plus fragiles, favoriser mises petites.")
    if circuit_profile.get("chaos_label") == "Chaotique":
        warnings.append("Historique safety car eleve: valeur possible, variance elevee.")
    if str(race.get("prediction_qualifying_source")) == "estimated_strength_order":
        warnings.append("Qualif estimee: refaire le ticket quand la vraie grille est disponible.")

    return {
        "safe": [
            {
                "driver_code": driver["driver_code"],
                "driver_name": driver["driver_name"],
                "probability_pct": driver["top10_probability_pct"],
                "fair_decimal_odds": driver["fair_decimal_odds"],
            }
            for driver in safe
        ],
        "value_watch": [
            {
                "driver_code": driver["driver_code"],
                "driver_name": driver["driver_name"],
                "probability_pct": driver["top10_probability_pct"],
                "fair_decimal_odds": driver["fair_decimal_odds"],
                "history_signal": driver.get("circuit_history", {}).get("signal", ""),
            }
            for driver in value_watch
        ],
        "warnings": warnings,
    }


def model_comparison_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    "model": str(row.get("model", "")),
                    "race_precision_at_10": round(safe_float(row.get("race_precision_at_10")), 3),
                    "f1": round(safe_float(row.get("f1")), 3),
                    "roc_auc": round(safe_float(row.get("roc_auc")), 3),
                }
            )
    return rows


def build_driver(row: dict[str, Any]) -> dict[str, Any]:
    probability = safe_float(row.get("top10_probability"))
    rank = safe_int(row.get("predicted_rank") or row.get("top10_probability_rank"))
    predicted_top10 = bool(safe_int(row.get("predicted_top10")))
    driver_code = str(row.get("driver_code", "")).strip()

    return {
        "driver_code": driver_code,
        "driver_name": str(row.get("driver_name", "")).strip(),
        "constructor_name": str(row.get("constructor_name", "")).strip(),
        "headshot_path": headshot_path(driver_code),
        "grid": safe_int(row.get("grid")),
        "qualifying_position": safe_int(row.get("qualifying_position")),
        "top10_probability": round(probability, 4),
        "top10_probability_pct": round(probability * 100, 1),
        "top10_probability_rank": safe_int(row.get("top10_probability_rank")),
        "predicted_rank": rank,
        "predicted_top10": predicted_top10,
        "predicted_finish_rank": safe_int(row.get("predicted_finish_rank")),
        "predicted_finish_position_raw": round(safe_float(row.get("predicted_finish_position_raw")), 2),
        "fair_decimal_odds": fair_decimal_odds(probability),
        "confidence_label": confidence_label(probability, rank),
        "recommendation": pick_recommendation(probability, rank, predicted_top10),
        "driver_points_before_race": round(safe_float(row.get("driver_points_before_race")), 1),
        "constructor_points_before_race": round(safe_float(row.get("constructor_points_before_race")), 1),
        "top10_rate_previous_5": round(safe_float(row.get("top10_rate_previous_5")), 2),
        "avg_finish_position_previous_5": round(safe_float(row.get("avg_finish_position_previous_5")), 1),
    }


def build_race(
    race_id: str,
    group: list[dict[str, Any]],
    history_df: pd.DataFrame | None,
) -> dict[str, Any]:
    ordered = sorted(
        group,
        key=lambda row: (
            safe_int(row.get("predicted_rank") or row.get("top10_probability_rank")),
            safe_int(row.get("top10_probability_rank")),
            str(row.get("driver_code", "")),
        ),
    )
    drivers = [build_driver(row) for row in ordered]
    top10 = [driver for driver in drivers if driver["predicted_top10"]]
    probabilities = [driver["top10_probability"] for driver in drivers]
    rank10 = probabilities[9] if len(probabilities) > 9 else 0.0
    rank11 = probabilities[10] if len(probabilities) > 10 else 0.0
    first = ordered[0]

    average_top10_probability = sum(driver["top10_probability"] for driver in top10) / max(1, len(top10))
    weather_condition = str(first.get("weather_condition", "Unknown")).strip()
    rainfall = safe_float(first.get("rainfall_percentage"))
    disruption = safe_float(first.get("circuit_race_disruption_score_avg_previous_3"))

    race = {
        "race_id": race_id,
        "season": safe_int(first.get("season")),
        "round": safe_int(first.get("round")),
        "grand_prix": str(first.get("grand_prix", "")).strip(),
        "race_date": str(first.get("race_date", "")).strip(),
        "circuit_name": str(first.get("circuit_name", "")).strip(),
        "weather_condition": weather_condition,
        "air_temp_mean": round(safe_float(first.get("air_temp_mean")), 1),
        "rainfall_percentage": round(rainfall, 1),
        "wet_race_probability": round(safe_float(first.get("wet_race")) * 100, 1),
        "race_disruption_score": round(disruption, 1),
        "prediction_weather_source": str(first.get("prediction_weather_source", "")).strip(),
        "prediction_qualifying_source": str(first.get("prediction_qualifying_source", "")).strip(),
        "average_top10_probability": round(average_top10_probability, 4),
        "average_top10_probability_pct": round(average_top10_probability * 100, 1),
        "bubble_gap": round(rank10 - rank11, 4),
        "bubble_gap_pct": round((rank10 - rank11) * 100, 1),
        "drivers": drivers,
    }
    circuit_asset = CIRCUIT_ASSETS_BY_NAME.get(race["circuit_name"], {})
    race["circuit_image_path"] = circuit_asset.get("path")
    race["circuit_image_attribution"] = {
        key: value for key, value in circuit_asset.items() if key != "path"
    }

    circuit_rows = completed_history(history_df, race["circuit_name"], race["race_date"])
    circuit_profile = build_circuit_profile(circuit_rows)
    model_safety_car_rate = safe_float(first.get("circuit_safety_car_rate_previous_3"))
    model_disruption_score = safe_float(first.get("circuit_race_disruption_score_avg_previous_3"))
    if model_safety_car_rate:
        circuit_profile["model_safety_car_rate_previous_3_pct"] = round(model_safety_car_rate * 100, 1)
    if model_disruption_score:
        circuit_profile["model_disruption_score_previous_3"] = round(model_disruption_score, 1)
    if model_safety_car_rate >= 0.65 or model_disruption_score >= 38:
        circuit_profile["chaos_label"] = "Chaotique"
    elif model_safety_car_rate >= 0.35 or model_disruption_score >= 22:
        circuit_profile["chaos_label"] = "Variable"

    for driver in drivers:
        driver["circuit_history"] = build_driver_circuit_history(driver, circuit_rows, circuit_profile)

    top_performers = sorted(
        drivers,
        key=lambda driver: (
            safe_float(driver.get("circuit_history", {}).get("top10_rate_pct")) * 1.4
            + max(0.0, 12 - safe_float(driver.get("circuit_history", {}).get("avg_finish") or 12))
            + safe_float(driver.get("circuit_history", {}).get("avg_position_gain")) * 2
            + min(12, safe_int(driver.get("circuit_history", {}).get("starts")) * 2),
            driver["top10_probability"],
        ),
        reverse=True,
    )

    strategy = build_strategy_context(race, group, circuit_profile)
    race["intelligence"] = {
        "circuit_profile": circuit_profile,
        "top_circuit_performers": [
            {
                "driver_code": driver["driver_code"],
                "driver_name": driver["driver_name"],
                "constructor_name": driver["constructor_name"],
                "top10_probability_pct": driver["top10_probability_pct"],
                "history": driver.get("circuit_history", {}),
            }
            for driver in top_performers[:6]
        ],
        "strategy": strategy,
        "betting_angles": build_betting_angles(drivers, race, circuit_profile),
    }
    return race


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(f"Missing neural-network predictions: {args.predictions}")

    with args.predictions.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        predictions = list(reader)

    if not predictions:
        raise ValueError(f"Predictions file is empty: {args.predictions}")
    if "race_id" not in predictions[0]:
        raise ValueError("Predictions file must include a race_id column.")

    history_df = pd.read_csv(args.history_data, low_memory=False) if args.history_data.exists() else None

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row.get("race_id", "")), []).append(row)

    races = [build_race(race_id, grouped[race_id], history_df) for race_id in sorted(grouped)]

    nn_summary = read_json(NN_SUMMARY_PATH)
    position_metrics = read_json(POSITION_METRICS_PATH)
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_predictions": str(args.predictions),
            "simulation_only": True,
            "market": "F1 top-10 finish",
            "currency": "virtual credits",
            "neural_network": nn_summary,
            "position_model": position_metrics,
            "model_comparison": model_comparison_rows(MODEL_COMPARISON_PATH),
        },
        "races": races,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"Wrote betting guide data for {len(races)} races to {args.output}")


if __name__ == "__main__":
    main()
