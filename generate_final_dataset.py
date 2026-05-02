from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw")
FINAL_PATH = Path("data/final")
FINAL_PATH.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FINAL_PATH / "f1_top10_model_dataset.csv"


def load_csv(filename):
    path = RAW_PATH / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_duplicate_columns(df):
    df = df.loc[:, ~df.columns.duplicated()]
    return df


# =========================
# 1. Load all datasets
# =========================

race_results = load_csv("race_results.csv")
qualifying_results = load_csv("qualifying_results.csv")
driver_info = load_csv("driver_info.csv")
team_info = load_csv("team_info.csv")
circuit_info = load_csv("circuit_info.csv")
constructor_standings = load_csv("constructor_standings.csv")
driver_standings = load_csv("driver_standings.csv")
form_data = load_csv("form_data.csv")
reliability_data = load_csv("reliability_data.csv")
lap_times = load_csv("lap_times.csv")
pit_stops = load_csv("pit_stops.csv")
weather_data = load_csv("weather_data.csv")
race_control_messages = load_csv("race_control_messages.csv")
telemetry_data = load_csv("telemetry_data.csv")


# =========================
# 2. Base dataset
# =========================

final_df = race_results.copy()

print("\nBase dataset:")
print(final_df.shape)


# =========================
# 3. Keep useful columns only
# =========================

qualifying_keep = [
    "race_id",
    "driver_id",
    "qualifying_position",
    "q1_seconds",
    "q2_seconds",
    "q3_seconds",
    "best_qualifying_seconds",
    "qualifying_gap_to_pole_seconds",
    "reached_q2",
    "reached_q3"
]

driver_info_keep = [
    "driver_id",
    "nationality",
    "date_of_birth",
    "first_season_in_dataset",
    "last_season_in_dataset",
    "number_of_seasons_in_dataset",
    "age_at_2026",
    "rookie_in_dataset"
]

team_info_keep = [
    "constructor_id",
    "constructor_nationality",
    "first_season_in_dataset",
    "last_season_in_dataset",
    "number_of_seasons_in_dataset",
    "team_experience_score"
]

circuit_info_keep = [
    "race_id",
    "circuit_length_km",
    "total_laps",
    "race_distance_km",
    "track_type",
    "overtaking_difficulty",
    "street_circuit"
]

constructor_standings_keep = [
    "race_id",
    "constructor_id",
    "constructor_position_before_race",
    "constructor_points_before_race",
    "constructor_wins_before_race",
    "constructor_performance_score"
]

driver_standings_keep = [
    "race_id",
    "driver_id",
    "driver_position_before_race",
    "driver_points_before_race",
    "driver_wins_before_race",
    "driver_performance_score"
]

form_data_keep = [
    "race_id",
    "driver_id",
    "races_count_previous_5",
    "top10_finishes_previous_5",
    "top10_rate_previous_5",
    "points_previous_5",
    "avg_points_previous_5",
    "points_finishes_previous_5",
    "avg_finish_position_previous_5",
    "avg_grid_position_previous_5",
    "best_finish_previous_5",
    "worst_finish_previous_5",
    "avg_laps_completed_previous_5",
    "dnf_previous_5"
]

reliability_keep = [
    "race_id",
    "driver_id",
    "dnf_previous_5",
    "dns_previous_5",
    "mechanical_dnf_previous_5",
    "accident_dnf_previous_5",
    "disqualified_previous_5",
    "finish_rate_previous_5",
    "reliability_score"
]

lap_times_keep = [
    "race_id",
    "driver_id",
    "races_with_lap_data_previous_3",
    "avg_lap_time_previous_3_races",
    "best_lap_time_previous_3_races",
    "lap_time_std_previous_3_races",
    "avg_race_pace_rank_previous_3_races",
    "avg_completed_laps_previous_3_races"
]

pit_stops_keep = [
    "race_id",
    "driver_id",
    "races_with_pit_data_previous_3",
    "avg_pit_stop_time_previous_3_races",
    "best_pit_stop_time_previous_3_races",
    "worst_pit_stop_time_previous_3_races",
    "total_pit_stops_previous_3_races",
    "avg_total_pit_time_previous_3_races",
    "avg_pit_stop_rank_previous_3_races"
]

weather_keep = [
    "race_id",
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
    "weather_data_available"
]

race_control_keep = [
    "race_id",
    "safety_car_count",
    "virtual_safety_car_count",
    "red_flag_count",
    "yellow_flag_count",
    "double_yellow_count",
    "black_flag_count",
    "track_limits_count",
    "investigation_count",
    "penalty_count",
    "incident_count",
    "race_control_messages_count",
    "race_control_data_available",
    "total_dnf_count",
    "classified_driver_count",
    "race_disruption_score"
]

telemetry_keep = [
    "race_id",
    "driver_id",
    "telemetry_races_previous_3",
    "avg_speed_previous_3_races",
    "max_speed_previous_3_races",
    "avg_throttle_previous_3_races",
    "avg_brake_previous_3_races",
    "avg_drs_previous_3_races",
    "drs_usage_rate_previous_3_races",
    "avg_distance_previous_3_races"
]


# =========================
# 4. Reduce datasets
# =========================

qualifying_results = qualifying_results[[col for col in qualifying_keep if col in qualifying_results.columns]]
driver_info = driver_info[[col for col in driver_info_keep if col in driver_info.columns]]
team_info = team_info[[col for col in team_info_keep if col in team_info.columns]]
circuit_info = circuit_info[[col for col in circuit_info_keep if col in circuit_info.columns]]
constructor_standings = constructor_standings[[col for col in constructor_standings_keep if col in constructor_standings.columns]]
driver_standings = driver_standings[[col for col in driver_standings_keep if col in driver_standings.columns]]
form_data = form_data[[col for col in form_data_keep if col in form_data.columns]]
reliability_data = reliability_data[[col for col in reliability_keep if col in reliability_data.columns]]
lap_times = lap_times[[col for col in lap_times_keep if col in lap_times.columns]]
pit_stops = pit_stops[[col for col in pit_stops_keep if col in pit_stops.columns]]
weather_data = weather_data[[col for col in weather_keep if col in weather_data.columns]]
race_control_messages = race_control_messages[[col for col in race_control_keep if col in race_control_messages.columns]]
telemetry_data = telemetry_data[[col for col in telemetry_keep if col in telemetry_data.columns]]


# Rename columns to avoid confusion
driver_info = driver_info.rename(columns={
    "nationality": "driver_info_nationality",
    "first_season_in_dataset": "driver_first_season_in_dataset",
    "last_season_in_dataset": "driver_last_season_in_dataset",
    "number_of_seasons_in_dataset": "driver_number_of_seasons_in_dataset"
})

team_info = team_info.rename(columns={
    "first_season_in_dataset": "team_first_season_in_dataset",
    "last_season_in_dataset": "team_last_season_in_dataset",
    "number_of_seasons_in_dataset": "team_number_of_seasons_in_dataset"
})

# Avoid duplicate dnf_previous_5 from form_data and reliability_data
form_data = form_data.rename(columns={
    "dnf_previous_5": "form_dnf_previous_5"
})


# =========================
# 5. Merge datasets
# =========================

merge_steps = [
    ("qualifying_results", qualifying_results, ["race_id", "driver_id"]),
    ("driver_info", driver_info, ["driver_id"]),
    ("team_info", team_info, ["constructor_id"]),
    ("circuit_info", circuit_info, ["race_id"]),
    ("constructor_standings", constructor_standings, ["race_id", "constructor_id"]),
    ("driver_standings", driver_standings, ["race_id", "driver_id"]),
    ("form_data", form_data, ["race_id", "driver_id"]),
    ("reliability_data", reliability_data, ["race_id", "driver_id"]),
    ("lap_times", lap_times, ["race_id", "driver_id"]),
    ("pit_stops", pit_stops, ["race_id", "driver_id"]),
    ("weather_data", weather_data, ["race_id"]),
    ("race_control_messages", race_control_messages, ["race_id"]),
    ("telemetry_data", telemetry_data, ["race_id", "driver_id"])
]

for name, dataset, keys in merge_steps:
    before_shape = final_df.shape

    final_df = final_df.merge(
        dataset,
        on=keys,
        how="left"
    )

    final_df = clean_duplicate_columns(final_df)

    after_shape = final_df.shape

    print(f"Merged {name}: {before_shape} -> {after_shape}")


# =========================
# 6. Clean target and IDs
# =========================

final_df["top10_finish"] = pd.to_numeric(
    final_df["top10_finish"],
    errors="coerce"
)

final_df = final_df.dropna(subset=["top10_finish"])

final_df["top10_finish"] = final_df["top10_finish"].astype(int)


# =========================
# 7. Convert dates
# =========================

if "race_date" in final_df.columns:
    final_df["race_date"] = pd.to_datetime(final_df["race_date"], errors="coerce")
    final_df["race_month"] = final_df["race_date"].dt.month
    final_df["race_day"] = final_df["race_date"].dt.day


if "driver_date_of_birth" in final_df.columns:
    final_df["driver_date_of_birth"] = pd.to_datetime(
        final_df["driver_date_of_birth"],
        errors="coerce"
    )

    final_df["driver_age_at_race"] = (
        final_df["race_date"].dt.year - final_df["driver_date_of_birth"].dt.year
    )


# =========================
# 8. Convert numeric columns
# =========================

for col in final_df.columns:
    if col not in [
        "race_id",
        "grand_prix",
        "race_date",
        "circuit_id",
        "circuit_name",
        "country",
        "locality",
        "driver_id",
        "driver_code",
        "driver_name",
        "driver_nationality",
        "driver_info_nationality",
        "driver_date_of_birth",
        "constructor_id",
        "constructor_name",
        "constructor_nationality",
        "weather_condition",
        "track_type",
        "overtaking_difficulty",
        "status",
        "position_text",
        "fastest_lap_time"
    ]:
        final_df[col] = pd.to_numeric(final_df[col], errors="ignore")


# =========================
# 9. Basic missing value handling
# =========================

numeric_cols = final_df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = final_df.select_dtypes(include=["object"]).columns.tolist()

for col in numeric_cols:
    if col != "top10_finish":
        final_df[col] = final_df[col].fillna(final_df[col].median())

for col in categorical_cols:
    final_df[col] = final_df[col].fillna("Unknown")


# =========================
# 10. Remove obvious leakage columns marker
# =========================
# Important:
# We keep these columns in the dataset for analysis,
# but during model training we will exclude:
# final_position, points, laps, status, position_text,
# fastest_lap_rank, fastest_lap_number, fastest_lap_time,
# fastest_lap_avg_speed_kph, top10_finish.


# =========================
# 11. Sort and export
# =========================

final_df = final_df.sort_values(
    by=["season", "round", "final_position", "driver_id"],
    ascending=[True, True, True, True]
)

final_df.to_csv(OUTPUT_PATH, index=False)

print("\nFinal dataset generated successfully.")
print(f"File: {OUTPUT_PATH}")
print(f"Rows: {final_df.shape[0]}")
print(f"Columns: {final_df.shape[1]}")

print("\nRows per season:")
print(final_df.groupby("season").size())

print("\nTarget distribution:")
print(final_df["top10_finish"].value_counts())

print("\nMissing values total:")
print(final_df.isna().sum().sum())

print("\nPreview:")
print(final_df.head(20))