# F1 Top 10 Prediction Report

## Objective

The objective of this project is to predict whether a Formula 1 driver will
finish a race in the top 10. The task is framed as a binary classification
problem where the target variable is `top10_finish`.

## Data

The project uses three public data sources:

- Jolpica F1 API, an Ergast-compatible API, for historical race results,
  qualifying results, drivers, constructors, circuits, standings and pit stops.
- Open-Meteo Archive for historical race-day weather enrichment.
- FastF1 for optional race-session enrichment, including weather and lap-based
  historical form features for supported recent seasons.

The current model-ready dataset contains:

- 6521 driver-race rows
- seasons from 2011 to 2026
- 166 columns after optional FastF1 enrichment
- no missing values after preprocessing

## Feature Engineering

The model uses only pre-race or historical features where possible. Important
feature families include:

- qualifying position and qualifying gaps
- driver and constructor standings before the race
- recent driver form over the previous 5 races
- reliability over previous races
- historical pit-stop performance
- circuit characteristics
- historical race-day weather
- optional FastF1 rolling lap and strategy features

Post-race leakage columns such as final position, points, laps completed, race
status and fastest lap information are kept for analysis but excluded during
training.

## Models

The project compares several machine learning algorithms:

- logistic regression
- random forest
- extra trees
- histogram gradient boosting
- neural network MLP

The neural network is included as a baseline, but the dataset is still tabular
and relatively small. In this context, tree-based methods remain strong and
more stable.

## Dedicated Neural Network Tuning

A separate neural-network tuning step compares several MLP configurations
without replacing the main champion model. The best dedicated MLP configuration
uses two hidden layers:

- hidden layers: 96 and 48 neurons
- activation: ReLU
- alpha: 0.003
- learning rate: 0.0008
- race precision@10: 0.779

This improves the neural-network branch compared with the default MLP baseline,
but tree-based models still remain more stable overall in the rolling
season-by-season backtest.

## Validation Strategy

The project uses temporal validation instead of a random split. This better
matches the real use case because the model should learn from past seasons and
predict future races.

Two validation views are used:

- a holdout season test
- an expanding-window season-by-season backtest

The main race-level metric is `race_precision_at_10`, which checks how many of
the predicted top 10 drivers actually finished in the top 10 for each race.

## Current Results

Current holdout season: 2025.

The current best holdout model by race precision@10 is histogram gradient
boosting:

- accuracy: 0.768
- precision: 0.763
- recall: 0.779
- F1: 0.771
- ROC-AUC: 0.835
- race precision@10: 0.767

Across the rolling backtest, random forest remains the most stable model:

- average accuracy: 0.780
- average F1: 0.779
- average ROC-AUC: 0.848
- average race precision@10: 0.768

The dedicated tuned neural network reaches a higher 2025 holdout
race precision@10 of 0.779, but its rolling-backtest average is lower than the
random forest. This makes it a useful secondary model rather than the safest
champion model.

## Latest Data Coverage Check

The latest data audit on 2026-05-03 found:

- 22 scheduled races for the 2026 season
- 3 race-result rounds available in Jolpica and already present locally
- 4 qualifying rounds available in Jolpica after Miami qualifying was published
- 4 qualifying rounds now present locally after importing Miami qualifying
- no Miami race result available yet, so the supervised training dataset still
  ends at 2026 round 3
- Open-Meteo historical weather is available for all 313 local race-result
  events
- FastF1 currently has 173 race rows, with 96 fully available race-session
  weather/lap rows and 77 unavailable rows due to timing API limits

The final dataset was regenerated and all models were retrained after adding
historical weather. Upcoming-race predictions were regenerated with actual
Miami qualifying data and Open-Meteo forecast weather for Miami.

## How to Run

```powershell
python scripts/run_pipeline.py
```

To run with optional FastF1 enrichment:

```powershell
python scripts/run_pipeline.py --with-fastf1 --with-historical-weather
```

To compare all algorithms:

```powershell
python scripts/evaluate_models.py
```

To export readable race predictions:

```powershell
python scripts/predict_top10.py
```

## Limitations

Some important racing factors are still approximate:

- race-control events and safety cars are not fully modeled
- full telemetry is not used directly
- FastF1 enrichment is partial because the timing API is rate limited
- Open-Meteo weather is historical race-day weather, not exact live race sensor
  weather

The next major improvement would be to expand FastF1 coverage and build
sequence-based features from lap-by-lap race evolution.
