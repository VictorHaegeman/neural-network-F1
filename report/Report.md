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
- FastF1 for optional race-session enrichment, including weather, lap-based
  historical form features and race-control messages for supported recent
  seasons.

The current model-ready dataset contains:

- 6521 driver-race rows
- seasons from 2011 to 2026
- 198 columns after optional FastF1 race, weather, lap and race-control
  enrichment
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
- FastF1-derived pre-race circuit and season disruption history, such as safety
  car rate, red-flag rate and average disruption score

Post-race leakage columns such as final position, points, laps completed, race
status, fastest lap information and same-race race-control counts are kept for
analysis but excluded during top-10 training. Only historical race-control
aggregates from previous races are used as predictive features.

## Models

The project compares several machine learning algorithms:

- logistic regression
- random forest
- extra trees
- histogram gradient boosting
- neural network MLP
- finish-position regressors for predicted race order

The neural network is included as a baseline, but the dataset is still tabular
and relatively small. In this context, tree-based methods remain strong and
more stable.

## Dedicated Neural Network Tuning

A separate neural-network tuning step compares several MLP configurations
without replacing the main champion model. The best dedicated MLP configuration
uses two hidden layers:

- hidden layers: 128 and 64 neurons
- activation: ReLU
- alpha: 0.002
- learning rate: 0.0005
- race precision@10: 0.771

This improves the neural-network branch compared with the default MLP baseline,
but tree-based models still remain more stable overall in the rolling
season-by-season backtest.

## Finish-Position Model

The project now includes a second modeling task: predicting the likely finishing
order. This is trained as a regression/ranking problem on `final_position` and
is used to add `predicted_finish_rank` to upcoming-race exports.

The best holdout finish-position model is a neural-network MLP regressor:

- race precision@10: 0.771
- raw position MAE: 3.24 positions
- actual-top-10 rank MAE: 2.65 positions
- mean race Spearman correlation: 0.655

This model complements the binary top-10 classifier. The classifier estimates
top-10 probability, while the position model gives an expected ordering among
the drivers.

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

- accuracy: 0.770
- precision: 0.778
- recall: 0.758
- F1: 0.768
- ROC-AUC: 0.836
- race precision@10: 0.775

Across the rolling backtest, random forest remains the most stable model:

- average accuracy: 0.782
- average F1: 0.780
- average ROC-AUC: 0.849
- average race precision@10: 0.774

The dedicated tuned top-10 neural network reaches a 2025 holdout race
precision@10 of 0.771. The neural-network finish-position model is more useful
for ranking the predicted top 10 than for replacing the champion classifier.

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
- FastF1 race-control messages are available for 163/173 attempted races; the
  final 10 unavailable rows are caused by API rate limits

The final dataset was regenerated and all models were retrained after adding
historical weather and race-control history. Upcoming-race predictions were
regenerated with actual Miami qualifying data, Open-Meteo forecast weather for
Miami and a finish-position ranking model.

## How to Run

```powershell
python scripts/run_pipeline.py
```

To run with optional FastF1 enrichment:

```powershell
python scripts/run_pipeline.py --with-fastf1 --with-historical-weather --with-race-control
```

To compare all algorithms:

```powershell
python scripts/evaluate_models.py
```

To export readable race predictions:

```powershell
python scripts/predict_top10.py
```

To train and use the finish-position ranking model:

```powershell
python scripts/train_position_model.py
python scripts/predict_upcoming_races.py --season 2026 --count 4 --current-date 2026-05-03 --position-model outputs/models/finish_position_regressor.joblib
```

## Limitations

Some important racing factors are still approximate:

- race-control events and safety cars are not fully modeled
- full telemetry is not used directly
- FastF1 enrichment is partial because the timing API is rate limited
- Open-Meteo weather is historical race-day weather, not exact live race sensor
  weather
- finish-position prediction is approximate and should be read as a ranked
  expectation, not an exact finishing order

The next major improvement would be to expand FastF1 coverage and build
sequence-based features from lap-by-lap race evolution.
