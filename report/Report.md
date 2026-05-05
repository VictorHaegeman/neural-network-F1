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
uses three hidden layers:

- hidden layers: 128, 64 and 32 neurons
- activation: ReLU
- alpha: 0.004
- learning rate: 0.0006
- race precision@10: 0.771

This improves the neural-network branch compared with the default MLP baseline,
but tree-based models still remain more stable overall in the rolling
season-by-season backtest.

## Finish-Position Model

The project now includes a second modeling task: predicting the likely finishing
order. This is trained as a regression/ranking problem on `final_position` and
is used to add `predicted_finish_rank` to upcoming-race exports.

The best holdout finish-position model is a histogram gradient boosting
regressor:

- race precision@10: 0.775
- raw position MAE: 3.25 positions
- actual-top-10 rank MAE: 2.60 positions
- mean race Spearman correlation: 0.648

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
- precision: 0.763
- recall: 0.779
- F1: 0.771
- ROC-AUC: 0.833
- race precision@10: 0.779

Across the rolling backtest, random forest remains the most stable model:

- average accuracy: 0.782
- average F1: 0.780
- average ROC-AUC: 0.851
- average race precision@10: 0.773

The dedicated tuned top-10 neural network reaches a 2025 holdout race
precision@10 of 0.771. It remains useful as a secondary model and for hidden
space visualization, while tree-based models remain the safer champion.

## Latest Data Coverage Check

The latest data audit on 2026-05-05 found:

- 22 scheduled races for the 2026 season
- 4 race-result rounds available in Jolpica and already present locally
- 4 qualifying rounds available in Jolpica and already present locally
- Miami 2026 race results and pit stops were imported incrementally without
  refetching older races
- Open-Meteo historical weather is available for all 314 local race-result
  events
- FastF1 currently has 177 race rows, with 100 fully available race-session
  weather/lap rows and 77 unavailable rows due to timing API limits
- FastF1 race-control messages are available for 167/177 attempted races; the
  final 10 unavailable rows are caused by API rate limits

The final dataset was regenerated and all models were retrained after adding
Miami results, historical weather and race-control history. Upcoming-race
predictions now start from Canada 2026 and include a finish-position ranking
model. A 3D Plotly visualization was also added for the neural-network hidden
representation and KMeans clusters.

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
python scripts/predict_upcoming_races.py --season 2026 --count 4 --current-date 2026-05-05 --position-model outputs/models/finish_position_regressor.joblib
```

To visualize the neural-network hidden space in 3D:

```powershell
python scripts/visualize_neural_network_3d.py --color-by cluster
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
