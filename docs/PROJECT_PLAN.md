# F1 Top 10 Prediction - Project Plan

This document tracks the project in concrete milestones. Each milestone should
be runnable, tested, committed and pushed before moving to the next one.

## Current Baseline

Status: runnable V1.

Implemented:

- Jolpica/Ergast-compatible data import from 2011 to 2026
- race results, qualifying, drivers, constructors, circuits and standings
- real Jolpica pit-stop events and previous-race pit-stop features
- final model dataset with no missing values
- multiple algorithms: logistic regression, random forest, extra trees,
  histogram gradient boosting, neural network MLP
- season holdout validation and expanding-window rolling backtest
- project charts and model comparison outputs
- one-command pipeline through `scripts/run_pipeline.py`

Current limitations:

- weather features are still placeholders
- race-control/safety-car features are still placeholders/proxies
- telemetry features are still proxies, not real FastF1 telemetry
- neural network is a tabular MLP baseline, not a sequence model
- submission ZIP/report/notebook still need to be regenerated

## Milestone 1 - Stabilize Project Structure

Goal: make the project easy to run and hard to lose.

Tasks:

- keep all executable Python files under `scripts/`
- keep raw data under `data/raw/`
- keep model-ready data under `data/final/`
- keep metrics, charts and models under `outputs/`
- keep assignment artifacts under `report/`, `notebooks/`, `submission/`
- document the one-command run flow

Validation:

```powershell
python scripts/run_pipeline.py --skip-evaluation
```

Git checkpoint:

```powershell
git add -A
git commit -m "Document project execution plan"
git push origin main
```

## Milestone 2 - Add FastF1 Enrichment

Goal: replace placeholder weather/lap/strategy features with richer data where
FastF1 can provide it.

Planned tasks:

- add `fastf1` to `requirements.txt`
- create `scripts/generate_fastf1_features.py`
- use a local ignored cache directory under `.fastf1_cache/`
- fetch race sessions with laps and weather enabled
- generate optional raw tables:
  - `data/raw/fastf1_weather.csv`
  - `data/raw/fastf1_lap_features.csv`
  - `data/raw/fastf1_strategy_features.csv`
- keep the FastF1 step optional so the Jolpica baseline still runs quickly

Validation:

```powershell
python scripts/generate_fastf1_features.py --start-year 2024 --end-year 2024 --max-races 1
python scripts/generate_final_dataset.py
python scripts/train_model.py --model random_forest
```

Git checkpoint:

```powershell
git add -A
git commit -m "Add optional FastF1 feature generation"
git push origin main
```

## Milestone 3 - Integrate FastF1 Into Modeling

Goal: use FastF1 features when available without breaking the existing dataset.

Tasks:

- merge FastF1 weather/lap/strategy CSV files in `generate_final_dataset.py`
- add availability flags for optional FastF1 features
- update charts to show feature coverage
- rerun model comparison

Validation:

```powershell
python scripts/run_pipeline.py --skip-evaluation
python scripts/evaluate_models.py
python scripts/make_charts.py
```

Git checkpoint:

```powershell
git add -A
git commit -m "Integrate FastF1 features into training dataset"
git push origin main
```

## Milestone 4 - Add Prediction Outputs

Goal: produce a useful top-10 prediction table, not only metrics.

Tasks:

- create `scripts/predict_top10.py`
- output `outputs/predictions/latest_race_top10_predictions.csv`
- include driver, constructor, race, predicted probability and rank
- optionally predict the latest completed race as a sanity check

Validation:

```powershell
python scripts/predict_top10.py
```

Git checkpoint:

```powershell
git add -A
git commit -m "Add top 10 prediction export"
git push origin main
```

## Milestone 5 - Report and Submission

Goal: make the project submit-ready.

Tasks:

- create a reproducible notebook or report-ready Markdown summary
- regenerate figures
- summarize dataset, features, models, evaluation and limitations
- rebuild `submission/IML_Assignment_GroupX.zip`

Validation:

```powershell
python scripts/run_pipeline.py
python scripts/make_charts.py
```

Git checkpoint:

```powershell
git add -A
git commit -m "Prepare assignment submission artifacts"
git push origin main
```

## Push Policy

Push after every milestone that:

- changes data generation
- changes model training/evaluation
- regenerates datasets or charts
- reorganizes project structure
- creates report or submission artifacts

Do not push:

- `.venv/`
- `.fastf1_cache/`
- `outputs/models/*.joblib`
- Python cache folders
