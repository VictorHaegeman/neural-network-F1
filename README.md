# f1-top10-prediction

Pipeline for predicting whether a Formula 1 driver finishes in the top 10.

## Project structure

- `scripts/`: data import, feature generation, training, evaluation and plotting scripts
- `data/raw/`: fetched and derived raw feature tables
- `data/final/`: model-ready dataset
- `outputs/`: metrics, comparison files, figures and locally generated models
- `notebooks/`, `report/`, `submission/`: assignment-facing material

## Data source

Raw data is fetched from the Jolpica F1 API, an Ergast-compatible public API:
https://github.com/jolpica/jolpica-f1

The importer uses Jolpica for race results, qualifying, drivers, constructors,
circuits, pit stops and historical standings-derived features. Race-day weather
is enriched with the Open-Meteo Archive where historical weather is available.
FastF1 race-control messages are used to build pre-race circuit/season
disruption history features. Telemetry features are still derived proxies when
the public API does not provide the exact signal.

Optional FastF1 enrichment can add richer race-session weather and historical
lap/strategy features for seasons supported by the F1 timing API.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, prefer `.\.venv\Scripts\python.exe` if the global `python`
command opens the Microsoft Store alias.

## Run the full V0

```powershell
python scripts/run_pipeline.py
```

By default, `run_pipeline.py` reuses existing `data/raw` files. To fetch fresh
raw data again:

```powershell
python scripts/run_pipeline.py --force-fetch
```

To include optional FastF1 feature generation:

```powershell
python scripts/run_pipeline.py --with-fastf1 --with-historical-weather --with-race-control
```

For a quick FastF1 smoke test:

```powershell
python scripts/run_pipeline.py --with-fastf1 --fastf1-start-year 2024 --fastf1-end-year 2024 --fastf1-max-races 1 --skip-evaluation
```

Or run each step manually:

```powershell
python scripts/generate_raw_data.py --start-year 2011 --end-year 2026
python scripts/audit_data_coverage.py --season 2026
python scripts/import_missing_completed_races.py --season 2026
python scripts/generate_historical_weather.py --start-year 2011 --end-year 2026 --incremental
python scripts/generate_fastf1_features.py --start-year 2018 --end-year 2026 --incremental
python scripts/generate_fastf1_race_control.py --start-year 2018 --end-year 2026 --incremental
python scripts/generate_final_dataset.py
python scripts/train_model.py --model hist_gradient_boosting
python scripts/tune_neural_network.py --force
python scripts/train_position_model.py
python scripts/evaluate_models.py
python scripts/make_charts.py
python scripts/visualize_neural_network_3d.py --color-by cluster
python scripts/predict_top10.py
python scripts/predict_upcoming_races.py --season 2026 --count 4 --current-date 2026-05-05 --position-model outputs/models/finish_position_regressor.joblib
python scripts/predict_upcoming_races.py --season 2026 --count 4 --current-date 2026-05-05 --model outputs/models/top10_neural_network_mlp.joblib --position-model outputs/models/finish_position_regressor.joblib --output-dir outputs/predictions/neural_network
python scripts/build_submission.py
python scripts/validate_project.py
```

You can train a specific algorithm:

```powershell
python scripts/train_model.py --model random_forest
python scripts/train_model.py --model hist_gradient_boosting
python scripts/train_model.py --model neural_network_mlp
```

Available algorithms:

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `neural_network_mlp`

To compare all models and run an expanding-window season backtest:

```powershell
python scripts/evaluate_models.py
```

Main outputs:

- `data/raw/*.csv`: fetched and derived raw feature tables
- `data/raw/pit_stop_events.csv`: real pit-stop events from Jolpica
- `data/raw/weather_data.csv`: Open-Meteo race-day weather enrichment
- `data/raw/fastf1_race_control.csv`: FastF1 race-control message counts
- `data/raw/race_control_history.csv`: pre-race circuit/season disruption features
- `data/raw/upcoming_qualifying_results.csv`: latest imported upcoming qualifying snapshot
- `data/raw/fastf1_*.csv`: optional FastF1 enrichment tables
- `data/final/f1_top10_model_dataset.csv`: model-ready dataset
- `outputs/data_coverage_report.csv`: local/API data coverage audit
- `outputs/models/top10_classifier.joblib`: trained model
- `outputs/metrics.json`: validation metrics
- `outputs/model_comparison.csv`: holdout comparison by algorithm
- `outputs/neural_network_tuning.csv`: dedicated MLP hyperparameter comparison
- `outputs/neural_network_summary.json`: best dedicated MLP configuration
- `outputs/position_model_comparison.csv`: finish-position ranking model comparison
- `outputs/position_model_metrics.json`: best finish-position model summary
- `outputs/rolling_backtest.csv`: progressive season-by-season validation
- `outputs/model_selection_summary.json`: best model summary
- `outputs/neural_network_embedding_3d.csv`: neural-network hidden-space PCA/clusters
- `outputs/predictions/*.csv`: readable race-level prediction exports
- `outputs/predictions/upcoming_top10_predictions.csv`: pre-race predictions for upcoming races
- `outputs/figures/*.png`: EDA and model figures
- `outputs/figures/neural_network_embedding_3d.html`: interactive 3D neural-network cluster view
- `submission/IML_Assignment_GroupX.zip`: reproducible submission archive

## Current V0 results

On the 2025 holdout season, `hist_gradient_boosting` currently performs best
among the standard model comparison set by race precision@10 (`0.779`). Across
the expanding-window rolling backtest, `random_forest` is currently the most
stable model on average (`0.773` race precision@10). The dedicated tuned neural
network reaches `0.771` race precision@10 with the `mlp_128_64_32_deep`
configuration, so it is kept as a secondary top-10 classifier rather than the
safest champion.

A separate finish-position model predicts race order directly. The best
position model is currently a histogram gradient boosting regressor, with 2025
holdout race precision@10 of `0.775`, actual-top-10 rank MAE of `2.60`
positions and mean race Spearman correlation of `0.648`. Upcoming-race exports therefore now
include both `top10_probability` and `predicted_finish_rank`.

Upcoming-race predictions use the latest completed race as the driver/team
state, Jolpica for the calendar and qualifying when available, Open-Meteo for
near-term forecasts, circuit history for longer-range weather fallback, and
FastF1-derived historical tyre/lap features where available.

Current coverage notes:

- Open-Meteo historical weather: 314/314 local race-result events.
- FastF1 optional enrichment: 177 race rows, 100 with available timing/weather
  data and 77 unavailable rows because of API limits.
- FastF1 race-control messages: 167/177 attempted race rows available; the
  last 10 rounds of 2025 were rate-limited and can be filled by rerunning the
  incremental script later.
- 2026 data audit on 2026-05-05: 4 race-result rounds, 4 qualifying rounds,
  4 FastF1 weather rows and 4 final-dataset races are available locally.
