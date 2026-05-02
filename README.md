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
circuits, pit stops and historical standings-derived features. Weather,
race-control and telemetry features are currently placeholders or derived
proxies when the public API does not provide the exact signal.

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
python scripts/run_pipeline.py --with-fastf1
```

For a quick FastF1 smoke test:

```powershell
python scripts/run_pipeline.py --with-fastf1 --fastf1-start-year 2024 --fastf1-end-year 2024 --fastf1-max-races 1 --skip-evaluation
```

Or run each step manually:

```powershell
python scripts/generate_raw_data.py --start-year 2011 --end-year 2026
python scripts/generate_fastf1_features.py --start-year 2024 --end-year 2024 --max-races 1
python scripts/generate_final_dataset.py
python scripts/train_model.py
python scripts/evaluate_models.py
python scripts/make_charts.py
python scripts/predict_top10.py
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
- `data/raw/fastf1_*.csv`: optional FastF1 enrichment tables
- `data/final/f1_top10_model_dataset.csv`: model-ready dataset
- `outputs/models/top10_classifier.joblib`: trained model
- `outputs/metrics.json`: validation metrics
- `outputs/model_comparison.csv`: holdout comparison by algorithm
- `outputs/rolling_backtest.csv`: progressive season-by-season validation
- `outputs/model_selection_summary.json`: best model summary
- `outputs/predictions/*.csv`: readable race-level prediction exports
- `outputs/figures/*.png`: EDA and model figures
- `submission/IML_Assignment_GroupX.zip`: reproducible submission archive

## Current V0 results

On the 2025 holdout season, `hist_gradient_boosting` currently performs best by
race precision@10. Across the expanding-window rolling backtest, `random_forest`
is currently the most stable model on average. The neural network is included as
a useful baseline, but this tabular dataset is still small enough that
tree-based models are expected to be strong.
