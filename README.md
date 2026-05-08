# F1 Top-10 Prediction

This project predicts whether a Formula 1 driver will finish a Grand Prix in
the top 10. It was built for the Introduction to Machine Learning group
assignment.

The core work is a supervised tabular machine-learning comparison. A custom F1
dataset was assembled, cleaned, enriched, explored and used to train several
predictive models. Neural-network experiments and 3D visualisations are included
as extensions, but the main assignment result is the comparison of traditional
machine-learning models on the top-10 classification task.

## What Is Included

- Data import scripts for F1 results, qualifying, circuits, standings, pit stops,
  weather and race-control features.
- A final model-ready dataset covering seasons 2010-2026.
- Multiple predictive models: logistic regression, random forest, extra trees,
  histogram gradient boosting and an MLP neural network.
- Model validation on a 2025 holdout season and rolling season backtests.
- Visual outputs for EDA, model comparison and race-level predictions.
- A generated Word/PDF report and a reproducible submission ZIP.

## Project Structure

| Folder | Purpose |
|---|---|
| `data/raw/` | Raw and intermediate CSV files |
| `data/final/` | Final model-ready dataset |
| `scripts/` | Data, training, evaluation and report scripts |
| `scripts/algorithms/` | Isolated model definitions |
| `outputs/` | Metrics, figures, predictions and generated artifacts |
| `report/` | Markdown, DOCX and PDF report |
| `notebooks/` | Assignment-facing notebook |
| `submission/` | Final ZIP archive |

## Data

The project uses public Formula 1 data from Jolpica F1, an Ergast-compatible API:
https://github.com/jolpica/jolpica-f1

Weather features are enriched with Open-Meteo historical/forecast data where
available. FastF1 is used for optional session and race-control features when
the public timing API allows it.

Final dataset:

- `6,999` driver-race rows
- `201` variables
- seasons `2010-2026`
- no missing values in the final training table

## Main Models

The main target is `top10_finish`, a binary label indicating whether the driver
finished in the top 10.

The compared classifiers are:

- logistic regression
- random forest
- extra trees
- histogram gradient boosting
- neural-network MLP

The best current holdout classifier is `random_forest`, with 2025 race
precision@10 of `0.775`. The neural network is competitive, but it is kept as an
additional experiment rather than the main model. The final feature schema avoids
future-known dataset-wide experience columns; driver age and experience counters
are calculated as pre-race values.

## Setup

```powershell
$py = "$env:LocalAppData\Programs\Python\Python312\python.exe"
& $py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Windows, use `.\.venv\Scripts\python.exe` if the global `python` command
opens the Microsoft Store alias or if the `py` launcher is not available.

## Run

Run the complete local pipeline:

```powershell
python scripts/run_pipeline.py
```

Run it with optional enrichment refreshes:

```powershell
python scripts/run_pipeline.py --with-fastf1 --with-race-control --with-upcoming-weather
```

Run the main model:

```powershell
python scripts/train_model.py --model random_forest
```

Compare all models:

```powershell
python scripts/evaluate_models.py
```

Regenerate charts and race prediction renders:

```powershell
python scripts/make_charts.py
python scripts/generate_prediction_renders.py --test-season 2025 --with-headshots
python scripts/generate_showcase_images.py
```

Refresh derived raw tables and upcoming weather snapshots without refetching
data that already exists:

```powershell
python scripts/rebuild_derived_raw_tables.py
python scripts/fetch_upcoming_weather_forecast.py --season 2026 --count 4
```

Generate a visual page for one race only:

```powershell
python scripts/generate_prediction_renders.py --test-season 2025 --round 8 --with-headshots
```

Open the interactive 3D neural-network view:

```powershell
python scripts/open_neural_network_3d.py
```

Rebuild the report and submission archive:

```powershell
python scripts/build_report_docx.py
python scripts/build_report_pdf.py
python scripts/build_submission.py
python scripts/validate_project.py
```

## Main Outputs

| Output | Description |
|---|---|
| `data/final/f1_top10_model_dataset.csv` | Final dataset used for modelling |
| `outputs/model_comparison.csv` | Holdout metrics by algorithm |
| `outputs/rolling_backtest.csv` | Season-by-season validation |
| `outputs/metrics.json` | Champion model metrics |
| `docs/DATA_IMPORT_RECOMMENDATIONS.md` | Data coverage audit and next import priorities |
| `outputs/figures/*.png` | EDA and model-comparison figures |
| `outputs/figures/predictions/race_overviews/*.png` | One-page race prediction visuals |
| `outputs/figures/predictions/race_cards/*.png` | Detailed virtual podium and top-10 cards |
| `outputs/figures/showcase/` | Curated PNGs for demos: selected 2025 results and upcoming forecast cards |
| `outputs/figures/neural_network_embedding_3d.html` | Interactive 3D neural-network view |
| `outputs/figures/neural_network_embedding_3d.png` | Static 3D view for the report |
| `report/Report.docx` | Generated Word report |
| `report/Report.pdf` | Generated PDF report |
| `submission/IML_Assignment_GroupX.zip` | Submission archive |

## Current Result Summary

On the 2025 holdout season:

| Model | Race Precision@10 |
|---|---:|
| Random forest | `0.775` |
| Histogram gradient boosting | `0.771` |
| Extra trees | `0.771` |
| Neural-network MLP | `0.758` |
| Logistic regression | `0.750` |

Rolling validation shows that extra trees and random forest are the most stable
models on average, while random forest is the best latest-season holdout model.

## Notes

The report should present the project mainly as a tabular ML comparison:
dataset preparation, EDA, preprocessing, model training, validation and
analysis. The neural network, finish-position ranking model and 3D cluster view
are useful extensions, but they are not the main assignment claim.

Generated model binaries in `outputs/models/` are intentionally excluded from
the submission ZIP because they can be recreated from the scripts.
