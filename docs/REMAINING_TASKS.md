# Remaining Work - F1 Top 10 Prediction

This file is the working checklist for what remains after the runnable ML
pipeline was created.

## Current State

Done:

- project structure under `scripts/`, `data/`, `outputs/`, `report/`,
  `notebooks/` and `submission/`
- Jolpica data from 2011 to 2026
- real Jolpica pit-stop events
- Open-Meteo historical race-day weather for all 313 local race-result events
- optional FastF1 features extended beyond 2024-2025 where the timing API and
  rate limits allowed it
- final dataset with 6521 rows and 166 columns
- no missing values in the final dataset
- model comparison across logistic regression, random forest, extra trees,
  histogram gradient boosting and neural network MLP
- dedicated neural network tuning across several MLP configurations
- expanding-window season backtest
- data coverage audit against Jolpica API
- targeted upcoming qualifying import script
- readable top-10 prediction export
- pre-race upcoming-race prediction export
- figures and metrics
- reproducible ZIP builder

## High Priority

- [x] Create a non-empty project notebook.
- [x] Generate a real `report/Report.docx` from the Markdown report.
- [x] Add a project validation script that checks important artifacts.
- [x] Rebuild the submission ZIP after artifact generation.
- [x] Add data coverage audit and import currently available Miami qualifying.
- [ ] Optionally generate `report/Report.pdf` if a PDF engine is available.
- [x] Extend FastF1 coverage beyond 2024-2025 where API limits allow it.
- [x] Train and compare dedicated neural network configurations.

## Medium Priority

- [ ] Add better race-control/safety-car features from a richer source.
- [ ] Re-run raw import after Miami 2026 race results are available.
- [ ] Resume FastF1 incremental import after the API rate limit resets to fill
  the remaining 2020-2023 gaps.
- [x] Add a small CLI option to predict a future race once entry list data is
  available.
- [ ] Add model calibration charts.
- [ ] Add permutation importance for models that do not expose feature
  importances.

## Low Priority

- [ ] Convert the Markdown report into a polished final PDF.
- [ ] Add a lightweight Streamlit dashboard.
- [ ] Add unit tests for feature builders.
- [ ] Add GitHub Actions for `py_compile` and project validation.

## Current Recommendation

For submission, the project now has a solid V0 with real historical weather,
real Jolpica pit-stop events, model comparison and upcoming-race prediction.
The largest remaining scientific improvements are fuller FastF1 lap/tyre
coverage after API limits reset, richer race-control features and calibration
analysis.
