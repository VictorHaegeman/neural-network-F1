# Remaining Work - F1 Top 10 Prediction

This file is the working checklist for what remains after the runnable ML
pipeline was created.

## Current State

Done:

- project structure under `scripts/`, `data/`, `outputs/`, `report/`,
  `notebooks/` and `submission/`
- Jolpica data from 2010 to 2026
- real Jolpica pit-stop events
- Open-Meteo historical race-day weather for all 333 local race-result events
- FastF1 race-control messages for 167/177 attempted races, plus historical
  pre-race circuit/season disruption features
- optional FastF1 features extended beyond 2024-2025 where the timing API and
  rate limits allowed it
- pre-race upcoming weather snapshot script with circuit-history fallback when
  a true forecast is not yet available
- richer previous-race pit-stop features rebuilt from the local pit-stop event
  table
- incremental missing-race importer that skips already-local `race_id` rows
- final dataset with 6999 rows and 201 columns
- no missing values in the final dataset
- model comparison across logistic regression, random forest, extra trees,
  histogram gradient boosting and neural network MLP
- dedicated neural network tuning across several MLP configurations
- finish-position/ranking model to predict likely order inside the top 10
- expanding-window season backtest
- data coverage audit against Jolpica API
- targeted upcoming qualifying import script
- readable top-10 prediction export
- pre-race upcoming-race prediction export
- interactive 3D neural-network hidden-space cluster visualization
- figures and metrics
- reproducible ZIP builder

## High Priority

- [x] Create a non-empty project notebook.
- [x] Generate a real `report/Report.docx` from the Markdown report.
- [x] Add a project validation script that checks important artifacts.
- [x] Rebuild the submission ZIP after artifact generation.
- [x] Add data coverage audit and import currently available Miami qualifying.
- [x] Import Miami 2026 race results and pit stops incrementally once available.
- [x] Import the missing 2010 season without refetching 2011-2026.
- [x] Generate `report/Report.pdf`.
- [x] Extend FastF1 coverage beyond 2024-2025 where API limits allow it.
- [x] Train and compare dedicated neural network configurations.
- [x] Re-engineer previous-race pit-stop features.
- [x] Add upcoming weather forecast/fallback snapshots.

## Medium Priority

- [x] Add better race-control/safety-car features from FastF1 messages.
- [ ] Resume FastF1 incremental import after the API rate limit resets to fill
  the remaining 2022-2023 gaps and any unavailable 2020 races.
- [ ] Resume FastF1 race-control incremental import after the API rate limit
  resets to fill the final 10 missing 2025 rounds.
- [x] Add a small CLI option to predict a future race once entry list data is
  available.
- [ ] Add model calibration charts.
- [ ] Add permutation importance for models that do not expose feature
  importances.
- [x] Add a 3D neural-network cluster visualization.

## Low Priority

- [x] Convert the Markdown report into a polished final PDF.
- [ ] Add a lightweight Streamlit dashboard.
- [ ] Add unit tests for feature builders.
- [ ] Add GitHub Actions for `py_compile` and project validation.

## Current Recommendation

For submission, the project now has a solid V0 with real historical weather,
real Jolpica pit-stop events, model comparison and upcoming-race prediction.
It also has a separate finish-position model for ranking the predicted top 10.
The largest remaining scientific improvements are fuller FastF1 lap/tyre and
race-control coverage after API limits reset, calibration analysis, permutation
importance, richer telemetry and real race-week forecast snapshots.
