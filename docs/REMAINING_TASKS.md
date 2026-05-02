# Remaining Work - F1 Top 10 Prediction

This file is the working checklist for what remains after the runnable ML
pipeline was created.

## Current State

Done:

- project structure under `scripts/`, `data/`, `outputs/`, `report/`,
  `notebooks/` and `submission/`
- Jolpica data from 2011 to 2026
- real Jolpica pit-stop events
- optional FastF1 features for 2024-2025
- final dataset with 6521 rows and 166 columns
- no missing values in the final dataset
- model comparison across logistic regression, random forest, extra trees,
  histogram gradient boosting and neural network MLP
- expanding-window season backtest
- readable top-10 prediction export
- figures and metrics
- reproducible ZIP builder

## High Priority

- [x] Create a non-empty project notebook.
- [x] Generate a real `report/Report.docx` from the Markdown report.
- [x] Add a project validation script that checks important artifacts.
- [x] Rebuild the submission ZIP after artifact generation.
- [ ] Optionally generate `report/Report.pdf` if a PDF engine is available.
- [ ] Extend FastF1 coverage beyond 2024-2025.

## Medium Priority

- [ ] Add better race-control/safety-car features from a richer source.
- [ ] Add a small CLI option to predict a future race once entry list data is
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

For submission, the project is strong enough once the notebook, Word report,
validation script and ZIP are regenerated. The largest scientific improvement
would be broader FastF1 coverage, but that is time-consuming and optional.
