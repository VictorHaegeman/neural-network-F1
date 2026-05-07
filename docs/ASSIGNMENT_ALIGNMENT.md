# Assignment Alignment Check

This checklist maps the current project to the CX016-2.5-3-IML group assignment requirements.

## Dataset Requirements

| Requirement | Current status | Evidence |
|---|---|---|
| One dataset | Satisfied | `data/final/f1_top10_model_dataset.csv` is the single model-ready dataset |
| Reasonable size | Satisfied | 6,999 driver-race rows |
| Not perfectly clean originally | Satisfied | Raw sources require merging, missing historical pit stops, partial FastF1 coverage and weather enrichment |
| Categorical and numeric data | Satisfied | 174 numeric/bool columns and 24 categorical/text columns |
| More than 12 variables | Satisfied | 198 final columns |
| Not a commonly experimented dataset | Satisfied | Custom Formula 1 top-10 prediction dataset assembled from public APIs |
| If imbalanced, address it | Satisfied | Target is reasonably balanced: 3,330 top-10 rows vs 3,669 non-top-10 rows; class weights are used where relevant |

## Report Requirements

| Required section | Current status | Evidence |
|---|---|---|
| Title and abstract | Satisfied | Styled cover page plus abstract in `report/Report.md`, `report/Report.docx`, `report/Report.pdf` |
| APU-style readability | Satisfied | 12pt report export, generated cover page, table of contents, styled tables and labelled figures |
| Introduction, aim and objectives | Satisfied | Report section added |
| Related works | Satisfied, can still be polished | Report includes related-work summary and references |
| Methods | Satisfied | Report describes data, features, models, packages and metrics |
| Dataset preparation and EDA | Satisfied | Report plus generated labelled EDA PNG figures |
| Model implementation | Satisfied | At least five classifiers compared; two-model minimum exceeded |
| Optimization/tuning | Satisfied | Model configurations and MLP tuning documented |
| Model validation | Satisfied | 2025 holdout, expanding-window rolling backtest and labelled algorithm PNG summaries |
| Analysis and recommendations | Satisfied | Report compares models and recommends HGB classifier |
| Concrete experiment outputs | Satisfied | Race-level prediction CSVs and virtual podium PNG renders are generated for the 2025 holdout season |
| Conclusion | Satisfied | Report includes project self-evaluation and future work |
| References | Satisfied, can still be improved | APA-style reference list added |
| Acknowledgements | Satisfied | Report includes an acknowledgements section as requested by the brief |

## Code and Submission Requirements

| Requirement | Current status | Evidence |
|---|---|---|
| All source code in `.ipynb` format | Satisfied | `notebooks/ML_Project_Code.ipynb` provides assignment-facing code |
| Reproducible scripts | Satisfied | `scripts/` contains import, preprocessing, training, evaluation and validation scripts |
| Code documentation | Satisfied | `README.md`, `docs/ALGORITHMS.md`, report code map and notebook explain the source code organization |
| Original/final dataset included | Satisfied | `data/` is included in the submission zip |
| Compressed submission | Satisfied | `submission/IML_Assignment_GroupX.zip` |
| Caches and model binaries excluded | Satisfied | validation checks exclude `.venv`, FastF1 cache and model binaries |

## Current Recommendation

For the graded submission, present the project as a classical supervised tabular ML task:

1. Main task: binary top-10 finish classification.
2. Main models: histogram gradient boosting, random forest, logistic regression, extra trees.
3. Neural network: secondary comparison and optional extension.
4. 3D visualization: optional appendix/demo, not the core assignment claim.

This framing is closer to the assignment than presenting the project mainly as a neural-network visualization project.
