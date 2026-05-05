# F1 Top-10 Finish Prediction with Machine Learning

## Abstract

This project investigates whether machine learning can predict if a Formula 1 driver will finish a Grand Prix in the top 10. This is a useful sporting analytics problem because top-10 classification is directly linked to points scoring, race strategy and team performance evaluation. A single model-ready dataset was built from public Formula 1 data covering 2010 to 2026, combining race results, qualifying, standings, circuit information, historical pit-stop indicators, weather, and selected FastF1-derived features. The dataset was cleaned, merged and transformed into 6,999 driver-race observations with 198 variables and no missing values. The main predictive task is binary classification with `top10_finish` as the target. Logistic regression, random forest, extra trees, histogram gradient boosting and a multilayer perceptron were compared using temporal validation. The best holdout model is histogram gradient boosting, with 2025 race precision@10 of 0.779. A separate neural-network regressor was also trained as a ranking extension for predicted finishing order, but the primary assignment result remains the comparison and validation of predictive models on the tabular top-10 classification task.

## Introduction, Aim and Objectives

Formula 1 results are influenced by many interacting factors: car performance, driver ability, qualifying position, reliability, circuit type, weather and race strategy. This makes the problem more challenging than a simple ranking by previous points. The goal of this project is to use historical and pre-race information to estimate whether each driver will finish inside the top 10.

The aim is to build, evaluate and compare machine learning models for Formula 1 top-10 finish prediction.

The objectives are:

- collect and consolidate a reasonably large, non-trivial Formula 1 dataset
- include both numeric and categorical features
- perform exploratory data analysis and preprocessing
- engineer predictive features that avoid direct post-race leakage
- train at least two predictive machine learning models
- tune and compare models using appropriate validation metrics
- interpret the results and recommend the most suitable model

The scope is supervised prediction at driver-race level. The main output is a probability and ranking of likely top-10 finishers. Exact race simulation, live telemetry modelling and lap-by-lap race strategy optimization are treated as future work rather than core assignment requirements.

## Related Works

Previous Formula 1 analytics work often focuses on lap-time prediction, race outcome prediction, tire degradation or separating driver and constructor effects. These studies informed the feature choices in this project: qualifying position and team strength are important baseline predictors, while circuit, weather, tire and historical form features can improve the model.

| Source | Focus | Relevance to this project |
|---|---|---|
| Breiman (2001) | Random forests | Motivates using tree ensembles for mixed tabular data |
| Friedman (2001) | Gradient boosting machines | Supports boosting as a strong tabular baseline |
| Pedregosa et al. (2011) | scikit-learn | Main software library used for modelling pipelines |
| Nigro (2020) | F1 race predictor | Shows practical race prediction using historical F1 data |
| Zhao (2024) | Deep neural network lap-time forecasting | Shows neural networks can model F1 performance signals |
| Jafri (2024) | F1 race outcome prediction | Uses historical race data for race prediction |
| Noe and Patel (2024) | Telemetry-based lap prediction | Motivates lap and performance features |
| Cappello and Hoegh (2026) | Tire degradation modelling | Supports the value of tire and strategy variables |
| race-outcome decomposition work (2025) | Driver vs constructor effects | Motivates separating driver and constructor strength |
| Jolpica F1 API | Ergast-compatible data source | Provides race, qualifying, standings and pit-stop data |
| FastF1 | Timing, lap and race-control data | Optional enrichment for recent seasons |
| Open-Meteo Archive | Historical weather | Adds race-day weather context |

Compared with much of the related work, this project does not try to predict the race winner only. It predicts top-10 classification for every driver-race row, which creates a larger supervised dataset and aligns with the points-scoring structure of modern F1.

## Methods

The final dataset is `data/final/f1_top10_model_dataset.csv`. It contains 6,999 rows, 198 columns, 174 numeric/bool columns and 24 categorical/text columns. Each row represents one driver in one race. The label is `top10_finish`, equal to 1 when the final classified result is tenth place or better, and 0 otherwise.

The main software packages are:

- pandas and numpy for data manipulation
- scikit-learn for preprocessing, pipelines, model training and metrics
- matplotlib and seaborn for EDA figures
- joblib for model serialization
- Plotly for optional neural-network visualization

The main models compared are:

- logistic regression
- random forest classifier
- extra trees classifier
- histogram gradient boosting classifier
- multilayer perceptron classifier

The primary evaluation metrics are accuracy, precision, recall, F1, ROC-AUC and race precision@10. Race precision@10 is important because the practical output is a race-level top-10 list rather than only independent row-level labels.

## Dataset Preparation and EDA

The assignment requires one reasonably large dataset with categorical and numeric data. Although several public sources were used, they are merged into one final modelling dataset. The raw tables are retained for reproducibility, while the final CSV is the single dataset used for modelling.

Dataset summary:

- rows: 6,999 driver-race observations
- seasons: 2010 to 2026
- target classes: 3,330 top-10 rows and 3,669 non-top-10 rows
- missing values after preprocessing: 0
- weather coverage: 333/333 local race-result events
- 2026 coverage on 2026-05-05: 4 completed races and 4 qualifying sessions locally available

The dataset is not perfectly clean in its original form. Race APIs use different identifiers, some historical pit-stop data is unavailable before 2011, FastF1 timing coverage is partial, and weather data is an external historical approximation rather than exact FIA sensor data. The preprocessing pipeline handles these issues by using availability flags, deterministic fallbacks, imputation and careful feature merging.

Key EDA observations:

- The target is reasonably balanced for a binary classification task: about 48% top-10 rows and 52% non-top-10 rows.
- Rows per season vary because the number of races and drivers changes over time.
- Qualifying/grid position is strongly related to finishing position, but it is not enough alone because reliability, circuit, weather and team performance can change the outcome.
- Categorical fields such as driver, constructor, circuit and weather condition require encoding before model training.

Preprocessing steps:

- merge raw race, qualifying, standings, weather, circuit and optional FastF1 tables
- create target variable `top10_finish`
- remove or exclude post-race leakage features during training, such as final position, points, fastest lap and same-race race-control counts
- impute numeric features with median values
- impute categorical features with the most frequent value
- one-hot encode categorical variables
- scale numeric features for logistic regression and neural-network models

The generated figures in `outputs/figures/` document target distribution, rows by season, top-10 rate by season, grid versus finish, model comparison, rolling backtest, feature importance and confusion matrix. Additional assignment-facing PNGs summarize the full pipeline, holdout algorithm results, validation metric table and expanding-window stability. These figures support the EDA, model implementation and validation sections with labelled evidence.

## Model Implementation

The modelling workflow is script-driven and reproducible:

```powershell
python scripts/generate_final_dataset.py
python scripts/evaluate_models.py
python scripts/train_model.py --model hist_gradient_boosting
python scripts/train_position_model.py
python scripts/make_charts.py
python scripts/build_report_docx.py
python scripts/build_report_pdf.py
python scripts/validate_project.py
```

For the main classification task, the selected champion model is histogram gradient boosting. This model was selected because it achieved the best 2025 holdout race precision@10 and handles non-linear interactions in tabular data well. Random forest remains the most stable model in rolling backtest, so it is kept as an important comparison baseline.

Hyperparameter optimization is included through fixed, documented model configurations and a dedicated neural-network tuning script. The neural-network tuning compares multiple MLP architectures and learning rates. The best dedicated classifier MLP uses hidden layers of 128 and 64 neurons with ReLU activation, alpha 0.002 and learning rate 0.0005.

## Model Validation

A temporal validation strategy is used instead of a random split. This is more realistic because a model should learn from past seasons and predict future races. The main holdout test season is 2025. An expanding-window rolling backtest is also used to compare stability across seasons.

Current holdout results on the 2025 season:

| Model | Accuracy | F1 | ROC-AUC | Race precision@10 |
|---|---:|---:|---:|---:|
| Histogram gradient boosting | 0.785 | 0.784 | 0.831 | 0.779 |
| Random forest | 0.779 | 0.786 | 0.844 | 0.775 |
| Neural network MLP | 0.766 | 0.771 | 0.829 | 0.771 |
| Extra trees | 0.752 | 0.761 | 0.833 | 0.750 |
| Logistic regression | 0.743 | 0.698 | 0.811 | 0.746 |

Rolling backtest averages show that random forest is the most stable model by race precision@10, with an average of 0.767. Histogram gradient boosting is still selected as the holdout champion because it has the strongest latest-season top-10 performance.

## Analysis and Recommendations

The results are broadly in line with expectations. Tree-based models perform strongly because the dataset is tabular, mixed-type and feature-engineered. Logistic regression is useful as a simple baseline but is less able to capture non-linear interactions such as driver-team-circuit effects. The neural-network classifier is competitive but does not clearly beat tree ensembles on this dataset. This supports the decision to present the neural network as an extension rather than the primary assignment result.

Adding the 2010 season increased the dataset from 6,543 to 6,999 rows. The holdout accuracy and F1 improved for the champion classifier, while race precision@10 stayed at 0.779. This suggests that extra historical data helps general classification stability, but very old F1 seasons are not always directly representative of modern seasons.

The separate finish-position ranking model is useful for producing ordered race predictions. Its best current version is a neural-network MLP regressor with race precision@10 of 0.779 and mean race Spearman correlation of 0.660. This is helpful for interpretation, but it should remain secondary because the assignment asks mainly for predictive model comparison.

Recommended final model:

- main assignment model: histogram gradient boosting classifier
- supporting baseline: random forest classifier
- optional extension: tuned MLP classifier and finish-position regressor

Generated report deliverables:

- `report/Report.md`: editable report source
- `report/Report.docx`: Word report version
- `report/Report.pdf`: PDF report version generated from the same source
- `outputs/figures/*.png`: labelled EDA, algorithm and validation figures used as report evidence

Recommended future improvements:

- add calibration plots for probability quality
- add permutation importance for model interpretability
- fill remaining FastF1 gaps after rate limits reset
- test whether older seasons should be down-weighted because modern F1 regulations differ
- improve the report with more domain-specific literature if more time is available

## Conclusion

This project satisfies the core assignment requirement by building a machine learning solution on one reasonably large, mixed-type and non-trivial dataset. The work includes data collection, cleaning, feature engineering, EDA, preprocessing, multiple predictive models, tuning, validation and comparison. The strongest model for the main top-10 classification task is histogram gradient boosting, with 2025 holdout race precision@10 of 0.779. The project also shows that neural networks are interesting but not automatically superior for this tabular problem. The main weakness is that some racing signals, especially full telemetry and older pit-stop details, are incomplete in public data. Overall, the project is a solid V0 for the assignment, with neural-network and 3D visualization work best positioned as optional extensions.

## References

Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32.

Cappello, C., & Hoegh, A. (2026). A state-space approach to modeling tire degradation in Formula 1 racing. Journal of Quantitative Analysis in Sports.

FastF1. (2026). FastF1 documentation. https://docs.fastf1.dev/

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. Annals of Statistics, 29(5), 1189-1232.

Jafri, A. (2024). Predicting Formula 1 race outcomes: A machine learning approach. https://aliabdullahjafri.com/

Jolpica. (2026). Jolpica F1 API. https://github.com/jolpica/jolpica-f1

Nigro, V. (2020). Formula 1 race predictor: A machine learning approach. Towards Data Science.

Noe, C., & Patel, N. (2024). Utilizing telemetry data for machine learning-driven lap prediction. University of Rochester.

Open-Meteo. (2026). Historical Weather API. https://open-meteo.com/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.

Zhao, Y. (2024). Deep neural network-based lap time forecasting of Formula 1 racing. Applied and Computational Engineering.
