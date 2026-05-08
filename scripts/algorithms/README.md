# Algorithms Folder

This folder contains only the model definitions and hyperparameters.

## Main Top-10 Classification

Open `classification.py` to see the algorithms used for the assignment's main
binary classification task:

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `neural_network_mlp`

The current assignment holdout champion is `random_forest`, with `extra_trees`
kept as the strongest rolling-backtest comparison.

## Finish-Position Ranking

Open `regression.py` to see the models that predict an ordered finishing
position for each driver:

- `hist_gradient_boosting_regressor`
- `random_forest_regressor`
- `neural_network_mlp_regressor`

The best current finish-position model is the histogram gradient boosting
regressor, with the neural-network MLP regressor kept as an extension.

## Neural Network Tuning

Open `neural_network.py` to see the dedicated MLP architectures tested by
`scripts/tune_neural_network.py`.

The most useful visual outputs are:

- `outputs/figures/neural_network_tuning.png`
- `outputs/figures/neural_network_embedding_3d.html`
- `outputs/figures/neural_network_embedding_3d.png`
- `outputs/figures/position_model_comparison.png`

The trained model files are generated in `outputs/models/`, but they are not
included in the submission ZIP because binary model artifacts are heavy and can
be recreated from the scripts.
