# Algorithm Locations

The project now separates model definitions from the rest of the data pipeline.

## Clean Algorithm Folder

Open `scripts/algorithms/` when you want to inspect only the algorithms:

| File | Purpose |
|---|---|
| `scripts/algorithms/classification.py` | Main top-10 classification models |
| `scripts/algorithms/regression.py` | Finish-position ranking models |
| `scripts/algorithms/neural_network.py` | Dedicated MLP neural-network tuning configs |
| `scripts/algorithms/README.md` | Short human-readable summary |

## Main Classification Algorithms

These predict `top10_finish`, the assignment's main binary target:

- `logistic_regression`
- `random_forest`
- `extra_trees`
- `hist_gradient_boosting`
- `neural_network_mlp`

The current assignment champion is `hist_gradient_boosting`.

## Neural Network Files

The neural-network code is split into three parts:

| File/output | What it shows |
|---|---|
| `scripts/algorithms/neural_network.py` | MLP architectures and hyperparameters |
| `scripts/tune_neural_network.py` | Training/tuning loop for the MLP classifier |
| `scripts/visualize_neural_network_3d.py` | 3D hidden-space visualisation generation |
| `outputs/figures/neural_network_tuning.png` | PNG comparison of MLP configs |
| `outputs/figures/neural_network_embedding_3d.html` | Interactive 3D cluster view |

To open the live 3D view in a browser:

```powershell
python scripts/open_neural_network_3d.py
```

## Trained Model Artifacts

Generated model binaries are written to `outputs/models/`:

- `outputs/models/top10_classifier.joblib`
- `outputs/models/top10_neural_network_mlp.joblib`
- `outputs/models/finish_position_regressor.joblib`
- `outputs/models/finish_position_neural_network_mlp.joblib`

These files are intentionally excluded from the submission ZIP because they are
large generated artifacts and can be recreated from the scripts.
