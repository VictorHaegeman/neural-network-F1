from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from train_model import TARGET, build_features


DATA_PATH = Path("data/final/f1_top10_model_dataset.csv")
MODEL_PATH = Path("outputs/models/top10_neural_network_mlp.joblib")
HTML_OUTPUT_PATH = Path("outputs/figures/neural_network_embedding_3d.html")
PNG_OUTPUT_PATH = Path("outputs/figures/neural_network_embedding_3d.png")
CSV_OUTPUT_PATH = Path("outputs/neural_network_embedding_3d.csv")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive 3D visualization of the neural network hidden representation."
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output", type=Path, default=HTML_OUTPUT_PATH)
    parser.add_argument("--png-output", type=Path, default=PNG_OUTPUT_PATH)
    parser.add_argument("--csv-output", type=Path, default=CSV_OUTPUT_PATH)
    parser.add_argument("--max-rows", type=int, default=3500)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument(
        "--cdn",
        action="store_true",
        help="Write a smaller HTML file that loads Plotly from CDN instead of embedding it.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated interactive HTML visualization in the default browser.",
    )
    parser.add_argument(
        "--color-by",
        choices=["cluster", "target", "probability", "season"],
        default="cluster",
    )
    return parser.parse_args()


def save_static_3d_plot(points: pd.DataFrame, color_column: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#F5F2E8")
    ax.set_facecolor("#FFFFFF")

    if color_column in {"cluster", "target"}:
        categories = sorted(points[color_column].astype(str).unique())
        cmap = plt.get_cmap("tab10")
        for index, category in enumerate(categories):
            mask = points[color_column].astype(str) == category
            ax.scatter(
                points.loc[mask, "x"],
                points.loc[mask, "y"],
                points.loc[mask, "z"],
                s=12,
                alpha=0.76,
                color=cmap(index % 10),
                label=f"{color_column} {category}",
                depthshade=True,
            )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 0.96), frameon=False, fontsize=9)
    else:
        scatter = ax.scatter(
            points["x"],
            points["y"],
            points["z"],
            c=points[color_column],
            cmap="viridis",
            s=12,
            alpha=0.76,
            depthshade=True,
        )
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.1)
        cbar.set_label(color_column.replace("_", " ").title())

    ax.set_title(
        "Neural Network Hidden-Space Embedding\n3D PCA projection of the MLP internal representation",
        fontsize=18,
        fontweight="bold",
        color="#123C43",
        pad=18,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    ax.view_init(elev=22, azim=135)
    ax.grid(True, alpha=0.35)
    fig.text(
        0.03,
        0.035,
        "Each point is a driver-race row. Colors show the selected cluster/target/probability/season view.",
        fontsize=10,
        color="#333333",
    )
    plt.tight_layout(rect=(0, 0.05, 0.92, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_pipeline(path: Path) -> Pipeline:
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected a sklearn Pipeline, got {type(model).__name__}")
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("The model pipeline must contain 'preprocessor' and 'classifier' steps.")
    return model


def dense_array(values: object) -> np.ndarray:
    if sparse.issparse(values):
        return values.toarray()
    return np.asarray(values)


def activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(values, 0)
    if name == "tanh":
        return np.tanh(values)
    if name == "logistic":
        values = np.clip(values, -500, 500)
        return 1.0 / (1.0 + np.exp(-values))
    if name == "identity":
        return values
    raise ValueError(f"Unsupported MLP activation: {name}")


def hidden_embedding(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    if not hasattr(classifier, "coefs_") or not hasattr(classifier, "intercepts_"):
        raise TypeError("The classifier does not expose MLP weights. Train/use neural_network_mlp first.")

    transformed = dense_array(preprocessor.transform(X))
    hidden = transformed
    for weights, bias in zip(classifier.coefs_[:-1], classifier.intercepts_[:-1]):
        hidden = activation(hidden @ weights + bias, classifier.activation)
    return hidden


def sample_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=RANDOM_STATE).sort_values(["season", "round", "driver_id"])


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    sampled = sample_rows(df, args.max_rows).reset_index(drop=True)
    pipeline = load_pipeline(args.model)
    X = build_features(sampled)
    embedding = hidden_embedding(pipeline, X)

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    coords = pca.fit_transform(embedding)
    clusters = KMeans(n_clusters=args.clusters, random_state=RANDOM_STATE, n_init=10).fit_predict(embedding)
    probabilities = pipeline.predict_proba(X)[:, 1]

    points = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "z": coords[:, 2],
            "cluster": clusters.astype(str),
            "top10_probability": probabilities,
            "target": sampled[TARGET].astype(int).astype(str),
            "season": sampled["season"].astype(int),
            "round": sampled["round"].astype(int),
            "race_id": sampled["race_id"].astype(str),
            "grand_prix": sampled.get("grand_prix", "Unknown Grand Prix"),
            "driver_id": sampled["driver_id"].astype(str),
            "driver_code": sampled.get("driver_code", sampled["driver_id"]).astype(str),
            "constructor_name": sampled.get("constructor_name", "Unknown").astype(str),
            "grid": sampled.get("grid", 0),
            "final_position": sampled.get("final_position", 0),
        }
    )

    color_column = {
        "cluster": "cluster",
        "target": "target",
        "probability": "top10_probability",
        "season": "season",
    }[args.color_by]

    fig = px.scatter_3d(
        points,
        x="x",
        y="y",
        z="z",
        color=color_column,
        hover_data=[
            "race_id",
            "grand_prix",
            "driver_code",
            "constructor_name",
            "grid",
            "final_position",
            "top10_probability",
            "cluster",
        ],
        title="Neural network hidden-space embedding",
        opacity=0.82,
        height=760,
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(
        title={
            "text": "Neural network hidden-space embedding - drag to rotate, scroll to zoom",
            "x": 0.5,
            "xanchor": "center",
        },
        scene={
            "xaxis_title": "PCA 1",
            "yaxis_title": "PCA 2",
            "zaxis_title": "PCA 3",
        },
        legend_title_text=args.color_by,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        args.output,
        include_plotlyjs="cdn" if args.cdn else True,
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "neural_network_embedding_3d",
                "height": 900,
                "width": 1400,
                "scale": 2,
            },
        },
    )
    points.to_csv(args.csv_output, index=False)
    save_static_3d_plot(points, color_column, args.png_output)

    explained = ", ".join(f"{value:.1%}" for value in pca.explained_variance_ratio_)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.png_output}")
    print(f"Wrote {args.csv_output}")
    print(f"PCA explained variance: {explained}")
    print(f"Rows visualized: {len(points)}")
    print("Interaction: drag to rotate, scroll to zoom, hover points for race/driver details.")

    if args.open:
        webbrowser.open(args.output.resolve().as_uri())


if __name__ == "__main__":
    main()
