"""PCA of the sensor features.

The scaler and PCA are fitted on the source batches (1-5) only. The same
transformation is then applied to the future batches (6-10). The result is
a descriptive visualisation, not a classification result.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.knn import FEATURES, TARGET
from src.split import make_temporal_splits

GAS_NAMES = {1: "Ethanol", 2: "Ethylene"}
GAS_COLORS = {1: "tab:blue", 2: "tab:red"}


def fit_pca(source: pd.DataFrame, n_components: int = 10):
    """Fit scaler and PCA using source observations only."""
    scaler = StandardScaler().fit(source[FEATURES])
    pca = PCA(n_components=n_components).fit(scaler.transform(source[FEATURES]))
    return scaler, pca


def project(df: pd.DataFrame, scaler, pca) -> pd.DataFrame:
    """Apply the already fitted transformation and keep the first two PCs."""
    scores = pca.transform(scaler.transform(df[FEATURES]))

    projected = pd.DataFrame(
        scores[:, :2], columns=["PC1", "PC2"], index=df.index
    )
    projected[TARGET] = df[TARGET].to_numpy()
    projected["batch_id"] = df["batch_id"].to_numpy()
    return projected


def explained_variance_table(pca) -> pd.DataFrame:
    ratios = pca.explained_variance_ratio_
    return pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(ratios))],
        "explained_variance_ratio": ratios,
        "cumulative": ratios.cumsum(),
    })


def plot_by_gas(source_scores, future_scores, ratios, output_path: Path):
    """Source and future batches side by side, coloured by gas."""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained"
    )

    panels = (
        (axes[0], source_scores, "Source batches (1-5), used to fit PCA"),
        (axes[1], future_scores, "Future batches (6-10), projected"),
    )
    for ax, scores, title in panels:
        for gas in (1, 2):
            part = scores[scores[TARGET] == gas]
            ax.scatter(
                part["PC1"], part["PC2"], s=8, alpha=0.4,
                color=GAS_COLORS[gas], label=GAS_NAMES[gas],
            )
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}% variance)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"PC2 ({ratios[1] * 100:.1f}% variance)")
    axes[0].legend(markerscale=2)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_by_batch(all_scores, ratios, output_path: Path):
    """One panel per gas, points coloured by batch."""
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.5), sharex=True, sharey=True, layout="constrained"
    )

    for ax, gas in zip(axes, (1, 2)):
        part = all_scores[all_scores[TARGET] == gas]
        scatter = ax.scatter(
            part["PC1"], part["PC2"], c=part["batch_id"],
            cmap="viridis", vmin=1, vmax=10, s=8, alpha=0.6,
        )
        ax.set_title(GAS_NAMES[gas])
        ax.set_xlabel(f"PC1 ({ratios[0] * 100:.1f}% variance)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(f"PC2 ({ratios[1] * 100:.1f}% variance)")
    fig.colorbar(scatter, ax=axes, label="Batch ID")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[1]

    config = yaml.safe_load(
        (project_root / "config.yaml").read_text(encoding="utf-8")
    )

    df = pd.read_csv(project_root / "data" / "processed" / "ethanol_ethylene.csv")
    parts = make_temporal_splits(df, config)

    source = parts["source_train"]
    future = parts["future_test"]

    scaler, pca = fit_pca(source)
    source_scores = project(source, scaler, pca)
    future_scores = project(future, scaler, pca)
    all_scores = pd.concat([source_scores, future_scores])

    ratios = pca.explained_variance_ratio_

    figures_dir = project_root / "results" / "figures"
    tables_dir = project_root / "results" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    table = explained_variance_table(pca)
    table.to_csv(tables_dir / "pca_explained_variance.csv", index=False)

    plot_by_gas(source_scores, future_scores, ratios, figures_dir / "pca_by_gas.png")
    plot_by_batch(all_scores, ratios, figures_dir / "pca_by_batch.png")

    print(table.to_string(index=False))
