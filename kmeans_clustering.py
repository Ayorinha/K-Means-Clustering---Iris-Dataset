"""K-Means clustering demonstration using the Iris dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score


N_CLUSTERS = 3
RANDOM_STATE = 42
OUTPUT_DIR = Path("outputs")


def main() -> None:
    """Run the clustering experiment and display the result."""
    iris = load_iris()
    features = iris.data

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = model.fit_predict(features)
    centroids = model.cluster_centers_

    silhouette_avg = silhouette_score(features, labels)
    print(f"Índice de Silhueta: {silhouette_avg:.3f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        features[:, 0],
        features[:, 1],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.7,
        edgecolors="k",
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        s=200,
        marker="X",
        label="Centros",
    )

    ax.set_title("K-Means Clustering — Iris Dataset")
    ax.set_xlabel("Comprimento da Sépala")
    ax.set_ylabel("Largura da Sépala")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="Cluster")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "iris-kmeans.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
