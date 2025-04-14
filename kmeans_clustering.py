# ============================================================ #
# ================== < K-Means Clustering > =================== #
# ============================================================ #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ============================================================ #

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data  # Atributos (features)

# Realizar K-Means com 3 clusters (para Iris, que tem 3 tipos de flores)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Obter os centros dos clusters e os rótulos atribuídos
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Avaliar a qualidade do clustering com o Índice de Silhueta
silhouette_avg = silhouette_score(X, labels)
print(f"Índice de Silhueta: {silhouette_avg:.2f}")

# Plotar os clusters
plt.figure(figsize=(8, 6))

# Plotando os pontos coloridos de acordo com o cluster
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6, edgecolors='k')

# Plotando os centros dos clusters
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centros')

plt.title("K-Means Clustering - Iris Dataset")
plt.xlabel("Comprimento da Sépala")
plt.ylabel("Largura da Sépala")
plt.legend()
plt.show()

# ============================================================ #
