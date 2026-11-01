
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Step 1: Load Iris Dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y_true = iris.target

print("Dataset Shape:", X.shape)
print("Features:", list(X.columns))


# Step 2: Standardize the Data
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Step 3: Apply Elbow Method
wcss = []

K_range = range(1, 11)

for k in K_range:

    km = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        random_state=42
    )

    km.fit(X_scaled)

    wcss.append(km.inertia_)

    print(f'K={k}: WCSS={km.inertia_:.2f}')


# Step 4: Plot Elbow Graph
plt.figure(figsize=(7, 5))

plt.plot(
    list(K_range),
    wcss,
    'o-'
)

plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')

plt.title('Elbow Method - Iris Dataset')

plt.grid(True)

plt.tight_layout()

plt.savefig('iris_elbow.png', dpi=120)

plt.close()


# Step 5: Select Optimal Number of Clusters
optimal_k = 3

final_km = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    random_state=42
)


# Step 6: Perform K-Means Clustering
cluster_labels = final_km.fit_predict(X_scaled)


# Step 7: Visualize Clusters
plt.figure(figsize=(7, 6))

plt.scatter(
    X['petal length (cm)'],
    X['petal width (cm)'],
    c=cluster_labels,
    cmap='viridis',
    s=35,
    edgecolors='k'
)

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.title(
    f'K-Means Clusters on Iris (K={optimal_k})'
)

plt.tight_layout()

plt.savefig('iris_kmeans_clusters.png', dpi=120)

plt.close()


# Step 8: Compare Clusters with True Species
comparison = pd.crosstab(
    cluster_labels,
    y_true,
    rownames=['Cluster'],
    colnames=['True Species']
)

print(
    '\nCluster vs. True Species Crosstab:\n',
    comparison
)


# Step 9: Project Completion
print("\nK-Means Clustering Project Completed!")

print("Generated files:")
print("1. iris_elbow.png")
print("2. iris_kmeans_clusters.png")
