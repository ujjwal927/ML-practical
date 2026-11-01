
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Step 1: Load Wine Dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("Features:", list(X.columns))
print("Dataset shape:", X.shape)


# Step 2: Standardize the Dataset
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Step 3: Apply PCA on All Components
pca_full = PCA()

pca_full.fit(X_scaled)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)

print(
    "Explained variance ratio (all components):",
    np.round(pca_full.explained_variance_ratio_, 3)
)

print(
    "Cumulative variance with 2 components: {:.2f}%".format(
        cum_var[1] * 100
    )
)


# Step 4: Reduce Dimensions to 2 Components
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


# Step 5: Plot PCA Visualization
plt.figure(figsize=(7, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap='viridis',
    s=35,
    edgecolors='k'
)

plt.xlabel(
    f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)'
)

plt.ylabel(
    f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)'
)

plt.title('Wine Data Projected onto 2 Principal Components')

plt.legend(
    *scatter.legend_elements(),
    title='Wine Category'
)

plt.tight_layout()

plt.savefig('pca_wine.png', dpi=120)

plt.close()


# Step 6: Plot Cumulative Explained Variance
plt.figure(figsize=(7, 5))

plt.plot(
    range(1, len(cum_var) + 1),
    cum_var,
    'o-'
)

plt.xlabel('Number of Principal Components')

plt.ylabel('Cumulative Explained Variance')

plt.title('PCA Scree Plot - Wine Dataset')

plt.grid(True)

plt.tight_layout()

plt.savefig('pca_scree_wine.png', dpi=120)

plt.close()


# Step 7: Display Completion Message
print("\nPCA Project Completed Successfully!")

print("Plots saved as:")
print("1. pca_wine.png")
print("2. pca_scree_wine.png")
