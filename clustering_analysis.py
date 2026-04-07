import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

# =========================
# 1. Load balanced dataset
# =========================
df = pd.read_csv("data/research_papers_balanced.csv")

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# =========================
# 2. TF-IDF feature engineering
# =========================
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(df["clean_text"])
true_labels = df["label"]

print("\nTF-IDF shape:", X.shape)

# =========================
# 3. K-Means Clustering
# =========================
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)

sil_kmeans = silhouette_score(X, kmeans_labels)
ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)

print("\nK-Means Results:")
print("Silhouette Score:", sil_kmeans)
print("Adjusted Rand Index:", ari_kmeans)

# =========================
# 4. Hierarchical Clustering
# =========================
# Hierarchical clustering works better on reduced dense features
pca_for_hier = PCA(n_components=100, random_state=42)
X_reduced_hier = pca_for_hier.fit_transform(X.toarray())

hier = AgglomerativeClustering(n_clusters=5)
hier_labels = hier.fit_predict(X_reduced_hier)

sil_hier = silhouette_score(X_reduced_hier, hier_labels)
ari_hier = adjusted_rand_score(true_labels, hier_labels)

print("\nHierarchical Clustering Results:")
print("Silhouette Score:", sil_hier)
print("Adjusted Rand Index:", ari_hier)

# =========================
# 5. Save clustering comparison
# =========================
results_df = pd.DataFrame({
    "Model": ["K-Means", "Hierarchical"],
    "Silhouette Score": [sil_kmeans, sil_hier],
    "Adjusted Rand Index": [ari_kmeans, ari_hier]
})

print("\nClustering comparison:")
print(results_df)

results_df.to_csv("results/clustering_results.csv", index=False)
print("\nSaved: results/clustering_results.csv")

# =========================
# 6. PCA visualization for K-Means
# =========================
pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X.toarray())

plt.figure(figsize=(8, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, s=12)
plt.title("K-Means Clusters (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("results/kmeans_pca_projection.png")
plt.close()
print("Saved: results/kmeans_pca_projection.png")

# =========================
# 7. Bar chart comparison
# =========================
plt.figure(figsize=(8, 5))
x = range(len(results_df))
plt.bar(x, results_df["Silhouette Score"])
plt.xticks(x, results_df["Model"])
plt.ylabel("Silhouette Score")
plt.title("Clustering Silhouette Score Comparison")
plt.tight_layout()
plt.savefig("results/clustering_silhouette_comparison.png")
plt.close()
print("Saved: results/clustering_silhouette_comparison.png")

# =========================
# 8. Top terms per K-Means cluster
# =========================
terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\nTop 10 terms per K-Means cluster:")
cluster_terms = {}

for i in range(5):
    top_terms = [terms[ind] for ind in order_centroids[i, :10]]
    cluster_terms[i] = top_terms
    print(f"Cluster {i}: {', '.join(top_terms)}")

cluster_terms_df = pd.DataFrame([
    {"Cluster": cluster_id, "Top Terms": ", ".join(terms_list)}
    for cluster_id, terms_list in cluster_terms.items()
])

cluster_terms_df.to_csv("results/kmeans_top_terms.csv", index=False)
print("\nSaved: results/kmeans_top_terms.csv")

# =========================
# 9. Save cluster assignments
# =========================
df["kmeans_cluster"] = kmeans_labels
df["hier_cluster"] = hier_labels
df.to_csv("results/papers_with_clusters.csv", index=False)
print("Saved: results/papers_with_clusters.csv")