from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_clustering(df, cfg):
    cluster_cols = cfg["clustering"]["cluster_columns"]
    k = cfg["clustering"]["k"]

    X = df[cluster_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, df["Cluster"], sample_size=3000, random_state=42)
    print(f"Silhouette score for k={k}: {sil_score:.4f}")
    print(f"(Note: k={k} does not necessarily maximize silhouette — see the notebook conclusions for the business justification)")

    return df
