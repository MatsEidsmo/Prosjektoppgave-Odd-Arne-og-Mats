from my_imports import *
import matplotlib.pyplot as plt
from data_load import load_fc_matrices, plot_fc_matrix, load_network_table


            
def perform_z_normalization(features_df: pd.DataFrame, group: int = "all"):
    z_normalized_df = features_df.apply(zscore)
    if group != "all":
        z_normalized_df = features_df[features_df['group'] == group]
    return z_normalized_df

def perform_PCA(features_df: pd.DataFrame, n_components=10, group: int = "all"):
    if group != "all":
        features_df = features_df[features_df['group'] == group]
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_df)

    features_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])
    
    explained = pca.explained_variance_ratio_
    print("Explained variance per PC:", explained)
    print("Cumulative explained variance:", explained.cumsum())
    return features_df

def perform_ward_hierarchical_linkage(features_df: pd.DataFrame):

    return linkage(features_df, method='ward')

def plot_dendogram(Z: np.array):
    plt.figure(figsize=(10, 7))
    dendrogram(Z,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Ward Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()

def find_clusters(features_df: pd.DataFrame, linkage_matrix: np.array = None, group: str = "any"):
    if group != "any":
        features_df = features_df[features_df['group'] == group]


    scores = {}
    for k in range(2, 12):
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(features_df, cluster_labels)
        scores[k] = score
        print(f"For n_clusters = {k}, the average silhouette_score is : {score}")
    
    best_score_k = max(scores, key=scores.get)
    print(f"Best number of clusters by silhouette score: {best_score_k} with score {scores[best_score_k]}")

    plt.figure()
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.show()




def PCA_subset_scores(pca_df: pd.DataFrame, components: list, method="hierarchical", k_range=(2, 10)):
    
    subset = pca_df[components]
    silhouette_scores = []
    labels_dict = {}
    ks = range(k_range[0], k_range[1] + 1)

    for k in ks:
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=42)
            labels = model.fit_predict(subset)
        elif method == "hierarchical":
            Z = linkage(subset, method="ward")
            labels = fcluster(Z, k, criterion="maxclust")
        else:
            raise ValueError("Unsupported method")

        score = silhouette_score(subset, labels)
        silhouette_scores.append(score)
        print(f"k={k}, silhouette_score={score:.4f}")
        labels_dict[k] = labels

    # Plot silhouette scores
    # plt.figure(figsize=(8, 5))
    # plt.plot(ks, silhouette_scores, marker='o')
    # plt.title(f"Silhouette Scores for {method.capitalize()} Clustering, PCA Components: {', '.join(components)}")
    # plt.xlabel("Number of Clusters (k)")
    # plt.ylabel("Silhouette Score")
    # plt.grid(True)
    # plt.show()

    
    # Best k
    best_k = ks[np.argmax(silhouette_scores)]
    best_labels = labels_dict[best_k]
    print(f"Best k: {best_k} with silhouette score {max(silhouette_scores):.4f}")

    return best_k, best_labels, silhouette_scores





def k_means_clustering(features_df: pd.DataFrame, cluster_range=(2, 10), group: str = "any"):
    if group != "any":
        features_df = features_df[features_df['group'] == group]

    # Keep only numeric columns
    numeric_df = features_df.select_dtypes(include=[np.number])

    silhouette_scores = {}
    for n_clusters in range(cluster_range[0], cluster_range[1] + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(numeric_df)
        score = silhouette_score(numeric_df, cluster_labels)
        silhouette_scores[n_clusters] = score
        print(f"For n_clusters = {n_clusters}, silhouette_score = {score:.4f}")

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()

    # Best cluster count
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Best number of clusters: {best_k} with silhouette score {silhouette_scores[best_k]:.4f}")

    return best_k, silhouette_scores



def plot_clustered_heatmap(features_df: pd.DataFrame, linkage_matrix: np.array):
    sns.clustermap(features_df, row_linkage=linkage_matrix, col_cluster=False, cmap='vlag', standard_scale=1)
    plt.title('Clustered Heatmap')
    plt.show()



def plot_clusters_scatter_pc2_pc3(pca_df: pd.DataFrame, labels: np.ndarray, title="Clusters in PCA Space (PC2 vs PC3)"):
    """
    Plot a scatter plot of PC2 vs PC3 colored by cluster labels.

    Parameters:
    -----------
    pca_df : pd.DataFrame
        PCA-transformed data (must include PC2 and PC3).
    labels : np.ndarray
        Cluster labels for each subject.
    title : str
        Title for the plot.
    """
    if "PC2" not in pca_df.columns or "PC3" not in pca_df.columns:
        raise ValueError("pca_df must contain PC2 and PC3 columns for plotting.")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(pca_df["PC2"], pca_df["PC3"], c=labels, cmap="viridis", s=50)
    plt.title(title)
    plt.xlabel("PC2")
    plt.ylabel("PC3")
    plt.colorbar(scatter, label="Cluster")
    plt.grid(True)
    plt.show()


def PCA_loadings(z_scored_features: pd.DataFrame, n_components=10):
    pca = PCA(n_components=n_components)
    pca.fit(z_scored_features)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(loadings, index=z_scored_features.columns, columns=[f'PC{i+1}' for i in range(n_components)])
    return loading_df, pca.explained_variance_ratio_

def plot_loadings(loading_df: pd.DataFrame, pcs: list = ["PC1", "PC2"], top_n: int = 10):
    network_pairs = [feat.split("_mean_conn")[0] for feat in loading_df.index]
    loading_df["network_pair"] = network_pairs
    agg_df = loading_df.groupby("network_pair")[pcs].mean()

    plt.figure(figsize=(10, len(agg_df) * 0.4))
    sns.heatmap(agg_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("PCA Loadings by Network Pair")
    plt.xlabel("Principal Component")
    plt.ylabel("Network Pair")
    plt.tight_layout()
    plt.show()