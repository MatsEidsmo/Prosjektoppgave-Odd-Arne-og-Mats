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

def plot_clustered_heatmap(features_df: pd.DataFrame, linkage_matrix: np.array):
    sns.clustermap(features_df, row_linkage=linkage_matrix, col_cluster=False, cmap='vlag', standard_scale=1)
    plt.title('Clustered Heatmap')
    plt.show()


def plot_PCA(features_df: pd.DataFrame):
    clustering = AgglomerativeClustering(n_clusters=2, linkage="ward")
    labels = clustering.fit_predict(features_df)
    plt.figure(figsize=(8,6))
    plt.scatter(features_df.iloc[:,0], features_df.iloc[:,1], c=labels, cmap="tab10", s=50)
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA Projection of fMRI Connectivity Features")
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