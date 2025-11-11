from imports import *
import matplotlib.pyplot as plt
from data_load import load_fc_matrices, plot_fc_matrix, load_network_table


            
def perform_z_normalization(features_df: pd.DataFrame):
    z_normalized_df = features_df.apply(zscore)
    return z_normalized_df

def perform_PCA(features_df: pd.DataFrame, n_components=10):
    
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features_df)

    pc_df = pd.DataFrame(data=principal_components,
                         columns=[f'PC{i+1}' for i in range(n_components)])
    
    explained = pca.explained_variance_ratio_
    print("Explained variance per PC:", explained)
    print("Cumulative explained variance:", explained.cumsum())
    return pc_df

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

def find_clusters(norm_data: np.array, linkage_matrix: np.array = None):
    scores = {}
    for k in range(2, 12):
        cluster_labels = fcluster(linkage_matrix, k, criterion='maxclust')
        score = silhouette_score(norm_data, cluster_labels)
        scores[k] = score
        print(f"For n_clusters = {k}, the average silhouette_score is : {score}")
    
    best_score_k = max(scores, key=scores.get)
    print(f"Best number of clusters by silhouette score: {best_score_k} with score {scores[best_score_k]}")

    plt.figure()
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.show()


