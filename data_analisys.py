from imports import *
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


    linked = linkage(features_df, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Ward Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()