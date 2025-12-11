from my_imports import *
import data_load as dl
import data_processing as dp
import data_analisys as da
import res_scores as rs

WFH = False

def main():
    
    
    npz_data = np.load("subject_features.npz", allow_pickle=True)

    values = npz_data['data']
    columns = npz_data['columns']
    index = npz_data['index']

    subject_features = dp.remove_duplicate_pairs(pd.DataFrame(values, columns=columns, index=index))


    #res_scores = rs.extract_res_scores_from_csv(r"C:\Users\matsei\Documents\ISC_data\Beh.csv")
    
    
    #dl.plot_fc_matrix(fc_matrices[0])
    
    
    z_scores = da.perform_z_normalization(subject_features)
    #print(f"Z-scores:\n{z_scores}")
    #da.k_means_clustering(z_scores, (2, 10))

    pc_df = da.perform_PCA(z_scores, n_components=20)
    
    #Cluster on PC2 only
    
    _, labels, _ = da.PCA_subset_scores(pc_df, ['PC2', 'PC3'], method="hierarchical", k_range=(2, 2))
    da.plot_clusters_scatter_pc2_pc3(pc_df, labels)

    loadings, _ = da.PCA_loadings(z_scores)
    #da.plot_loadings(loadings)
    
    
    #linkage_matrix = da.perform_ward_hierarchical_linkage(z_scores)

    
    #da.plot_dendogram(linkage_matrix)
    #da.find_clusters(z_scores.values, linkage_matrix)
    #print("Extracted Features:\n", subject_features)
    #da.plot_clustered_heatmap(z_scores, linkage_matrix)
    ###PLOTTING A SINGLE fc matrix

main()