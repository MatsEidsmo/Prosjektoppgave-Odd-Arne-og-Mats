from my_imports import *
import data_load as dl
import data_processing as dp
import data_analisys as da

WFH = False

def main():
    if WFH:
        fc_matrices = dp.generate_synthetic_feature_dataframe(n_subjects=20, n_rois=100, n_networks=5, random_state=42)

    else:
        #fc_matrices = dl.load_fc_matrices()
        subject_features = dp.create_feature_dataframe()
    np.savez("subject_features.npz", data=subject_features.values, columns=subject_features.columns, index=subject_features.index)
    
    #dl.plot_fc_matrix(fc_matrices[0])
    
    
    z_scores = da.perform_z_normalization(subject_features)
    print(f"Z-scores:\n{z_scores}")
    pc_df = da.perform_PCA(z_scores, n_components=20)
   
    loadings, _ = da.PCA_loadings(z_scores)
    da.plot_loadings(loadings)
    
    
    # linkage_matrix = da.perform_ward_hierarchical_linkage(z_scores)

    
    # #da.plot_dendogram(linkage_matrix)
    # da.find_clusters(z_scores.values, linkage_matrix)
    # #print("Extracted Features:\n", subject_features)
    # da.plot_clustered_heatmap(z_scores, linkage_matrix)
    # ###PLOTTING A SINGLE fc matrix

main()