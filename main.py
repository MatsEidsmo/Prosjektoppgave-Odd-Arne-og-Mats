from imports import *
import data_load as dl
import data_processing as dp
import data_analisys as da

def main():
    rand_df = dp.generate_synthetic_feature_dataframe(n_subjects=20, n_rois=100, n_networks=5, random_state=42)
    z_scores = da.perform_z_normalization(rand_df)
    pc_df = da.perform_PCA(rand_df, n_components=20)
    linkage_matrix = da.perform_ward_hierarchical_linkage(pc_df)
   # da.plot_dendogram(linkage_matrix)
    da.find_clusters(z_scores.values, linkage_matrix)
    #dl.plot_fc_matrix(fc_matrix)
    # subject_features = dp.extract_subject_connectivity_features(fc_matrix, dp.get_network_table())
    # print("Extracted Features:\n", subject_features)

main()