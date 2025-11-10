from imports import *
import data_load as dl
import data_processing as dp
import data_analisys as da

def main():
    fc_matrix = dl.generate_random_fc_matrix(454, 7, random_state=42)
    #dl.plot_fc_matrix(fc_matrix)
    subject_features = dp.extract_subject_connectivity_features(fc_matrix, dp.get_network_table())
    print("Extracted Features:\n", subject_features)

main()