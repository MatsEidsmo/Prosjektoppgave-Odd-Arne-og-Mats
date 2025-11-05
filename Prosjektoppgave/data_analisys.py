from imports import *
from data_load import load_fc_matrices, plot_fc_matrix, load_network_table

network_table = load_network_table("C:\\Mats og Odd Arne\\Prosjektoppgave\\Schaefer2018_400Parcels_7Networks_order.lut")
network_labels = network_table["Network"].values


def extract_subject_connectivity_features(fc_matrix: np.array, network_table: pd.Dataframe):
    
    networks = network_table["Network"].values
    unique_networks = np.unique(networks)

    features = {}
    for net_i in unique_networks:
        
        idx_i = np.where(networks == net_i)[0]
        
        for net_j in unique_networks:
            
            idx_j = np.where(networks == net_j)[0]
            sub_matrix = fc_matrix[np.ix_(idx_i, idx_j)]
           
            mean_conn = np.mean(sub_matrix)
            feature_name = f"{net_i}_{net_j}_mean_conn"
            features[feature_name] = mean_conn

    
    return features

def create_feature_dataframe():
    fc_matrices = load_fc_matrices()

    all_features = [extract_subject_connectivity_features(fc_matrix, network_table) for fc_matrix in fc_matrices]
    features_df = pd.DataFrame(all_features)
    return features_df
            
