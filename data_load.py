from imports import *



def load_fc_matrices():
    N_FC_Matrices_m2 = 216
    ## Load data from matlab files
    subjects = []
    for i in range(N_FC_Matrices_m2):
        filepath = f"C:\\Mats og Odd Arne\\Prosjektoppgave\\sch407\\YA\\zFCmat\\sub-11{i:03d}_task-video_run-2 __zFCmat.mat"

        try:
            fc_mat_m2 = loadmat(filepath)
            
            fc_array = np.array(fc_mat_m2['zfcmatrix'])
        except Exception as e:
            continue
        
        subjects.append(fc_array)
        # Process fc_matrix as needed
    print(f"Loaded {len(subjects)} FC matrices.")

    print(f"Matrix shape:\n {subjects[0].shape}")

    return subjects

def plot_fc_matrix(fc_matrix):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(fc_matrix, cmap='RdBu_r', center=0, square=True, cbar_kws={'label': 'Correlation'})
    plt.title('Functional Connectivity Matrix')
    plt.xlabel('Regions')
    plt.ylabel('Regions')
    plt.show()

def load_network_table(filepath):
    network_table = pd.read_csv(filepath,
                             sep = r'\s+', header=None, names = ["index", "R", "G", 'B', "Label"])

    network_table["Network"] = network_table["Label"].str.split("_").str[2]
    network_table["Hemisphere"] = network_table["Label"].str.split("_").str[1]
    network_table.drop(columns=["R", "G", "B", "Label"], inplace=True)

    return network_table


def generate_random_fc_matrix(n_rois=400, n_networks=7, random_state=None):
    """
    Generate a synthetic, Fisher z-transformed functional connectivity matrix.
    Mimics realistic block structure with within-network coherence.
    
    Parameters
    ----------
    n_rois : int
        Number of ROIs (e.g. 400 for Schaefer400).
    n_networks : int
        Number of networks to simulate (e.g. 7 for Yeo-7).
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    fc_matrix : np.ndarray
        Symmetric (n_rois x n_rois) matrix with diagonal = 1.
    """

    rng = np.random.default_rng(random_state)

    # Randomly assign each ROI to a "network"
    network_labels = np.repeat(np.arange(n_networks), n_rois // n_networks)
    remainder = n_rois - len(network_labels)
    if remainder > 0:
        network_labels = np.concatenate([network_labels, rng.integers(0, n_networks, remainder)])

    # Initialize empty matrix
    fc_matrix = np.zeros((n_rois, n_rois))

    # Define mean Fisher z-values for within- and between-network connections
    mean_within = 0.8   # higher z-values (stronger connectivity)
    mean_between = 0.1  # weaker or slightly positive connectivity
    sd = 0.3            # variability

    # Fill the matrix
    for i in range(n_rois):
        for j in range(i, n_rois):
            if network_labels[i] == network_labels[j]:
                z_val = rng.normal(mean_within, sd)
            else:
                z_val = rng.normal(mean_between, sd)
            fc_matrix[i, j] = z_val
            fc_matrix[j, i] = z_val  # symmetry

    # Set diagonal to 1 (self-correlation)
    np.fill_diagonal(fc_matrix, 1.0)

    return fc_matrix