from my_imports import *
import matplotlib.pyplot as plt
import os
SubcorticalROI = False



def load_fc_matrices() -> pd.DataFrame:
    N_FC_Matrices_m2 = 400

    # Base directories (adjust if needed)
    base_ya = r"C:\Users\matsei\Documents\Mats og Odd Arne\Prosjektoppgave\sch407\YA\zFCmat"
    base_oa = r"C:\Users\matsei\Documents\Mats og Odd Arne\Prosjektoppgave\sch407\OA\zFCmat"

    rows = []
    missing = []

    for i in range(N_FC_Matrices_m2):
        # Build two subjects per i: sub-11XXX and sub-12XXX
        for cohort_code, base_dir in [("11", base_ya), ("12", base_oa)]:
            Subject = f"sub-{cohort_code}{i+1:03d}"
            filename = f"{Subject}_task-video_run-2__zFCmat.mat"
            filepath = os.path.join(base_dir, filename)

            try:
                fc_mat_m2 = loadmat(filepath)
                fc_array = np.array(fc_mat_m2['zfcmatrix'], dtype=float)

                # Ensure diagonal 1.0 and optionally trim to 400x400
                np.fill_diagonal(fc_array, 1.0)
                if not SubcorticalROI:
                    fc_array = fc_array[:400, :400]

                rows.append({"Subject": Subject, "FC": fc_array})

            except Exception as e:
                missing.append((Subject, filepath, str(e)))
                continue

    df = pd.DataFrame(rows)
   
    return df


def plot_fc_matrix(fc_matrix):
    #fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=1.0, neginf=0.0)
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