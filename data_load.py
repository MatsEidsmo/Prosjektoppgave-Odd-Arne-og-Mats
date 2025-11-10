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
                             sep = '\s+', header=None, names = ["index", "R", "G", 'B', "Label"])

    network_table["Network"] = network_table["Label"].str.split("_").str[2]
    network_table["Hemisphere"] = network_table["Label"].str.split("_").str[1]
    network_table.drop(columns=["R", "G", "B", "Label"], inplace=True)

    return network_table
