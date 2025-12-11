from my_imports import *
from data_load import load_fc_matrices, plot_fc_matrix, load_network_table

WFH = False


def get_network_table():
    if WFH:
        network_table = load_network_table(r"C:\Users\matse\OneDrive - NTNU\Documents\Kyb\2025 Høst\Prosjektoppgave\Schaefer2018_400Parcels_7Networks_order.lut")
    else:
        network_table = load_network_table(r"Schaefer2018_400Parcels_7Networks_order.lut")
    
    return network_table
    



def extract_subject_connectivity_features(fc_matrix: np.array, network_table: pd.DataFrame, group: int = "all"):
    networks = network_table["Network"].values
    unique_networks = np.unique(networks)

    features = {}
    for i, net_i in enumerate(unique_networks):
        idx_i = np.where(networks == net_i)[0]

        for j, net_j in enumerate(unique_networks):
            if j < i:  # Skip lower triangle to avoid duplicates
                continue

            idx_j = np.where(networks == net_j)[0]
            sub_matrix = fc_matrix[np.ix_(idx_i, idx_j)]

            mean_conn = np.mean(sub_matrix)
            feature_name = f"{net_i}_{net_j}_mean_conn"
            features[feature_name] = mean_conn

    #features["group"] = group
    return features



def create_feature_dataframe(merged: pd.DataFrame,
                             network_table: pd.DataFrame,
                             group_from: str = "subject_prefix",
                             group_threshold: int = 72) -> pd.DataFrame:
    """
    Build a per-subject feature DataFrame from a merged DataFrame with columns:
        ['Subject', 'FC', 'Emo_res'].
    Adds Subject, Emo_res, and group to each feature row.

    group_from:
        - "subject_prefix": group=0 for 'sub-11...' (YA), group=1 for 'sub-12...' (OA)
        - "index_threshold": group=0 for first 'group_threshold' rows, else 1
        - "none": do not assign a group (set to None)
    """
    rows = []
    for i, row in merged.iterrows():
        subject = row["Subject"]
        fc_matrix = row["FC"]
        emo_res = row["Emo_res"]

        # Decide group
        if group_from == "subject_prefix":
            # Example: sub-11xxx -> group 0 (YA), sub-12xxx -> group 1 (OA)
            if isinstance(subject, str) and subject.startswith("sub-11"):
                group = 0
            elif isinstance(subject, str) and subject.startswith("sub-12"):
                group = 1
            else:
                group = None  # fallback if pattern differs
        elif group_from == "index_threshold":
            group = 0 if i < group_threshold else 1
        else:
            group = None

        # Compute features for this subject
        feats = extract_subject_connectivity_features(
            fc_matrix=fc_matrix,
            network_table=network_table,
            group=group if group is not None else "all",  # your function expects an int or "all"
        )

        # Attach identifiers and target
        feats["Subject"] = subject
        feats["Emo_res"] = emo_res
        feats["group"] = group

        rows.append(feats)

    features_df = pd.DataFrame(rows)

    # Reorder columns: Subject | Emo_res | group | features...
    first_cols = ["Subject", "Emo_res", "group"]
    other_cols = [c for c in features_df.columns if c not in first_cols]
    features_df = features_df[first_cols + other_cols]

    return features_df





def generate_synthetic_feature_dataframe(n_subjects=10, n_rois=200, n_networks=7, random_state=None):
    """
    Generate a small synthetic feature DataFrame for testing the pipeline.

    Parameters
    ----------
    n_subjects : int
        Number of synthetic subjects to simulate.
    n_rois : int
        Number of ROIs (e.g., 200 for quick testing).
    n_networks : int
        Number of functional networks (e.g., 7 for Yeo-7).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    features_df : pd.DataFrame
        Synthetic subjects × features table.
    """

    from data_load import generate_random_fc_matrix  # assuming your generator is in data_load


    rng = np.random.default_rng(random_state)

    # --- 1. Create a mock LUT dataframe ---
    network_labels = np.repeat(np.arange(n_networks), n_rois // n_networks)
    remainder = n_rois - len(network_labels)
    if remainder > 0:
        network_labels = np.concatenate([network_labels, rng.integers(0, n_networks, remainder)])

    network_names = np.array(["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"])
    net_map = {i: network_names[i % len(network_names)] for i in range(n_networks)}

    lut_df = pd.DataFrame({
        "Index": np.arange(n_rois),
        "Network": [net_map[i] for i in network_labels],
        "Hemisphere": rng.choice(["LH", "RH"], size=n_rois)
    })

    # --- 2. Simulate FC matrices and extract features ---
    all_features = []
    for subj in range(n_subjects):
        fc = generate_random_fc_matrix(n_rois=n_rois, n_networks=n_networks, random_state=subj)
        feats = extract_subject_connectivity_features(fc, lut_df)
        feats["subject"] = subj + 1
        all_features.append(feats)

    # --- 3. Combine into one DataFrame ---
    features_df = pd.DataFrame(all_features).set_index("subject")

    print(f"✅ Generated synthetic feature dataframe: {features_df.shape[0]} subjects × {features_df.shape[1]} features")
    return features_df




def remove_duplicate_pairs(df):
    keep, seen = [], set()
    for c in df.columns:
        if "_mean_conn" in c:
            parts = c.replace("_mean_conn", "").split("_")
            key = tuple(sorted(parts[:2]))
            if key not in seen:
                keep.append(c)
                seen.add(key)
        else:
            keep.append(c)
    return df[keep]

