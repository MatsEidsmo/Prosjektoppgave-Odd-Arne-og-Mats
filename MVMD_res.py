from my_imports import *
import os, re, glob
from res_scores import extract_res_scores_from_csv

res_df = extract_res_scores_from_csv(r"C:\Users\matsei\Documents\Mats og Odd Arne\Prosjektoppgave\ISC_data\Beh.csv")
res_df = res_df.drop_duplicates(subset=["Subject"], keep="last")
res_df = res_df.sort_values("Subject").reset_index(drop=True)

mvmd_dir = r".\MVMD_decomposed"
files = sorted(glob.glob(os.path.join(mvmd_dir, "mvmd_decomposed_modes*.npz")))


def extract_file_index(fp: str) -> int:
    m = re.search(r"mvmd_decomposed_modes(\d+)\.npz$", os.path.basename(fp))
    if not m:
        raise ValueError(f"Cannot parse index from filename: {fp}")
    return int(m.group(1))

file_records = [{"file_path": fp, "FileIndex": extract_file_index(fp)} for fp in files]
files_df = pd.DataFrame(file_records).sort_values("FileIndex").reset_index(drop=True)


n_files = len(files_df)
n_subjects = len(res_df)
n = min(n_files, n_subjects)  # safe guard if counts differ
files_df = files_df.iloc[:n].copy()
res_df   = res_df.iloc[:n].copy()

paired = files_df.copy()
paired["Subject"] = res_df["Subject"].values
paired = paired.merge(res_df[["Subject", "Emo_res"]], on="Subject", how="left")

print(f"Paired DataFrame:\n{paired.head()}")




def rms_mode(arr: np.ndarray) -> float:
    """RMS of a mode array across all its elements (time ± extra dims)."""
    a = np.asarray(arr)
    return float(np.sqrt(np.mean(a**2)))


def load_network_table(path="network_table.txt") -> pd.DataFrame:
    """
    Load network_table.txt and return a DataFrame with useful columns:
    ['label', 'roi', 'hemi', 'network', 'subnet'].
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    data = []
    for i in range(0, len(lines), 2):  # step through pairs
        label = lines[i]
        nums = lines[i + 1].split()
        roi = int(nums[0])

        # Parse components from label: e.g., '7Networks_LH_Default_PFC_1'
        parts = label.split("_")
        hemi = parts[1] if len(parts) > 1 else None
        network = parts[2] if len(parts) > 2 else None
        subnet = parts[3] if len(parts) > 3 else None

        data.append({"label": label, "roi": roi, "hemi": hemi, "network": network, "subnet": subnet})

    return pd.DataFrame(data)

def get_rois_by_network(df: pd.DataFrame, network_name: str, hemis=("LH", "RH")) -> list:
    """
    Return ROI indices for a given network (e.g., 'Default') and hemispheres.
    """
    return df[(df["network"] == network_name) & (df["hemi"].isin(hemis))]["roi"].tolist()


def plot_network_lowfreq_vs_resilience(
    paired: pd.DataFrame,
    roi_indices: list,
    low_mode_by_freq: bool = False,
    low_amp_col_name: str = "LowFreqAmp_DMN",  # name to store (optional)
    figsize=(7,4)
):
    """
    Plot LOW-IMF amplitude vs Resilience (Emo_res) for a specific network (ROIs given).

    Parameters
    ----------
    paired : pd.DataFrame
        Must contain columns: ['file_path', 'Subject', 'Emo_res'].
    roi_indices : list[int]
        ROI indices belonging to the network (e.g., from get_rois_by_network(...)).
    low_mode_by_freq : bool
        If True, select the lowest-frequency IMF by argmin(frequencies).
        If False, use mode index 0 as the low-frequency IMF.
    low_amp_col_name : str
        Column name to store computed amplitudes in a copy of `paired`.
    figsize : tuple
        Figure size.

    Returns
    -------
    dict : {"pearson": (r, p), "spearman": (rho, p), "df": DataFrame_used_for_plot}
    """

    # Minimal checks
    for c in ["file_path", "Emo_res", "Subject"]:
        if c not in paired.columns:
            raise KeyError(f"`paired` missing required column: {c}")
    if not len(roi_indices):
        raise ValueError("`roi_indices` is empty. Provide ROIs for the network.")

    def rms(a: np.ndarray) -> float:
        # RMS across all elements (time and selected ROIs)
        return float(np.sqrt(np.mean(np.square(a))))

    def compute_lowfreq_rms_for_file(fp: str) -> float:
        with np.load(fp, allow_pickle=True) as npz:
            modes = npz["modes"]  # shape: (K, T, R) or (K, T) if single ROI
            # Ensure we index ROIs robustly
            if modes.ndim == 3:
                # (K, T, R): restrict to selected ROIs, then RMS over T and ROIs
                if low_mode_by_freq and "frequencies" in npz:
                    f = np.array(npz["frequencies"]).reshape(-1)
                    k = int(np.argmin(f))
                else:
                    k = 0
                m = modes[k, :, roi_indices]  # shape: (T, len(ROIs))
                return rms(m)
            elif modes.ndim == 2:
                # (K, T): no ROI dimension; just RMS on mode k
                if low_mode_by_freq and "frequencies" in npz:
                    f = np.array(npz["frequencies"]).reshape(-1)
                    k = int(np.argmin(f))
                else:
                    k = 0
                m = modes[k, :]  # (T,)
                return rms(m)
            else:
                raise ValueError(f"Unexpected `modes` shape: {modes.shape}")

    # Compute amplitudes
    df = paired.copy()
    df[low_amp_col_name] = df["file_path"].apply(compute_lowfreq_rms_for_file)

    # Sort by low-frequency amplitude (Fig. 2 logic)
    d = df[["Subject", "Emo_res", low_amp_col_name]].dropna().sort_values(low_amp_col_name).reset_index(drop=True)

    x = np.arange(len(d))
    y_low = d[low_amp_col_name].to_numpy(dtype=float)
    y_res = d["Emo_res"].to_numpy(dtype=float)

    # Correlations
    r, rp = pearsonr(y_low, y_res)
    rho, rhop = spearmanr(y_low, y_res)

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)
    
    ax1.scatter(x, y_low, color="#1f77b4", s=40, alpha=0.8, label="LOW IMF (amplitude)")
    

    ax1.set_xlabel("Subjects (sorted by LOW IMF amplitude)")
    ax1.set_ylabel("LOW IMF amplitude (RMS)", color="#1f77b4")
    ax1.tick_params(axis='y', labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.scatter(x, y_res, color="#d62728", s=40, alpha=0.8, label="Resilience (Emo_res)")
    ax2.set_ylabel("Resilience score (Emo_res)", color="#d62728")
    ax2.tick_params(axis='y', labelcolor="#d62728")

    ax1.set_title(
        f"LOW IMF amplitude (network ROIs) vs Resilience\n"
        f"Pearson r = {r:.3f} (p={rp:.3g})  |  Spearman = {rho:.3f} (p={rhop:.3g})"
    )

    # Combined legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_lines()
        lines.extend(L)
        labels.extend([l.get_label() for l in L])
    ax1.legend(lines, labels, loc="best", frameon=True)

    plt.tight_layout()
    plt.show()

    return {"pearson": (float(r), float(rp)), "spearman": (float(rho), float(rhop)), "df": d}

DMN_rois = get_rois_by_network(load_network_table(), "Default")
#DMN_rois = [r - 1 for r in DMN_rois]  # convert to 0-based indexing
#plot_network_lowfreq_vs_resilience(paired, DMN_rois)




import numpy as np
import pandas as pd

def network_connectivity_for_subject(file_path: str,
                                     net_df: pd.DataFrame,
                                     networks_order: tuple[str, ...],
                                     k_select: int = 0,
                                     hemis: tuple[str, ...] = ("LH", "RH")) -> np.ndarray:
    """
    Compute a subject-level inter-network Pearson correlation matrix over time.

    Parameters
    ----------
    file_path : str
        Path to the subject's MVMD .npz file. Must contain 'modes' with shape (K, T, R).
    net_df : pd.DataFrame
        Output of load_network_table(...). Must contain columns ['roi','hemi','network'].
        'roi' is 1-based (1..R); will be converted to 0-based.
    networks_order : tuple[str, ...]
        Ordered network names to include (e.g., ("Default","Salience","Limbic","Control","DorsAttn","SomMot","Visual")).
    k_select : int
        IMF index to use as the signal (0-based). No frequency inference is performed here.
    hemis : tuple[str, ...]
        Hemispheres to include (default LH and RH).

    Returns
    -------
    C : np.ndarray
        Pearson correlation matrix of shape (Nnet, Nnet), where Nnet = len(networks_order).

    Raises
    ------
    AssertionError
        If any dimension assumption fails or inputs are inconsistent.
    ValueError
        If a network has no ROIs or ROI indices are out of range.
    """

    # --- Load and validate array layout ---
    with np.load(file_path, allow_pickle=True) as z:
        assert "modes" in z, "'modes' key not found in npz."
        modes = z["modes"]  # expected shape: (K, T, R)

    assert modes.ndim == 3, f"'modes' must be 3D, got {modes.ndim}D with shape {modes.shape}."
    K, T, R = modes.shape  # (K, T, R) — strict assumption

    # IMF selection
    assert 0 <= k_select < K, f"k_select={k_select} out of range 0..{K-1}."

    # Extract the selected mode and set canonical (T, R) view
    # [LAYOUT] If your storage is (K, R, T), replace the next line with: TR = modes[k_select].T
    TR = modes[k_select]            # shape: (T, R)

    # --- Prepare network ROI indices (1-based -> 0-based) ---
    X_list = []
    for net in networks_order:
        rois_1b = net_df[(net_df["network"] == net) & (net_df["hemi"].isin(hemis))]["roi"].to_numpy(int)
        if rois_1b.size == 0:
            raise ValueError(f"No ROIs found for network '{net}'. Check network_table names.")
        rois0 = rois_1b - 1
        if rois0.min() < 0 or rois0.max() >= R:
            raise ValueError(f"ROI indices out of range for '{net}': min={rois_1b.min()}, max={rois_1b.max()}, valid 1..{R}")

        # --- Average over ROIs to get one time series per network ---
        ts = np.mean(TR[:, rois0], axis=1)  # (T,)
        # Strict: require variability (corr undefined for constant series)
        assert np.isfinite(ts).all(), f"Non-finite values in time series for '{net}'."
        assert ts.std() > 0.0, f"Zero-variance time series for '{net}'."

        X_list.append(ts)

    # Stack network time series: variables in rows → np.corrcoef(rows) returns (Nnet × Nnet)
    X = np.vstack(X_list)  # shape: (Nnet, T)
    assert X.shape == (len(networks_order), T), f"Unexpected stacked shape {X.shape}; expected ({len(networks_order)}, {T})."

    C = np.corrcoef(X)     # (Nnet, Nnet)
    assert C.shape == (len(networks_order), len(networks_order)), f"Unexpected corr shape {C.shape}."

    return C




def plot_extreme_connectivity(paired, net_df,
                              networks_order=("Default","SalVentAttn","Limbic","Cont","DorsAttn","SomMot","Vis"),
                              k_select="minfreq", cmap="coolwarm", savepath=None):
    i_min = paired["Emo_res"].idxmin()
    i_max = paired["Emo_res"].idxmax()
    rmin = paired.loc[i_min]
    rmax = paired.loc[i_max]

    C_min = network_connectivity_for_subject(rmin["file_path"], net_df, networks_order, k_select=k_select)
    C_max = network_connectivity_for_subject(rmax["file_path"], net_df, networks_order, k_select=k_select)

    short = ["DMN","SalVentAttn","Limbic","Cont","DorsAttn","SomMot","Vis"]

    fig, axes = plt.subplots(1,2, figsize=(10,4), constrained_layout=True)
    for ax, C, title in (
        (axes[0], C_min, f"Lowest resilience\n (Emo_res={rmin['Emo_res']:.1f})"),
        (axes[1], C_max, f"Highest resilience\n (Emo_res={rmax['Emo_res']:.1f})"),
    ):
        im = ax.imshow(C, vmin=-1, vmax=1, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
        ax.set_xticklabels(short, rotation=45, ha="right"); ax.set_yticklabels(short)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Pearson r")

    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()

    return {"C_min": C_min, "C_max": C_max,
            "min_subject": int(rmin["Subject"]), "max_subject": int(rmax["Subject"])}


def compute_connectivity_matrices(paired: pd.DataFrame,
                                  net_df: pd.DataFrame,
                                  networks_order: tuple[str, ...],
                                  k_select: int = 0) -> np.ndarray:
    """
    Build connectivity matrices for all subjects.

    Parameters
    ----------
    paired : DataFrame
        Must contain ['Subject', 'Emo_res', 'file_path'].
    net_df : DataFrame
        Output of load_network_table(...).
    networks_order : tuple[str, ...]
        ('Default','Salience','Limbic','Control','DorsAttn','SomMot','Visual'), etc.
    k_select : int
        IMF index to use.

    Returns
    -------
    M : np.ndarray
        Shape (Nsub, Nnet, Nnet), subject-wise connectivity matrices.
    """
    mats = []
    for _, row in paired.iterrows():
        C = network_connectivity_for_subject(
            file_path=row["file_path"],
            net_df=net_df,
            networks_order=networks_order,
            k_select=k_select
        )
        mats.append(C)
    M = np.stack(mats, axis=0)  # (Nsub, Nnet, Nnet)
    return M


def edgewise_corr_vs_resilience(M: np.ndarray,
                                emo_res: np.ndarray,
                                networks_order: tuple[str, ...]) -> pd.DataFrame:
    """
    For each edge (i<j), compute Pearson correlation between edge strength across subjects and Emo_res.

    Parameters
    ----------
    M : np.ndarray
        (Nsub, Nnet, Nnet)
    emo_res : np.ndarray
        (Nsub,) resilience scores aligned to the same subject order as M.
    networks_order : tuple[str, ...]
        Names for formatting.

    Returns
    -------
    df : DataFrame
        Columns: ['edge_i','edge_j','edge_name','r','p'] ready for later FDR correction.
    """
    Nsub, Nnet, _ = M.shape
    assert emo_res.shape[0] == Nsub, "Emo_res length must match number of subjects."

    rows = []
    for i in range(Nnet):
        for j in range(i+1, Nnet):
            y = M[:, i, j]                     # edge strength across subjects
            r, p = pearsonr(y, emo_res)        # correlation with resilience
            rows.append({
                "edge_i": i,
                "edge_j": j,
                "edge_name": f"{networks_order[i]}–{networks_order[j]}",
                "r": float(r),
                "p": float(p)
            })
    df = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    return df

networks_order=("Default","SalVentAttn","Limbic","Cont","DorsAttn","SomMot","Vis")
net_df = load_network_table()

M = compute_connectivity_matrices(paired, net_df, networks_order, k_select=0)

emo_res = paired["Emo_res"].to_numpy(dtype=float)
df_edges = edgewise_corr_vs_resilience(M, emo_res, networks_order)
print(df_edges.head(10))


def plot_edge_bars(df_edges, top_k=15, sort_by="p"):
    """
    df_edges columns: ['edge_name','r','p'] (from your edgewise_corr_vs_resilience).
    """
    d = df_edges.sort_values(sort_by).head(top_k).copy()
    colors = np.where(d["r"] >= 0, "#d62728", "#1f77b4")  # red=positive, blue=negative

    plt.figure(figsize=(10, 4.5))
    plt.barh(d["edge_name"], d["r"], color=colors)
    plt.axvline(0, color="k", lw=1)
    for i, (r, p) in enumerate(zip(d["r"], d["p"])):
        dx = 0.02
        if r >= 0:
            x = r - dx; ha = "right"
        else:
            x = r + dx; ha = "left"

        
        plt.text(x, i, f"p={p:.3f}", va="center", ha=ha, fontsize=8,
            color="white", fontweight="bold")

    plt.xlabel("Correlation with resilience (r)")
    plt.title("Top edges by p-value")
    plt.gca().invert_yaxis()
    #plt.tight_layout()
    #plt.savefig("Figurer\top_edges_resilience_correlation.svg")
    plt.show()

plot_edge_bars(df_edges, top_k=15, sort_by="p")
