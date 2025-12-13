from my_imports import *
import pandas as pd


def extract_res_scores_from_csv(file_path: str) -> pd.DataFrame:

    all_data = pd.read_csv(file_path)
    res_scores_df = all_data[["Subject", "Emo_res"]]
    print(f"Extracted res_scores:\n{res_scores_df}")

    return res_scores_df

def merge_res_scores(fc_matrices: pd.DataFrame) -> pd.DataFrame:
    

    # Load CSV scores
    res_df = extract_res_scores_from_csv(r"C:\Users\matsei\Documents\Mats og Odd Arne\Prosjektoppgave\ISC_data\Beh.csv")
    res_df = res_df.drop_duplicates(subset=["Subject"], keep="last")

    # Normalize Subject keys for merging
    
    fc_matrices["Subject"] = fc_matrices["Subject"].map(normalize_subject_key)
    res_df["Subject"] = res_df["Subject"].map(normalize_subject_key)

    # Inner-join to keep only subjects present in both sources
    merged = pd.merge(fc_matrices, res_df, on="Subject", how="inner")

    return merged



def normalize_subject_key(x):
    """
    Returns subject key as 'sub-xxxxx' in lowercase, where xxxxx are digits.
    Handles inputs like 11001 (int), '11001', 'sub-11001', ' SUB-11001 '.
    Leaves other strings untouched but lowercased/stripped.
    """
    s = str(x).strip().lower()
    if s.startswith("sub-"):
        return s  # already correct form
    # If it's numeric-like (e.g., '11001' or 11001), prepend 'sub-'
    digits = "".join(ch for ch in s if ch.isdigit())
    if digits:  # we found digits like '11001'
        # If your filenames are 'sub-11xxx' and 'sub-12xxx' (5 digits total), keep as is:
        return f"sub-{digits}"
    return s
