from my_imports import *
import pandas as pd


def extract_res_scores_from_csv(file_path: str) -> pd.DataFrame:

    all_data = pd.read_csv(file_path)
    res_scores_df = all_data[["Subject", "Emo_res"]]
    print(f"Extracted res_scores:\n{res_scores_df}")

    return res_scores_df