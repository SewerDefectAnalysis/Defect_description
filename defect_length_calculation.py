import pandas as pd
import numpy as np

def normalize_longitudinal_distance(df_defects, df_cctv, df_pipes,
                                    inspection_id_col="Inspection_ID",
                                    pipe_id_col="Pipe_ID",
                                    ld_col="Longitudinal_distance",
                                    pipe_length_col="Pipe_length"):
    """
    Adds Pipe_ID and normalized longitudinal distance to the defects dataframe.

    Steps:
    1. Merge defects with CCTV to get Pipe_ID
    2. Merge with pipes to get Pipe_length
    3. Compute normalized longitudinal distance
    """

    # Step 1: Add Pipe_ID via CCTV
    df = df_defects.merge(
        df_cctv[[inspection_id_col, pipe_id_col]],
        on=inspection_id_col,
        how="left"
    )

    # Step 2: Add Pipe_length via pipes
    df = df.merge(
        df_pipes[[pipe_id_col, pipe_length_col]],
        on=pipe_id_col,
        how="left"
    )

    # Step 3: Compute normalized longitudinal distance
    df["Longitudinal_distance_normalized"] = df[ld_col] / df[pipe_length_col]

    return df

def process_continuous_defects(df,
                               pipe_id_col="Pipe_ID",
                               defect_type_col="Defect_code",
                               cont_def_col="Continuous_defect",
                               ld_norm_col="Longitudinal_distance_normalized"):
    """
    Combines start (S) and finish (F) defects into a single row
    and keeps ALL original columns.
    """

    df = df.copy()
    df[cont_def_col] = df[cont_def_col].fillna("")

    # Extract type (S/F) and ID
    df["cont_type"] = df[cont_def_col].str[0]
    df["cont_id"] = df[cont_def_col].str[1:]

    df_start = df[df["cont_type"] == "S"].copy()
    df_finish = df[df["cont_type"] == "F"].copy()

    # Merge (keep START row as base)
    df_merged = df_start.merge(
        df_finish,
        on=[pipe_id_col, defect_type_col, "cont_id"],
        suffixes=("", "_end"),
        how="left"
    )

    # Compute values
    df_merged["start_LD_norm"] = df_merged[ld_norm_col]
    df_merged["end_LD_norm"] = df_merged[f"{ld_norm_col}_end"]

    df_merged["defect_length"] = (
        df_merged["end_LD_norm"] - df_merged["start_LD_norm"]
    ).abs()

    # Drop duplicated "_end" columns
    cols_to_drop = [col for col in df_merged.columns if col.endswith("_end")]
    df_merged = df_merged.drop(columns=cols_to_drop)

    return df_merged

def extract_single_defects(df,
                           cont_def_col="Continuous_defect",
                           ld_norm_col="Longitudinal_distance_normalized"):
    """
    Keeps original rows for single defects and adds new columns.
    """

    df = df.copy()
    df[cont_def_col] = df[cont_def_col].fillna("")

    mask = (
        (df[cont_def_col] == "") |
        ~df[cont_def_col].str.startswith(("S", "F"))
    )

    df_single = df[mask].copy()

    df_single["start_LD_norm"] = df_single[ld_norm_col]
    df_single["end_LD_norm"] = df_single[ld_norm_col]
    df_single["Defect_length"] = 0.0

    return df_single

def build_defect_dataset(df_defects, df_cctv, df_pipes):

    # Step 1
    df_norm = normalize_longitudinal_distance(df_defects, df_cctv, df_pipes)

    # Step 2
    df_cont = process_continuous_defects(df_norm)

    # Step 3
    df_single = extract_single_defects(df_norm)

    # Combine
    df_final = pd.concat([df_cont, df_single], ignore_index=True)

    df_defects2 = df_final.drop(columns=["Pipe_length", 'cont_type', 'cont_id'])

    return df_defects2