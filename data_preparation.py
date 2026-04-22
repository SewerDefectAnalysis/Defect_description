import config as cfg
from typing import Mapping, Sequence, Dict, List, Tuple, Optional
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors

class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""
    pass

def merge_df_pipes_hydraulic(
    df_pipes: pd.DataFrame,
    df_hydraulic: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the pipes dataframe with the hydraulic dataframe using a left join
    on the column 'Pipe_ID'.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    if "Pipe_ID" not in df_pipes.columns or "Pipe_ID" not in df_hydraulic.columns:
        raise KeyError("Column 'Pipe_ID' not found")

    df_merged = df_pipes.merge(
        df_hydraulic,
        how="left",
        on="Pipe_ID"
    )

    return df_merged

def get_available_materials(df_pipes, material_col="Material"):
    """
    Returns and prints the unique materials available in the dataset.
    """

    materials = sorted(df_pipes[material_col].dropna().unique())

    print("\nAvailable materials in dataset:")
    for m in materials:
        print(f"- {m}")

    return materials

def filter_by_material(
    df: pd.DataFrame,
    selected_materials: Sequence[str] | None = None,
    material_column: str = "Material"
) -> pd.DataFrame:
    """
    Filter any DataFrame by selected materials.
    Invalid materials are ignored with a warning.
    """

    # Use global config if not provided
    if selected_materials is None:
        selected_materials = cfg.selected_materials

    if selected_materials is None:
        raise DataPreparationError(
            "selected_materials is not initialized. Please set cfg.selected_materials first."
        )

    # 1. Validate column exists
    if material_column not in df.columns:
        raise DataPreparationError(
            f"Column '{material_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # 2. Available materials
    materials_in_df = df[material_column].dropna().unique().tolist()

    # 3. Separate valid and invalid
    valid_materials = [m for m in selected_materials if m in materials_in_df]
    invalid_materials = [m for m in selected_materials if m not in materials_in_df]

    # 4. Print warnings instead of raising error
    if invalid_materials:
        print("\n❌ Materials NOT found in dataset:")
        for m in invalid_materials:
            print(f"   - {m}")

    if valid_materials:
        print("\n✅ Materials used for filtering:")
        for m in valid_materials:
            print(f"   - {m}")

    # 5. Handle case: no valid materials
    if not valid_materials:
        print("\n⚠️ No valid materials selected → using ALL data")
        return df.copy()

    # 6. Filter
    df_filtered = df[df[material_column].isin(valid_materials)].copy()

    print(f"\nPipes selected: {len(df_filtered)}")

    return df_filtered

def validate_materials(
    df_pipes: pd.DataFrame,
    selected_materials: Sequence[str] | None = None,
    material_column: str = "Material"
) -> bool:
    """
    Validates that the selected materials exist in the 'MATERIAL' column of df_pipes.

    Parameters:
    -----------
    df_pipes : pd.DataFrame
        DataFrame containing a 'MATERIAL' column.
    selected_materials : list
        List of materials to validate.

    Returns:
    --------
    bool
        True if all selected materials exist in df_pipes, False otherwise.
    """
    # Use global selected_materials if not provided
    if selected_materials is None:
        selected_materials = cfg.selected_materials

    if selected_materials is None:
        raise ValueError("selected_materials is not initialized. Please set cfg.selected_materials first.")

    materials_in_df = df_pipes['Material'].unique()
    missing_materials = [m for m in selected_materials if m not in materials_in_df]

    if missing_materials:
        print(
            f"The following materials are not present in PIPES: {missing_materials}. "
            f"The available materials in PIPES are: {list(materials_in_df)}"
        )
        return False
    else:
        print(
            f"The material selection was successful. "
            f"The selected materials are: {selected_materials}"
        )
        return True

def build_material_color_map(
    selected_materials: Sequence[str] | None = None,
    predefined_colors: Dict[str, str] | None = None,
    others_color: str = "#999999",
    colormap_name: str = "Paired"
) -> Dict[str, str]:
    """
    Build a color dictionary for materials. This color will consistently represent the material across all charts where the materials are plotted.

    Materials with predefined colors keep them. Remaining materials
    are assigned colors from a matplotlib colormap. 'OTHERS' is always
    assigned a fixed gray color.

    Uses cfg.selected_materials and cfg.colors_paper if not provided.

    Parameters
    ----------
    selected_materials : sequence of str, optional
        Materials to include in the color map. If None, uses cfg.selected_materials.
    predefined_colors : dict, optional
        Dictionary of material -> hex color. If None, uses cfg.colors_paper.
    others_color : str, default "#999999"
        Color assigned to 'OTHERS'.
    colormap_name : str, default "Paired"
        Matplotlib colormap used for remaining materials.

    Returns
    -------
    dict
        Dictionary mapping material to hex color.
    """
    # Use globals if not provided
    if selected_materials is None:
        selected_materials = cfg.selected_materials
    if predefined_colors is None:
        predefined_colors = cfg.colors_paper

    if selected_materials is None:
        raise ValueError("selected_materials is not initialized. Please set cfg.selected_materials first.")

    if not selected_materials:
        raise ValueError("selected_materials cannot be empty.")

    predefined_colors = predefined_colors or {}

    # Materials without predefined colors
    remaining_materials = [m for m in selected_materials if m not in predefined_colors]

    # Create colormap for remaining materials
    cmap = mpl.colormaps[colormap_name].resampled(len(remaining_materials)) if remaining_materials else []

    colors_materials: Dict[str, str] = {}

    # Assign predefined colors
    for material, color in predefined_colors.items():
        if material in selected_materials:
            colors_materials[material] = color

    # Assign colormap colors
    for i, material in enumerate(remaining_materials):
        colors_materials[material] = mcolors.to_hex(cmap(i))

    # Assign fixed color for OTHERS
    colors_materials["OTHERS"] = others_color

    return colors_materials

def validate_factors(
    df_pipes: pd.DataFrame,
    factors: Sequence[str] | None = None,
    raise_error: bool = True
) -> List[str]:
    """
    Validate that selected factors exist in the Pipes DataFrame.
    Uses cfg.factors if factors is not provided.

    Parameters
    ----------
    df_pipes : pd.DataFrame
        Pipes DataFrame.
    factors : sequence of str, optional
        Factors selected by the user. If None, uses cfg.factors.
    raise_error : bool, default True
        Whether to raise an error if factors are missing.

    Returns
    -------
    list of str
        List of validated pipe factors.

    Raises
    ------
    DataPreparationError
        If one or more factors are missing and raise_error is True.
    """
    # Use global factors if not provided
    if factors is None:
        factors = cfg.factors

    if factors is None:
        raise ValueError("factors is not initialized. Please set cfg.factors first.")

    available_factors = df_pipes.columns.tolist()
    missing_factors = [f for f in factors if f not in available_factors]

    if missing_factors:
        message = (
            f"The following factors are not present in the PIPES dataset: "
            f"{missing_factors}. "
            f"Available factors in PIPES: {available_factors}"
        )

        if raise_error:
            raise DataPreparationError(message)

        print(f"️ {message}")
        # Return only the factors that exist
        return [f for f in factors if f in available_factors]

    print(
        "The factors were successfully selected. "
        f"Selected factors: {list(factors)}"
    )

    return list(factors)


def classify_pipe_factors(
    df_pipes: pd.DataFrame,
    factors: Sequence[str] | None = None,
    numeric_factors_override: Optional[Sequence[str]] = None,
    categorical_factors_override: Optional[Sequence[str]] = None,
    verbose: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Classify pipe factors into numeric and categorical.

    Automatic classification is based on pandas dtypes, but the user
    can override the classification if needed.

    Uses cfg.factors if factors is not provided.

    Parameters
    ----------
    df_pipes : pd.DataFrame
        Pipes DataFrame.
    factors : sequence of str, optional
        Selected pipe factors. If None, uses cfg.factors.
    numeric_factors_override : sequence of str, optional
        User-defined numeric factors.
    categorical_factors_override : sequence of str, optional
        User-defined categorical factors.
    verbose : bool, default True
        Whether to print classification details.

    Returns
    -------
    factors_num : list of str
        Numeric factors.
    factors_cat : list of str
        Categorical factors.

    Raises
    ------
    DataPreparationError
        If overridden factors are invalid or inconsistent.
    """
    # Use global factors if not provided
    if factors is None:
        factors = cfg.factors

    if factors is None:
        raise ValueError("factors is not initialized. Please set cfg.factors first.")

    # --- Automatic classification ---
    numeric_columns = df_pipes.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df_pipes.select_dtypes(exclude=["number"]).columns.tolist()

    auto_num = [f for f in factors if f in numeric_columns]
    auto_cat = [f for f in factors if f in categorical_columns]

    # --- Apply user overrides if provided ---
    factors_num = auto_num
    factors_cat = auto_cat

    if numeric_factors_override is not None or categorical_factors_override is not None:
        num_override = set(numeric_factors_override or [])
        cat_override = set(categorical_factors_override or [])

        # --- Validation ---
        overlap = num_override & cat_override
        if overlap:
            raise DataPreparationError(
                f"The following factors are classified as both numeric and categorical: "
                f"{list(overlap)}"
            )

        all_overridden = num_override | cat_override
        invalid = [f for f in all_overridden if f not in df_pipes.columns]
        if invalid:
            raise DataPreparationError(
                f"The following overridden factors do not exist in PIPES: {invalid}"
            )

        # --- Final classification ---
        factors_num = [f for f in factors if f in num_override]
        factors_cat = [f for f in factors if f in cat_override]

    if verbose:
        print("Classification:")
        print(f"Numeric factors: {factors_num}")
        print(f"Categorical factors: {factors_cat}")

    return factors_num, factors_cat


def merge_cctv_defects(
        df_cctv: pd.DataFrame,
        df_pipes_filtered: pd.DataFrame,
        df_defects: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge filtered CCTV and pipe data into the defects dataframe.

    Parameters:
    - df_cctv: DataFrame with CCTV inspections
    - df_pipes_filtered: DataFrame with filtered pipe information
    - df_defects: DataFrame with defects to be enriched

    Returns:
    - df_defects merged with CCTV and pipe data
    """
    # Merge CCTV with filtered pipes
    df_cctv_filtered = pd.merge(df_cctv, df_pipes_filtered, on='Pipe_ID', how='right')

    # Merge defects with the enriched CCTV data
    df_defects_filtered = pd.merge(df_defects, df_cctv_filtered, on='Pipe_ID', how='right')

    return df_cctv_filtered, df_defects_filtered

def create_defect_color_map(
    df: pd.DataFrame,
    Defect_code_col: str = "Defect_code",
    default_palette: Dict[str, str] | None = None
) -> Dict[str, str]:
    """
    Create a color map for defect types using a default palette.

    Uses cfg.palette_defects_final if default_palette is not provided.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing defect codes.
    Defect_code_col : str
        Name of the column with defect codes.
    default_palette : dict, optional
        Dictionary mapping defect codes to colors. If None, uses cfg.palette_defects_final.

    Returns
    -------
    dict
        Dictionary mapping defect codes to colors.
    """
    # Use global palette if not provided
    if default_palette is None:
        default_palette = cfg.palette_defects_final

    if default_palette is None:
        raise ValueError("default_palette is not initialized. Please set cfg.palette_defects_final first.")

    palette_defects_final_local: Dict[str, str] = {}
    for defect in df[Defect_code_col].unique():
        # Use color from default palette if available, else gray
        palette_defects_final_local[defect] = default_palette.get(defect, "#999999")

    return palette_defects_final_local
