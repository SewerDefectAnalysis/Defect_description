from pathlib import Path
from typing import Sequence, Dict

import pandas as pd
DEFAULT_EXCEL_PATH = "..\..\2.Data_validation\Validation_rules\Validated_data.xlsx"

def load_multiple_sheets(
    excel_path: str | Path = DEFAULT_EXCEL_PATH,
    sheet_names: Sequence[str] | None = None,
    engine="openpyxl",
    report: bool = True,  # 👈 nuevo parámetro opcional
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple sheets from an Excel file into a dictionary of DataFrames
    and optionally report number of rows loaded.

    Parameters
    ----------
    excel_path : str or Path
        Path to the Excel file.
    sheet_names : list-like of str, optional
        Names of the sheets to load. If None, all sheets are loaded.
    engine : str
        Excel engine to use.
    report : bool, default=True
        Whether to print number of rows loaded per sheet.

    Returns
    -------
    dict
        Dictionary mapping sheet_name -> DataFrame.
    """
    excel_path = Path(excel_path)

    if not excel_path.exists():
        raise FileNotFoundError(
            f"Excel file not found: {excel_path}. "
            "Please check the file name and path."
        )

    xls = pd.ExcelFile(excel_path, engine=engine)
    available = xls.sheet_names

    # If user does not specify, load all sheets
    if sheet_names is None:
        sheet_names = available
    else:
        missing = [s for s in sheet_names if s not in available]
        if missing:
            print(
                f"⚠️ Warning: the following sheets were not found in {excel_path}: {missing}"
            )
            sheet_names = [s for s in sheet_names if s in available]

    dataframes: Dict[str, pd.DataFrame] = {}

    for s in sheet_names:
        df = pd.read_excel(excel_path, sheet_name=s, engine=engine)
        df.columns = [str(c).strip() for c in df.columns]
        dataframes[s] = df

        # 👇 aquí agregamos el reporte
        if report:
            print(f"Loaded {len(df)} rows from '{s}'.")

    return dataframes