import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import config as cfg
from data_preparation import build_material_color_map

from collections.abc import Sequence
from typing import Tuple, Optional, List

colors_material = build_material_color_map(cfg.selected_materials, cfg.colors_paper)

## Statistics summary --------------------------------------------------------------------------------------------------
def combined_summary_two_tables(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    factors_num: Sequence[str] | None = None,
    factors_cat: Sequence[str] | None = None,
    names: Tuple[str, str] = ("CCTV", "Network")
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generates numeric and categorical summaries for two datasets
    and returns them as two separate DataFrames.

    Uses cfg.factors_num and cfg.factors_cat if not provided.

    Parameters
    ----------
    df1, df2 : pd.DataFrame
        DataFrames to summarize.
    factors_num : sequence of str, optional
        Numeric factors to summarize. If None, uses cfg.factors_num.
    factors_cat : sequence of str, optional
        Categorical factors to summarize. If None, uses cfg.factors_cat.
    names : tuple of str, default ("CCTV", "Network")
        Names of the datasets for labeling.

    Returns
    -------
    summary_num : pd.DataFrame or None
        Numeric summary for both datasets.
    summary_cat : pd.DataFrame or None
        Categorical summary for both datasets.
    """
    # Use globals if not provided
    if factors_num is None:
        factors_num = cfg.factors_num
    if factors_cat is None:
        factors_cat = cfg.factors_cat

    if factors_num is None and factors_cat is None:
        raise ValueError("Both factors_num and factors_cat are None. Please set cfg.factors_num and/or cfg.factors_cat.")

    summary_num = None
    summary_cat = None

    # --- Numeric summary ---
    if factors_num:
        summaries = []
        for df, name in zip([df1, df2], names):
            desc = df[factors_num].describe(percentiles=[0.5]).T
            desc["median"] = df[factors_num].median()
            desc = desc[["count", "mean", "std", "min", "median", "max"]].reset_index()
            desc = desc.rename(columns={"index": "variable"})
            desc["dataset"] = name
            cols = ["variable", "dataset", "count", "mean", "std", "min", "median", "max"]
            desc = desc[cols]
            summaries.append(desc)
        summary_num = pd.concat(summaries, ignore_index=True).sort_values(
            by=["variable", "dataset"], ascending=[True, False]
        ).reset_index(drop=True)
        summary_num['variable'] = pd.Categorical(summary_num['variable'], categories=factors_num, ordered=True)
        summary_num = summary_num.sort_values('variable', key=lambda col: col.map({f: i for i, f in enumerate(factors_num)}))

    # --- Categorical summary ---
    if factors_cat:
        summaries = []
        for df, name in zip([df1, df2], names):
            rows = []
            for col in factors_cat:
                value_pct = df[col].value_counts(normalize=True, dropna=False) * 100
                formatted = ', '.join([
                    f"{'NaN' if pd.isna(cat) else cat} ({pct:.1f}%)"
                    for cat, pct in value_pct.items()
                ])
                rows.append({'variable': col, 'dataset': name, 'summary': formatted})
            summaries.append(pd.DataFrame(rows))
        summary_cat = pd.concat(summaries, ignore_index=True).sort_values(
            by=["variable", "dataset"], ascending=[True, False]
        ).reset_index(drop=True)
        summary_cat['variable'] = pd.Categorical(summary_cat['variable'], categories=factors_cat, ordered=True)
        summary_cat = summary_cat.sort_values('variable', key=lambda col: col.map({f: i for i, f in enumerate(factors_cat)}))

    return summary_num, summary_cat

## Statistic distribution of CCTV and Pipes ----------------------------------------------------------------------------
new_labels = {
    'Installation_year': 'Installation year (-)',
    'Age_CCTV': 'Inspection age (yrs.)',
    'Diameter': 'Diameter (mm)',
    'Pipe_length': 'Length (m)',
    'Slope': 'Slope (%)',
    'Depth': 'Depth (m)',
    'Dry_peak_flow_rate': 'Dry max. flow rate (L/s)',
    'Wet_peak_flow_rate': 'Wet max. flow rate (L/s)',
}
#---Functions to use ---
def ecdf(x):
    #---Empirical CDF of array x---
    x = np.asarray(list(x), dtype=float)
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys

def set_percentile_xlim(ax, series_list,  include_zero=False,p_low=0, p_high=99,col_name=None):
    #---Adjust x-axis limits based on combined percentiles of all series.
    vals = pd.concat([s.dropna() for s in series_list])
    lo, hi = np.percentile(vals, [p_low, p_high])
    left = min(0.0, lo) if include_zero else lo
    right = hi if hi > left else vals.max()
    if col_name in ['Dry_peak_flow_rate', 'Wet_peak_flow_rate']:
        right = 100

    span = max(right - left, np.finfo(float).eps)
    left -= 0.02 * span
    ax.set_xlim(left, right)
    left -= 0.02 * span
    ax.set_xlim(left, right)

def prob_superiority(a, b):
    #---Probability that values in a > values in b.
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a)==0 or len(b)==0:
        return np.nan
    ranks = pd.Series(np.r_[a, b]).rank(method="average").values
    R1 = ranks[:len(a)].sum()
    U1 = R1 - len(a)*(len(a)+1)/2
    return U1 / (len(a)*len(b))

# --- PS metrics ---
def compute_ps_table(
    df_population,
    df_sample,
    material_col,
    numeric_cols,
    materials,
    labels_boxplot: dict | None = None
):
    """
    Compute probability of superiority (PS) and return table with:
    rows = variables
    columns = materials
    """

    # --- Default labels (para nombres bonitos en paper) ---
    if labels_boxplot is None:
        labels_boxplot = {
            "Installation_year": "Installation Year",
            "Diameter": "Diameter",
            "Pipe_length": "Length",
            "Depth": "Average Depth",
            "Slope": "Slope",
            "Dry_peak_flow_rate": "Dry Max Flow Rate",
            "Wet_peak_flow_rate": "Wet Max Flow Rate",
        }

    rows = []

    # --- Compute PS ---
    for col in numeric_cols:
        for m in materials:
            pop = df_population.loc[df_population[material_col] == m, col].dropna()
            sam = df_sample.loc[df_sample[material_col] == m, col].dropna()

            if len(pop) > 0 and len(sam) > 0:
                ps = prob_superiority(sam, pop)
            else:
                ps = np.nan

            rows.append({
                "Variable": labels_boxplot.get(col, col),
                "Material": m,
                "P_superiority": ps
            })

    df_ps = pd.DataFrame(rows)

    # --- Pivot: materials as columns ---
    df_ps_pivot = df_ps.pivot(
        index="Variable",
        columns="Material",
        values="P_superiority"
    )

    df_ps_pivot = df_ps_pivot[materials]
    df_ps_pivot = df_ps_pivot.round(3)

    return df_ps_pivot


# --- Plotting ---
def plot_ecdf_by_material_overlay(
    df_population: pd.DataFrame,
    df_sample: pd.DataFrame,
    material_col: str = "Material",
    numeric_cols: list | None = None,
    materials: list | None = None,
    n_cols: int = 2,
    figsize: tuple = (18, 15),
    population_label: str = "Complete Network",
    sample_label: str = "Sample CCTV",
    colors_materials: dict | None = None,
    include_zero_cols: list = ["Pipe_length", "Slope", "Depth"],
    labels_boxplot: dict | None = None,
):
    """
    Plot ECDF overlays for numeric columns grouped by material.
    Uses global selected_materials and colors_materials if not provided.
    """
    if colors_materials is None:
        colors_materials = cfg.colors_materials
    if materials is None:
        materials = cfg.selected_materials

    if materials is None or colors_materials is None:
        raise ValueError("Please initialize cfg.selected_materials and cfg.colors_materials first.")

    if numeric_cols is None:
        numeric_cols = cfg.factors_num

    if labels_boxplot is None:
        labels_boxplot = {
        "Installation_year": "Installation year (-)",
        "Diameter": "Diameter (mm)",
        "Wet_peak_flow_rate": "Wet max. flow rate (L/s)",
        "Pipe_length": "Length (m)",
        "Depth": "Depth (m)",
        "Slope": "Slope (%)",
        }
    # --- Grid layout ---
    n_plots = len(numeric_cols)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        series_all = []

        for mat in materials:
            pop = df_population[df_population[material_col] == mat][col].dropna()
            samp = df_sample[df_sample[material_col] == mat][col].dropna()
            series_all += [pop, samp]

            color = colors_materials.get(mat, "#999999")
            if len(pop) > 0:
                xp, yp = ecdf(pop)
                ax.plot(xp, yp, drawstyle="steps-post", color=color, linestyle="-", lw=1.8, alpha=0.95,
                        label=f"{population_label} • {mat}")
            if len(samp) > 0:
                xs, ys = ecdf(samp)
                ax.plot(xs, ys, drawstyle="steps-post", color=color, linestyle="--", lw=1.8, alpha=0.95,
                        label=f"{sample_label} • {mat}")

        # Population vs sample legend
        style_legend = [
            Line2D([0], [0], color="k", lw=2.5, linestyle="-", label=population_label),
            Line2D([0], [0], color="k", lw=2.5, linestyle="--", label=sample_label)
        ]
        ax.legend(handles=style_legend, fontsize=14, loc="lower right", frameon=True)

        ax.set_xlabel(labels_boxplot.get(col, col), fontsize=14)
        ax.set_ylabel("ECDF", fontsize=14)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        ax.text(
            0.5, 1.02,
            f"ECDF - {labels_boxplot.get(col, col)}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold"
        )

        include_zero = col in include_zero_cols
        set_percentile_xlim(ax, series_all, include_zero, 0, 99, col_name=col)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    # Material legend
    material_legend = [
        Line2D([0], [0], color=colors_materials.get(m, "#999999"), lw=4, label=m)
        for m in materials
    ]
    fig.legend(handles=material_legend, title="Material",
               fontsize=13, title_fontsize=15,
               loc="center left", bbox_to_anchor=(1, 0.5), frameon=True, markerscale=5)

    plt.tight_layout(rect=[0, 0, 1, 1], h_pad=2, w_pad=1.5)
    plt.show()


def analyze_ecdf_by_material(
    df_population: pd.DataFrame,
    df_sample: pd.DataFrame,
    material_col: str = "Material",
    numeric_cols: list | None = None,
    materials: list | None = None,
    colors_materials: dict | None = None
):
    """
    Master function: plots ECDFs and returns KS & Probability Superiority table.
    Uses global selected_materials and colors_materials if not provided.
    Parameters
    ----------
    df_population : pandas.DataFrame
        DataFrame containing the full population data.
        It must include the material identifier column and the numeric variables
        to be analyzed. This dataset is used as the reference distribution.

    df_sample : pandas.DataFrame
        DataFrame containing the sample or subset data to be compared against
        the population distribution. It must have the same structure as
        `df_population` for the selected variables.

    material_col : str, default "Material"
        Name of the column that identifies the pipe material in both input
        DataFrames.

    numeric_cols : list or None, default None
        List of numeric variables for which ECDFs, KS statistics, and probability
        superiority are computed. If None, the function uses `cfg.factors_num`
        defined in the configuration.

    materials : list or None, default None
        List of materials to include in the analysis and plots, and their order.
        If None, the function defaults to `cfg.selected_materials`.

    colors_materials : dict or None, default None
        Dictionary mapping each material to a color used in the ECDF plots.
        If None, the function defaults to `cfg.colors_materials`.

    Returns
    -------
    pandas.DataFrame
        A table with variables as rows and materials as columns, containing
        Probability Superiority values for each variable–material combination.
        The columns are ordered according to `cfg.selected_materials`.
    """
    # --- Validate inputs ---
    if df_population is None or df_population.empty:
        print("df_population is missing or empty.")
        return None
    if df_sample is None or df_sample.empty:
        print("df_sample is missing or empty.")
        return None

    if numeric_cols is None:
        numeric_cols = cfg.factors_num
    if materials is None:
        materials = cfg.selected_materials
    if colors_materials is None:
        colors_materials = cfg.colors_materials

    if materials is None or colors_materials is None:
        raise ValueError("Please initialize cfg.selected_materials and cfg.colors_materials first.")

    # --- Plot ECDFs ---
    plot_ecdf_by_material_overlay(
        df_population, df_sample,
        material_col=material_col,
        numeric_cols=[c for c in numeric_cols if c != "Dry_peak_flow_rate"],
        materials=materials,
        colors_materials=colors_materials
    )

    # --- Compute Probability Superiority table ---
    summary = compute_ps_table(df_population, df_sample, material_col, numeric_cols, materials)
    return summary


## Boxplots ------------------------------------------------------------------------------------------------------------
def plot_boxplots_grid(
    df: pd.DataFrame,
    factors_num: list | None = None,
    group_col: str = "MATERIAL",
    selected_materials: list | None = None,
    colors_materials: dict | None = None,
    labels_boxplot: dict | None = None,
):
    """
    Creates a grid of boxplots for numeric variables, grouped by material
    (or another categorical column).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    factors_num : list of str or None, default None
        Numeric variables to plot. If None, uses cfg.factors_num.
    group_col : str, default "MATERIAL"
        Column used to group the data on the x-axis.
    selected_materials : list or None, default None
        Order of categories on the x-axis. If None, inferred from data.
    colors_materials : dict or None, default None
        Dictionary mapping materials to colors.
    labels_boxplot : dict or None, default None
        Dictionary mapping variable names to nicer y-axis labels.
    """

    # --- Defaults from config --- #
    if factors_num is None:
        factors_num = cfg.factors_num

    if selected_materials is None:
        selected_materials = cfg.selected_materials

    if colors_materials is None:
        colors_materials = cfg.colors_materials

    # --- Default labels --- #
    if labels_boxplot is None:
        labels_boxplot = {
            "Installation_year": "Installation year (-)",
            "Diameter": "Diameter (mm)",
            "Wet_peak_flow_rate": "Wet max. flow rate (L/s)",
            "Pipe_length": "Length (m)",
            "Depth": "Depth (m)",
            "Slope": "Slope (%)",
        }

    # --- Validation --- #
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty or None.")

    if not factors_num:
        raise ValueError("factors_num is empty. Please define cfg.factors_num or pass it explicitly.")

    if group_col not in df.columns:
        raise ValueError(f"Grouping column '{group_col}' not found in DataFrame.")

    for col in factors_num:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' from factors_num not found in DataFrame.")

    # --- Grid layout --- #
    n_vars = len(factors_num)
    n_cols = math.ceil(math.sqrt(n_vars))
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows)
    )
    axes = axes.flatten()

    # --- Validate and filter materials --- #
    materials_in_df = df[group_col].dropna().unique().tolist()

    valid_materials = [m for m in selected_materials if m in materials_in_df]
    invalid_materials = [m for m in selected_materials if m not in materials_in_df]

    # Warn user (do not stop execution)
    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plots):")
        for m in invalid_materials:
            print(f"   - {m}")

    # If none are valid → fallback
    if not valid_materials:
        print("\n⚠️ No valid materials selected → using all available materials")
        valid_materials = sorted(materials_in_df)

    # Filter dataframe
    df = df[df[group_col].isin(valid_materials)].copy()

    # --- Boxplots --- #
    for i, col in enumerate(factors_num):
        sns.boxplot(
            data=df,
            x=group_col,
            y=col,
            ax=axes[i],
            palette=colors_materials,
            order=valid_materials,
            width=0.8,
        )

        axes[i].set_ylabel(labels_boxplot.get(col, col))
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=0)

        axes[i].set_title(
            f"Distribution of {labels_boxplot.get(col, col)}",fontweight="bold"
        )

        if axes[i].get_legend():
            axes[i].legend_.remove()

    # --- Remove empty axes --- #
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(h_pad=2)
    plt.show()