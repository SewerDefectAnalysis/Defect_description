import config as cfg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Mapping, Sequence, Dict, List, Tuple, Optional

def plot_defect_counts_per_pipe(
    df_defects: pd.DataFrame,
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """
    Plot the average number of defects per pipe, grouped by material.

    Parameters
    ----------
    df_defects : pd.DataFrame
        DataFrame containing defect records.
        Must include 'Material' and 'Pipe_ID' columns.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the plot on. If None, a new figure and axis are created.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the bar plot.

    Raises
    ------
    ValueError
        If required columns are missing or input DataFrame is empty.
    """

    # --- Axis setup ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    # --- Validate input ---
    if df_defects is None or df_defects.empty:
        raise ValueError("df_defects is empty or None.")

    required_cols = {"Material", "Pipe_ID"}
    missing = required_cols - set(df_defects.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in df_defects: {missing}"
        )

    # --- Count defects per pipe ---
    defects_per_pipe = (
        df_defects
        .groupby(["Pipe_ID", "Material"], observed=False)
        .size()
        .reset_index(name="defect_count")
    )

    # --- Average defects per pipe by material ---
    df_group = (
        defects_per_pipe
        .groupby("Material", observed=False)["defect_count"]
        .mean()
        .reset_index()
    )

    # --- Enforce material order from config ---
    # Get valid materials (present in df)
    valid_materials = [
        m for m in cfg.selected_materials
        if m in df_group["Material"].unique()
    ]

    # Optional warning
    invalid_materials = [
        m for m in cfg.selected_materials
        if m not in df_group["Material"].unique()
    ]

    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plot):")
        for m in invalid_materials:
            print(f"   - {m}")

    # Apply categorical order ONLY with valid materials
    df_group["Material"] = pd.Categorical(
        df_group["Material"],
        categories=valid_materials,
        ordered=True
    )

    # --- Plot ---
    sns.barplot(
        data=df_group,
        x="Material",
        y="defect_count",
        palette=cfg.colors_materials,
        ax=ax,
        legend=False,
        width=0.8
    )

    # --- Styling ---
    ax.set_xlabel("")
    ax.set_ylabel("Defects per pipe", fontsize=14)
    #ax.set_title("(b) Average defects per pipe", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # --- Annotate bars ---
    for bar in ax.patches:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.1f}",
                (bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=12.5
            )
    plt.tight_layout()
    plt.show()
    return ax


## Distribution of type of defects -------------------------------------------------------------------------------------
def prepare_defect_data(df_defects_filtered, threshold=None):
    """
    Prepares defect data for plotting.
    Splits defects into those >= threshold and those < threshold.
    Returns pivot tables for both cases.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format table with (at least) columns:
          - 'Material'   : material/category of the pipe
          - 'Defect_code'  : defect code/name
          - 'Defects'   : count of defects (numeric)
    threshold : float, default 10
        Minimum percentage (0–100) a defect must contribute within a material
        to be treated as "frequent" (and get its own column in df_keep_pivot).

    Returns
    -------
    df_keep_pivot : pandas.DataFrame
        Wide table (Material × {frequent defects + 'Others'}) with percentages
        that sum to ~100% by row (subject to data completeness).
    df_rare_pivot : pandas.DataFrame
        Wide table (Material × rare defects) with percentages *within the rare subset*
        that sum to ~100% by row only for materials that have rare defects.
    """

    # --- validate ---
    if df_defects_filtered is None or not isinstance(df_defects_filtered, pd.DataFrame) or df_defects_filtered.empty:
        raise ValueError("Defects DataFrame is missing or empty.")
    required = {"Material", "Defect_code"}
    missing = required - set(df_defects_filtered.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df_defects_filtered[df_defects_filtered["Material"].notna() & df_defects_filtered["Defect_code"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with non-null Material and Defect_code after filtering.")

    # --- counts per (Material, Defect_code) ---
    g = (df.groupby(["Material", "Defect_code"], observed=False)
           .size().reset_index(name="count"))
    g = g[~g["Defect_code"].astype(str).str.fullmatch(r"\d+")].copy()

    if threshold is None:
        totals = g.groupby("Material", observed=False)["count"].transform("sum")
        g["defect_percentage"] = g["count"] / totals * 100

        g["Material"] = pd.Categorical(g["Material"], categories=cfg.selected_materials, ordered=True)
        g = g.sort_values("Material")
        df_keep_pivot = (g.pivot(index="Material", columns="Defect_code", values="defect_percentage")
                           .fillna(0))
        df_keep_pivot = df_keep_pivot.loc[:, df_keep_pivot.sum().sort_values(ascending=False).index]
        df_rare_pivot = pd.DataFrame(index=df_keep_pivot.index)
        return df_keep_pivot, df_rare_pivot

    # --- threshold path---
    g["total"] = g.groupby("Material", observed=False)["count"].transform("sum")
    g["defect_percentage"] = g["count"] / g["total"] * 100

    g["Material"] = pd.Categorical(g["Material"], categories=cfg.selected_materials, ordered=True)
    g = g.sort_values("Material")

    keep_df = g[g["defect_percentage"] >= float(threshold)].copy()
    rare_df = g[g["defect_percentage"] <  float(threshold)].copy()

    df_keep_pivot = (keep_df.pivot(index="Material", columns="Defect_code", values="defect_percentage")
                       .fillna(0))
    #--- add Others---
    if not rare_df.empty:
        others_sum = rare_df.groupby("Material", observed=False)["defect_percentage"].sum()
        df_keep_pivot["Others"] = others_sum
    else:
        df_keep_pivot["Others"] = 0.0

    df_keep_pivot = df_keep_pivot.loc[:, df_keep_pivot.sum().sort_values(ascending=False).index]

    rare_df["others_total"] = rare_df.groupby("Material", observed=False)["count"].transform("sum")
    rare_df["defect_percentage_others"] = rare_df["count"] / rare_df["others_total"] * 100
    df_rare_pivot = (rare_df.pivot(index="Material", columns="Defect_code", values="defect_percentage_others")
                           .fillna(0))

    return df_keep_pivot, df_rare_pivot

def plot_defects_stacked_with_others(
    df,
    threshold=2.5,
    color=None,
    label_min_top=0.5,
    label_min_bottom=0.25,
    bottom_max_labels_per_bar=16,
    show_legend=False,
):
    """
    Create stacked bar charts showing the distribution of defect types,
    separating dominant defects from low-frequency defects grouped as "Others".

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing defect data already aggregated or suitable
        for processing by `prepare_defect_data`. The data must allow computing
        percentage contributions per defect type.

    threshold : float, optional
        Minimum percentage contribution required for a defect type to be shown
        individually in the top plot. Defect types below this threshold are
        grouped into the "Others" category. Default is 2.5.

    color : dict or None, optional
        Dictionary mapping defect codes to colors. If None, an empty mapping is
        used and matplotlib defaults are applied for missing entries.

    label_min_top : float, optional
        Minimum segment height (in percentage units) required to display a text
        label in the top plot. Smaller segments are not labeled. Default is 0.5.

    label_min_bottom : float, optional
        Minimum segment height (in percentage units) required to display a text
        label in the bottom plot. Default is 0.25.

    bottom_max_labels_per_bar : int, optional
        Maximum number of defect labels displayed per bar in the bottom plot.
        This prevents overcrowding when many rare defects are present.
        Default is 16.

    show_legend : bool, optional
        Whether to display a legend for defect types. If False, legends are
        removed from individual axes. Default is False.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the generated stacked bar charts.
    """

    # --- Prepare data ---
    df_keep_pivot, df_rare_pivot = prepare_defect_data(df, threshold)
    color_map = color or {}

    if df_keep_pivot is None or df_keep_pivot.empty:
        raise ValueError("No data available to plot defects.")

    # --- Normalize column names ---
    df_keep_pivot.columns = df_keep_pivot.columns.astype(str).str.strip()
    if df_rare_pivot is not None and not df_rare_pivot.empty:
        df_rare_pivot.columns = df_rare_pivot.columns.astype(str).str.strip()

    # --- Helpers ---
    def _move_others_last(pvt):
        if 'Others' not in pvt.columns:
            return pvt
        cols = [c for c in pvt.columns if c != 'Others'] + ['Others']
        return pvt[cols]

    def _order_cols_desc_total(pvt):
        totals = pvt.sum(axis=0).sort_values(ascending=False)
        return pvt[totals.index]

    def _align_color_map(pvt, cmap):
        return {c: cmap[c] for c in pvt.columns if c in cmap}

    def _hatch_others(ax_plot, pvt):
        if 'Others' not in pvt.columns:
            return
        n_rows = len(pvt.index)
        j = list(pvt.columns).index('Others')
        for i in range(n_rows):
            idx = j * n_rows + i
            if idx < len(ax_plot.patches):
                p = ax_plot.patches[idx]
                p.set_hatch('///')
                p.set_edgecolor('k')
                p.set_linewidth(0.6)

    def _label_segments(
        ax_plot,
        pvt,
        min_h,
        max_labels_per_bar=None,
        dx_frac=0.04,
        dy_min=1.4,
        fontsize=11,
    ):
        n_rows = len(pvt.index)
        patches = ax_plot.patches
        ax_plot.margins(x=0.18)

        for i in range(n_rows):
            candidates = []
            for j, col in enumerate(pvt.columns):
                idx = j * n_rows + i
                if idx >= len(patches):
                    continue
                p = patches[idx]
                h = p.get_height()
                if h > 0 and h >= min_h:
                    candidates.append((h, p, col))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            if max_labels_per_bar is not None:
                candidates = candidates[:max_labels_per_bar]

            candidates = sorted(
                [(p.get_y() + p.get_height() / 2, p, col) for _, p, col in candidates],
                key=lambda x: x[0],
            )

            last_y = -1e9
            for y_mid, p, col in candidates:
                if y_mid - last_y < dy_min:
                    y_mid = last_y + dy_min
                last_y = y_mid

                x_text = p.get_x() + p.get_width() * (1 + dx_frac)
                x_bar = p.get_x() + p.get_width()
                y_bar = p.get_y() + p.get_height()/2

                ax_plot.annotate(
                    col,
                    xy=(x_bar, y_bar),
                    xytext=(x_text, y_mid),
                    textcoords='data',
                    ha='left',
                    va='center',
                    fontsize=fontsize,
                    bbox=dict(
                        boxstyle='round,pad=0.12',
                        facecolor='white',
                        alpha=0.65,
                        edgecolor='none',
                    ),
                    arrowprops=dict(
                        arrowstyle='-',
                        lw=0.8,
                        color='black',
                        shrinkA=0,
                        shrinkB=0
                    ),
                    clip_on=False,
                )

    # --- Reorder columns ---
    df_keep_pivot = _move_others_last(df_keep_pivot)
    if df_rare_pivot is not None and not df_rare_pivot.empty:
        df_rare_pivot = _order_cols_desc_total(df_rare_pivot)

    # --- Align color maps ---
    color_top = _align_color_map(df_keep_pivot, color_map)
    color_bottom = (
        _align_color_map(df_rare_pivot, color_map)
        if df_rare_pivot is not None and not df_rare_pivot.empty
        else None
    )

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 16), sharex=True,
        gridspec_kw={'height_ratios': [1, 1.25]}
    )

    df_keep_pivot.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=color_top,
    )
    ax1.set_xlabel("")
    ax1.set_title('Fraction of Defect Types', fontsize=15)
    ax1.set_ylabel('Percentage (%)', fontsize=13)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', labelsize=13.5)
    _hatch_others(ax1, df_keep_pivot)
    if not show_legend and ax1.get_legend():
        ax1.legend_.remove()

    _label_segments(
        ax1,
        df_keep_pivot,
        min_h=label_min_top,
        fontsize=11,
    )

    if df_rare_pivot is not None and not df_rare_pivot.empty:
        df_rare_pivot.plot(
            kind='bar',
            stacked=True,
            ax=ax2,
            color=color_bottom,
        )
        ax2.set_xlabel("")
        ax2.set_title('Fraction of "Others"', fontsize=15)
        ax2.set_ylabel('Percentage (%)', fontsize=13)
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        ax2.tick_params(axis='both', labelsize=13.5)
        if not show_legend and ax2.get_legend():
            ax2.legend_.remove()

        _label_segments(
            ax2,
            df_rare_pivot,
            min_h=label_min_bottom,
            max_labels_per_bar=bottom_max_labels_per_bar,
            fontsize=11,
        )
    else:
        ax2.set_visible(False)

    if show_legend:
        legend_labels = list(dict.fromkeys(
            list(df_keep_pivot.columns) +
            (list(df_rare_pivot.columns) if df_rare_pivot is not None else [])
        ))
        if 'Others' in legend_labels:
            legend_labels = [c for c in legend_labels if c != 'Others'] + ['Others']

        handles = []
        for lbl in legend_labels:
            patch = Patch(facecolor=color_map.get(lbl, '#cccccc'), label=lbl)
            if lbl == 'Others':
                patch.set_hatch('///')
                patch.set_edgecolor('k')
                patch.set_linewidth(0.6)
            handles.append(patch)

        fig.legend(
            handles,
            legend_labels,
            title='Defect Type',
            loc='upper right',
            bbox_to_anchor=(0.99, 0.99),
            fontsize=12.5,
            title_fontsize=12,
        )

    materials = df_keep_pivot.index.astype(str)

    ax1.set_xticklabels(materials, rotation=45, ha='center')
    ax1.tick_params(axis='x', length=0)

    if ax2.get_visible():
        ax2.set_xticklabels(materials, rotation=0, ha='center')
        ax2.tick_params(axis='x', length=0)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    plt.show()