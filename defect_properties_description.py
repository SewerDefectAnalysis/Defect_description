import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import config as cfg
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable

def create_material_subplot_layout(valid_materials, width=4, height=9, max_cols=4):
    import numpy as np

    n = len(valid_materials)
    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / max_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * width, n_rows * height),
        constrained_layout=True
    )

    axes = np.array(axes).reshape(-1)

    return fig, axes, n_cols

#%%
## Size-----------------------------------------------------------------------------------------------------------------
def plot_defect_size_bars(
    df: pd.DataFrame,
    selected_materials: list | None = None,
    valid_sizes: list = ["S", "M", "L"],
    min_count: int = 15,
    exclude_defects: list | None = None,
):
    """
    Plot horizontal stacked bar charts showing the percentage distribution
    of defect sizes for each selected pipe material.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing defect information. It must include the
        columns 'Material', 'Quantification', and 'Defect_code'.

    selected_materials : list of str or None, optional
        List of pipe materials to include in the plot. If None, the default
        materials defined in cfg.selected_materials are used.

    valid_sizes : list of str, optional
        Defect size categories to include in the stacked bars.
        Default is ['S', 'M', 'L'].

    min_count : int, optional
        Minimum number of observations required for a defect type to be
        considered valid. Defects with fewer occurrences are shown as
        "insufficient data". Default is 15.

    exclude_defects : list of str or None, optional
        List of defect codes to exclude from the analysis.
        If None, ['TM', 'PX'] is used by default.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the generated plots.
    """

    # Verification of required data
    if selected_materials is None:
        selected_materials = cfg.selected_materials

    if not selected_materials:
        raise ValueError("No materials provided for plotting.")

    # --- Validate materials (soft) --- #
    materials_in_df = df["Material"].dropna().unique().tolist()

    valid_materials = [m for m in selected_materials if m in materials_in_df]
    invalid_materials = [m for m in selected_materials if m not in materials_in_df]

    # Print warning
    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plots):")
        for m in invalid_materials:
            print(f"   - {m}")

    # Fallback if none valid
    if not valid_materials:
        print("\n⚠️ No valid materials selected → using all available materials")
        valid_materials = sorted(materials_in_df)

    df = df[df["Material"].isin(valid_materials)].copy()

    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    required_cols = {"Material", "Quantification", "Defect_code"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    # --- Filter valid sizes ---
    df_use = (
        df[
            df["Quantification"].isin(valid_sizes) &
            ~df["Defect_code"].isin(["PX", "TM"])
            ].copy()
    )

    # --- Set defect order ---
    defect_order = df_use["Defect_code"].value_counts().index.tolist()

    n = len(valid_materials)

    fig, axes, n_cols = create_material_subplot_layout(valid_materials)

    if n == 1:
        axes = [axes]

    colors = ["#ffe5b4", "#ffb347", "#ff7f0e", "lightgray"]

    # --- Calculate the percentage of sizes for defect type and material ---
    for i, material in enumerate(valid_materials):
        ax = axes[i]
        df_mat = df_use[df_use["Material"] == material]

        if df_mat.empty:
            ax.set_axis_off()
            continue
        col = i % n_cols

        if col == 0:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=True,labelsize=13)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', labelleft=False)

        counts = (
            df_mat
            .groupby(["Defect_code", "Quantification"])
            .size()
            .reset_index(name="count")
        )

        totals = counts.groupby("Defect_code")["count"].transform("sum")
        counts["percentage"] = counts["count"] / totals * 100

        pivot = (
            counts
            .pivot(index="Defect_code", columns="Quantification", values="percentage")
            .fillna(0)
            .reindex(columns=valid_sizes)
        )

        # --- Identify defect types  with insufficient data ---
        defect_totals = df_mat["Defect_code"].value_counts()
        invalid_defects = defect_totals[defect_totals < min_count].index

        missing_defects = [d for d in defect_order if d not in pivot.index]
        missing_df = pd.DataFrame(0, index=missing_defects, columns=valid_sizes)

        full = pd.concat([pivot, missing_df]).loc[defect_order[::-1]]

        full["gray"] = 0
        full.loc[invalid_defects, "gray"] = 100
        full.loc[missing_defects, "gray"] = 100
        full.loc[invalid_defects, valid_sizes] = 0

        # --- Plot ---
        full_plot = full[valid_sizes + ["gray"]]
        full_plot.plot(
            kind="barh",
            stacked=True,
            color=colors,
            ax=ax,
            legend=False,
            edgecolor="white",
            linewidth=1.2,
            width=0.9
        )


        # --- Adjust plot ---
        for patch in ax.patches:
            patch.set_height(1.0)

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(full) - 0.5)
        ax.set_title(material, fontsize=14)
        defects = full.index.tolist()
        ax.set_yticklabels([f"{d} ─" for d in defects])
        ax.tick_params(axis='x', labelsize=12)

    for j in range(len(valid_materials), len(axes)):
        axes[j].axis('off')

    # --- Add legend and axis labels ---
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    labels = ["S", "M", "L", f"Insufficient data (n < {min_count})"]

    fig.legend(
        handles,
        labels,
        title="Defect size",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
    )

    fig.text(0.04, 0.5, "Defect type", va="center", rotation="vertical", fontsize=13)
    fig.text(0.5, -0.03, "Percentage (%)", ha="center", fontsize=13)

    plt.tight_layout(rect=[0.06, 0, 1, 0.98])
    plt.show()

    return None

#%%
## Longitudinal distance------------------------------------------------------------------------------------------------
def plot_defect_heatmaps_longitudinal(
    df: pd.DataFrame,
    group_col: str = 'Defect_code',
    distance_col: str = 'Longitudinal_distance_normalized',
    min_count: int = 15,
    selected_materials: list | None = None,
):
    """
    Plot longitudinal-position heatmaps of defects for each selected material.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format DataFrame containing defect observations. Must include
        'Material', the grouping column, and the longitudinal distance column.

    group_col : str, default 'Defect_code'
        Column identifying defect types (rows of the heatmap).

    distance_col : str, default 'Longitudinal_distance_normalized'
        Column with normalized longitudinal position along the pipe (0–1).

    min_count : int, default 0
        Minimum number of observations per defect *within each material*.
        Defects below this threshold are shown in gray.

    selected_materials : list or None, default None
        Materials to plot. If None, cfg.selected_materials is used.

    Returns
    -------
    None
        Displays the heatmaps.
    """

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Input DataFrame is empty or invalid.")

    if selected_materials is None:
        selected_materials = cfg.selected_materials

    if not selected_materials:
        raise ValueError("No materials provided for plotting.")

    # --- Validate materials (soft) --- #
    materials_in_df = df["Material"].dropna().unique().tolist()

    valid_materials = [m for m in selected_materials if m in materials_in_df]
    invalid_materials = [m for m in selected_materials if m not in materials_in_df]

    # Print warning
    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plots):")
        for m in invalid_materials:
            print(f"   - {m}")

    # Fallback if none valid
    if not valid_materials:
        print("\n⚠️ No valid materials selected → using all available materials")
        valid_materials = sorted(materials_in_df)

    df = df[df["Material"].isin(valid_materials)].copy()

    required = {'Material', group_col, distance_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df['LD_PERC'] = df[distance_col] * 100

    bins = np.arange(0, 110, 10)
    df['LD_PERC_BIN'] = pd.cut(
        df['LD_PERC'], bins=bins, right=False, include_lowest=True
    )
    bin_labels = [f"{int(b.left)}-{int(b.right)}" for b in df['LD_PERC_BIN'].cat.categories]

    defect_order = df[group_col].value_counts().index.tolist()
    if not defect_order:
        raise ValueError("No defect types found in the DataFrame.")

    base_cmap = plt.get_cmap("Blues", 256)
    base_colors = base_cmap(np.linspace(0, 1, 256))
    gray_color = np.array([[0.85, 0.85, 0.85, 1.0]])
    final_cmap = ListedColormap(np.vstack([base_colors, gray_color]))

    bounds = np.concatenate([np.linspace(0, 1, 256), [998, 1000]])
    norm = BoundaryNorm(bounds, final_cmap.N)

    fig, axes, n_cols = create_material_subplot_layout(valid_materials)

    if len(valid_materials) == 1:
        axes = np.array([axes])

    hm_for_cbar = None

    for i, (material, ax) in enumerate(zip(valid_materials, axes)):
        df_mat = df[df['Material'] == material]

        if df_mat.empty:
            ax.axis('off')
            ax.set_title(f"{material}\n(no data)")
            continue

        freq = (
            df_mat
            .groupby([group_col, 'LD_PERC_BIN'], observed=False)
            .size()
            .reset_index(name='FREQ')
        )

        table = (
            freq
            .pivot(index=group_col, columns='LD_PERC_BIN', values='FREQ')
            .fillna(0)
            .reindex(defect_order, fill_value=0)
        )

        table_norm = (
            table
            .div(table.max(axis=1).replace(0, np.nan), axis=0)
            .fillna(0)
        )

        counts = df_mat[group_col].value_counts()
        for defect in defect_order:
            if counts.get(defect, 0) < min_count:
                table_norm.loc[defect] = 999

        table_norm.columns = bin_labels

        if table_norm.empty:
            ax.axis('off')
            ax.set_title(f"{material}\n(no eligible defects)")
            continue

        hm = sns.heatmap(
            table_norm,
            cmap=final_cmap,
            norm=norm,
            ax=ax,
            cbar=False,
            linewidths=0,
        )
        hm_for_cbar = hm

        defects = table_norm.index.tolist()
        ax.set_yticklabels([f"{d} ─" for d in defects])

        ax.set_title(material, fontsize=13)
        ax.set_xlabel("")
        ax.set_xticks(np.arange(len(bin_labels)) + 0.5)
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=12)

        col = i % n_cols

        # --- Y axis ---
        if col == 0:
            ax.set_ylabel("Defect type", fontsize=15)
            ax.tick_params(axis='y', labelsize=12, rotation=0, pad=0, length=0)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, labelleft=False)

        # --- X axis ---
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=12)
        ax.tick_params(axis='x', labelsize=12)

        # for y in range(1, len(table_norm)):
        #     ax.hlines(y, *ax.get_xlim(), colors="white", linewidth=0.7)

        for y in range(1, len(table_norm)):
            ax.axhline(y, color="white", lw=0.9, alpha=0.6)

    for j in range(len(valid_materials), len(axes)):
        axes[j].axis('off')

    if hm_for_cbar is not None:
        mappable = hm_for_cbar.collections[0]
    else:
        mappable = ScalarMappable(norm=norm, cmap=final_cmap)
        mappable.set_array(np.array([]))

    cax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Normalized frequency", fontsize=12)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="major",labelsize=9)

    fig.supxlabel("Pipe length (%)", fontsize=15)
    plt.show()


#%%
## Extent --------------------------------------------------------------------------------------------------------------
def plot_defect_density_extent_horizontal(
    df,
    group_col='Defect_code',
    distance_col='Longitudinal_distance_normalized',
    length_col='Defect_length',
    materials=None,
    min_count=10,
    resolution=0.1,
):
    """
    Plot longitudinal defect density heatmaps (extent-based) for multiple materials
    arranged horizontally.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-format DataFrame containing defect observations. Must include:
        - 'Material' : pipe material/category,
        - group_col : defect identifier,
        - distance_col : defect start position along the pipe,
        - length_col : defect longitudinal extent.
    group_col : str, default 'Defect_code'
        Column identifying the defect type. Each defect is plotted as one heatmap row.
    distance_col : str, default 'Longitudinal_distance_normalized'
        Column with the defect **start position** along the pipe (typically normalized 0–1).
    length_col : str, default 'Defect_length'
        Column with the **length/extent** of the defect along the pipe
        (same units as `distance_col`).
    materials : list-like or None, default None
        Ordered list of pipe materials to plot. If None, uses `cfg.selected_materials`.
    min_count : int, default 10
        Minimum number of defects required (per material and defect type) to be displayed
        with color. Defects below this threshold are rendered in gray.
    resolution : float, default 0.1
        Bin width along the pipe length axis (same units as `distance_col`).

    """

    # --- Defaults ---
    if materials is None:
        materials = cfg.selected_materials

    if not materials:
        print("No materials provided.")
        return

    # --- Validate materials --- #
    materials_in_df = df["Material"].dropna().unique().tolist()

    valid_materials = [m for m in materials if m in materials_in_df]
    invalid_materials = [m for m in materials if m not in materials_in_df]

    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plots):")
        for m in invalid_materials:
            print(f"   - {m}")

    if not valid_materials:
        print("\n⚠️ No valid materials selected → using all available materials")
        valid_materials = sorted(materials_in_df)



    # --- ERROR CHECKS ---
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("Input DataFrame is missing or empty.")
        return

    required = {group_col, distance_col, length_col, 'Material'}
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Missing required columns: {sorted(missing)}")
        return

    end_sum = df[distance_col] + df[length_col]
    if (end_sum > 1).any():
        print("Some defects exceed 100% of pipe length (distance + length > 1).")
        print(f"Number of problematic rows: {(end_sum > 1).sum()}")
        return

    # --- Filter valid rows ---
    mask = df[length_col].notna() & df[distance_col].notna()
    df = df.loc[mask].copy()
    df = df[df["Material"].isin(valid_materials)].copy()

    # --- Global defect order ---
    ordered_defects = df[group_col].value_counts().index.tolist()
    hm_for_cbar = None

    # --- Bin definition ---
    max_position = (df[distance_col] + df[length_col]).max()
    grid = np.arange(0, max_position + resolution, resolution)
    segment_labels = [
        f'{int(grid[i] * 100)}-{int(grid[i + 1] * 100)}'
        for i in range(len(grid) - 1)
    ]

    # --- Colormap ---
    base_cmap = plt.get_cmap('Purples', 256)
    base_colors = base_cmap(np.linspace(0, 1, 256))
    gray_color = np.array([[0.85, 0.85, 0.85, 1.0]])
    final_cmap = ListedColormap(np.vstack([base_colors, gray_color]))
    bounds = np.concatenate([np.linspace(0, 1, 256), [998, 1000]])
    norm = BoundaryNorm(bounds, final_cmap.N)

    # --- Figure ---
    fig, axes, n_cols = create_material_subplot_layout(valid_materials)

    if len(valid_materials) == 1:
        axes = [axes]

    for i, material in enumerate(valid_materials):
        ax = axes[i]
        df_mat = df[df['Material'] == material]

        heatmap_data = pd.DataFrame(
            0, index=ordered_defects, columns=segment_labels
        )

        for defect in ordered_defects:
            df_def = df_mat[df_mat[group_col] == defect]
            for _, row in df_def.iterrows():
                start = row[distance_col]
                end = start + row[length_col]
                for j in range(len(grid) - 1):
                    if grid[j] < end and grid[j + 1] > start:
                        heatmap_data.loc[defect, segment_labels[j]] += 1

        heatmap_norm = heatmap_data.div(
            heatmap_data.max(axis=1), axis=0
        )

        defect_counts = df_mat[group_col].value_counts()
        for defect in ordered_defects:
            if defect_counts.get(defect, 0) < min_count:
                heatmap_norm.loc[defect, :] = 999
        col = i % n_cols

        hm = sns.heatmap(
            heatmap_norm,
            cmap=final_cmap,
            norm=norm,
            ax=ax,
            xticklabels=True,
            yticklabels=(col == 0),
            cbar=False
        )



        # --- Y axis ---
        if col == 0:
            defects = heatmap_norm.index.tolist()
            ax.set_yticklabels([f"{d} ─" for d in defects])

            ax.set_ylabel("Defect type", fontsize=15)
            ax.tick_params(axis='y', labelsize=12, rotation=0, pad=0, length=0)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.set_title(material, fontsize=13)
        ax.tick_params(axis='x', labelsize=10, rotation=45)
        ax.tick_params(axis='y', labelsize=12, rotation=0,pad=0,length=0)

        for y in range(1, len(heatmap_norm)):
            ax.hlines(
                y,
                xmin=0,
                xmax=len(segment_labels),
                colors="white",
                linewidth=1
            )
    for j in range(len(valid_materials), len(axes)):
        axes[j].axis('off')

    # --- Colorbar ---
    from matplotlib.cm import ScalarMappable
    mappable = ScalarMappable(norm=norm, cmap=final_cmap)
    mappable.set_array([])

    cax = fig.add_axes([1, 0.15, 0.015, 0.75])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Normalized frequency", fontsize=11)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="major",labelsize=9)

    fig.supxlabel("Pipe length (%)", fontsize=12)
    plt.show()


## Clock reference position --------------------------------------------------------------------------------------------------------------
#%%

def plot_defect_position_heatmaps(
    df: pd.DataFrame,
    materials=None,
    min_count: int = 15
):
    """
    Plot circumferential (clock-reference) defect position heatmaps for multiple materials.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing defect observations. Must include the columns:
        'Material', 'Circumferential_start', 'Circumferential_end', and 'Defect_code'.
        Each row represents one defect instance.
    materials : list-like or None, default None
        Ordered list of pipe materials to include in the plot. One subplot is created
        per material. If None, uses `cfg.selected_materials`.
    min_count : int, default 15
        Minimum number of occurrences required for a defect type (within a material)
        to be considered representative. Defects below this threshold are rendered
        in gray in the heatmap.
    """

    # --- Defaults ---
    if materials is None:
        materials = cfg.selected_materials

    # --- Validate materials (soft) --- #
    materials_in_df = df["Material"].dropna().unique().tolist()

    valid_materials = [m for m in materials if m in materials_in_df]
    invalid_materials = [m for m in materials if m not in materials_in_df]

    if invalid_materials:
        print("\n⚠️ Materials not present in data (ignored in plots):")
        for m in invalid_materials:
            print(f"   - {m}")

    if not valid_materials:
        print("\n⚠️ No valid materials selected → using all available materials")
        valid_materials = sorted(materials_in_df)

    df = df[df["Material"].isin(valid_materials)].copy()

    # --- Validation of the required information ---
    required_cols = ['Material', 'Circumferential_start', 'Circumferential_end', 'Defect_code']
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty or missing.")
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the DataFrame.")

    # --- Helper: compute clock-position range ---
    def position_range(start, end):

        if (pd.isna(start) or start == 0) and (pd.isna(end) or end == 0):
            return []
        if pd.isna(start) or start == 0:
            return []
        start = int(start)
        if pd.isna(end) or end == 0:
            return [start]
        end = int(end)
        if start < end:
            return list(range(start, end + 1))
        elif start == end:
            return list(range(1, end + 1))
        else:
            return list(range(start, 13)) + list(range(1, end + 1))

    # --- Apply position_range to all rows ---
    df['Position'] = df.apply(lambda r: position_range(r['Circumferential_start'], r['Circumferential_end']), axis=1)

    # --- Keep only defects with at least one valid position ---
    df_def_pos = df[df['Position'].map(len) > 0].copy()
    df_def_pos['Defect_code'] = df_def_pos['Defect_code'].astype(str)

    # --- Defect order based on valid positions only ---
    order_defects_pos = df_def_pos['Defect_code'].value_counts().index.tolist()

    # --- Figure setup ---
    n = len(valid_materials)
    fig, axes, n_cols = create_material_subplot_layout(valid_materials)

    if n == 1:
        axes = [axes]

    # --- Colormap (gray for low-frequency defects) ---
    base_cmap = plt.get_cmap('Greens', 256)
    base_colors = base_cmap(np.linspace(0, 1, 256))
    gray_color = np.array([[0.85, 0.85, 0.85, 1.0]])
    final_cmap = ListedColormap(np.vstack([base_colors, gray_color]))
    bounds = np.concatenate([np.linspace(0, 1, 256), [998, 1000]])
    norm = BoundaryNorm(boundaries=bounds, ncolors=257)


    # --- Plot per material ---
    for i, (material, ax) in enumerate(zip(valid_materials, axes)):
        col = i % n_cols
        df_mat = df_def_pos[df_def_pos['Material'] == material].copy()
        if df_mat.empty:
            ax.axis('off')
            ax.set_title(f"{material}\n(no data)")
            continue

        # Explode positions
        df_exploded = df_mat.explode("Position")

        # Frequency table
        freq = (
            df_exploded
            .groupby(["Defect_code", "Position"])
            .size()
            .reset_index(name="FREQ")
        )

        # Pivot table
        position_table = (
            freq
            .pivot(index="Defect_code", columns="Position", values="FREQ")
            .fillna(0)
        )

        # Normalize
        position_norm = position_table.div(position_table.max(axis=1), axis=0).fillna(0)

        # Mark defects below min_count in gray
        defect_counts = df_mat['Defect_code'].value_counts().to_dict()
        for defect in order_defects_pos:
            if defect_counts.get(defect, 0) < min_count:
                if defect not in position_norm.index:
                    position_norm.loc[defect] = 0
                position_norm.loc[defect, :] = 999

        # Reindex to ensure order, remove empty rows
        position_norm = position_norm.reindex(order_defects_pos).fillna(0)

        # --- Heatmap ---
        hm = sns.heatmap(
            position_norm,
            cmap=final_cmap,
            norm=norm,
            ax=ax,
            cbar=False,
            annot=False,
            linewidths=0
        )

        defects = position_norm.index.tolist()
        y_pos = np.arange(len(defects)) + 0.5
        ax.set_yticks(y_pos)

        if col == 0:
            ax.set_yticklabels([f"{d} ─" for d in defects])
            ax.set_ylabel("Defect type", fontsize=15)
            ax.tick_params(axis='y', labelsize=12, rotation=0, pad=0, length=0)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis='y', left=False, labelleft=False)

        ax.tick_params(axis="x", labelsize=11, rotation=45)

        for y in range(1, len(position_norm)):
            ax.axhline(y, color="white", lw=0.9, alpha=0.6)

    for j in range(len(valid_materials), len(axes)):
        axes[j].axis('off')

    # --- Colorbar ---
    from matplotlib.cm import ScalarMappable
    mappable = ScalarMappable(norm=norm, cmap=final_cmap)
    mappable.set_array([])

    cax = fig.add_axes([1, 0.15, 0.015, 0.75])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Normalized frequency", fontsize=11)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="major",labelsize=9)
    plt.show()
