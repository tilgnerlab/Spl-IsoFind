# SPDX-License-Identifier: MIT
"""
Spatial plotting utilities for long-read spatial transcriptomics data.

This module provides functions to visualize spatial transcriptomics data
as hexbin overlays, read tree-traversal isoform results, and generate
heatmap tiles with accompanying bar plots.
"""
import os
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from matplotlib import cm

import seaborn as sns

def spatial_hexplot(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    varName: str,
    imarray=None,
    celltype: str='',
    region: str='',
    subregion: str='',
    hexsize: int = 120,
    fig_size: tuple = (5, 5),
    ax=None,
    plot_lim: tuple = None,
    alpha: float = 1,
    cmap='viridis',
    show_colorbar: bool = True,
    staining_max: str = 'grey',
    staining_min: str = 'white',
    linewidths: float = 0.1
) -> plt.Axes:
    """
    Overlay a hexbin of relative isoform expression values on a background staining image.

    Extracts values for `varName` from `x`, subsets by `celltype` if provided,
    computes hexbin grid based on spot coordinates, and draws on `ax` or a new figure.

    Parameters
    ----------
    x : pandas.DataFrame
        Feature matrix of shape (n_cells, n_isoforms), with relative expression values between 0 and 1.
    labels : pandas.DataFrame
        Cell metadata with columns ['x','y','first_type','spot_class'].
    varName : str
        Name of the column in `x` to plot.
    imarray : array-like or None
        Background image array; if None, only hexbin is shown.
    celltype : str
        If non-empty, only spots of this `first_type` and singlets are shown.
    region : str
        If non-empty, only spots of this region are shown.
    subregion : str
        If non-empty, only spots of this subregion are shown.
    hexsize : int
        Approximate pixel diameter for hexbin cells.
    fig_size : tuple
        Size of new figure if `ax` is None.
    ax : matplotlib.axes.Axes or None
        Axis to draw on; if None, a new one is created.
    plot_lim : tuple (xmin, xmax, ymin, ymax) or None
        If provided, crops both image and hex positions.
    alpha : float
        Transparency for background image overlay.
    cmap : str or Colormap
        Colormap for hexbin values.
    show_colorbar : bool
        Whether to display a colorbar labeled "Fraction".
    staining_max, staining_min : color spec
        Colors to map background image to.
    linewidths : float
        Width of edges between hex cells.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis containing the hexbin overlay.
    """
    # Subset data
    idx_var = np.where(x.columns == varName)[0][0]
    if celltype != '':
        idx_ct = np.where((labels['first_type'] == celltype) &
                          (labels['spot_class'] == 'singlet'))[0]
        x = x.iloc[idx_ct, idx_var]
        labels = labels.iloc[idx_ct]
    else:
        x = x.iloc[:, idx_var]

    if region != '':
        idx_reg = np.where(labels['region'] == region)[0]
        x = x.iloc[idx_reg]
        labels = labels.iloc[idx_reg]
    
    if subregion != '':
        idx_subreg = np.where(labels['subregion'] == subregion)[0]
        x = x.iloc[idx_subreg]
        labels = labels.iloc[idx_subreg]

    if len(x.dropna()) < 10: #Check whether we have enough cells to plot
        print('Less than 10 cells to plot, skipping.')
        return None

    x = x.dropna()
    labels = labels.loc[x.index]
    
    g_x = int(np.sqrt(3) * ((labels['x'].values.max() -
                             labels['x'].values.min())) / hexsize)
    g_y = int((labels['y'].values.max() -
               labels['y'].values.min()) / hexsize)

    staining_cmap = LinearSegmentedColormap.from_list("white_to_grey", [staining_min, staining_max])
    
    if ax is not None:
        fig = ax.figure
    else:
        fig, ax = plt.subplots(figsize=fig_size)

    if plot_lim is not None:
        if imarray is not None:
            ax.imshow(imarray[plot_lim[2]:plot_lim[3],plot_lim[0]:plot_lim[1]], alpha=alpha, cmap=staining_cmap)
        ax.set_xlim([0, plot_lim[1]-plot_lim[0]])
        ax.set_ylim([0, plot_lim[3]-plot_lim[2]])
        hb = ax.hexbin(x=labels['x']-plot_lim[0],
            y=labels['y']-plot_lim[2],
            C=x, cmap=cmap, gridsize=(g_x, g_y),
            reduce_C_function=np.mean, vmin=0, vmax=1, rasterized=True,
            edgecolors='white',
            linewidths=linewidths)

    else:
        if imarray is not None:
            ax.imshow(imarray, alpha=alpha, cmap=staining_cmap, origin='lower')
        hb = ax.hexbin(x=labels['x'],
                       y=labels['y'],
                       C=x, cmap=cmap, gridsize=(g_x, g_y),
                       reduce_C_function=np.mean, vmin=0, vmax=1, rasterized=True,
                       edgecolors='white',
                       linewidths=linewidths)

    if show_colorbar:
        cb = fig.colorbar(
            hb,
            ax=ax,
            orientation='vertical',
            pad=0.05,
            shrink=0.25,
            aspect=10
        )
        # remove the old (vertical) label if you set one
        cb.set_label(None)
        # set a horizontal title, left-aligned
        txt = cb.ax.set_title("Fraction", pad=10, fontsize=10)
        txt.set_ha("left")
        txt.set_position((-0.05, 1.02))

    title = f"{celltype}\n{varName}" if celltype else varName
    ax.set_title(title)
    ax.set_aspect('equal')
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def spatial_hexplot_sparse(
    x_sparse: csr_matrix,
    labels: pd.DataFrame,
    var_info: pd.DataFrame,
    varName: str,
    **kwargs
) -> plt.Axes:
    """
    Wrapper for `spatial_hexplot` that handles sparse input.

    Extracts the requested isoform column from a CSR sparse matrix and constructs a
    DataFrame for compatibility with `spatial_hexplot`. Only explicitly stored
    values (including explicit zeros) are included in the plot.

    Parameters
    ----------
    x_sparse : scipy.sparse.csr_matrix
        Sparse matrix of relative expression values (cells × isoforms).
        Explicit zeros are retained, missing values are implicit.
    
    labels : pandas.DataFrame
        Cell metadata indexed by cell IDs. Must contain columns ['x', 'y', 'first_type', 'spot_class'].
    
    var_info : pandas.DataFrame
        Isoform metadata with at least a 'Transcript ID' column that matches `x_sparse` columns.
    
    varName : str
        Name of the transcript to plot (must match a value in `var_info['Transcript ID']`).
    
    **kwargs :
        All additional parameters are passed directly to `spatial_hexplot`
        (e.g., imarray, fig_size, cmap, alpha, celltype, ax, etc).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the hexbin overlay.
    """
    
    idx_var = np.where(var_info['Transcript ID'] == varName)[0]

    if len(idx_var) == 0:
        raise ValueError(f"Transcript '{varName}' not found in var_info['Transcript ID'].")

    x_sparse_transcript = x_sparse[:,idx_var]
    idx_notNaN = x_sparse_transcript.tocoo().row
    labels = labels.iloc[idx_notNaN]
    x = pd.DataFrame(x_sparse_transcript[idx_notNaN].toarray(),
                                index = labels.index,
                                columns = var_info.values[idx_var,1])

    ax = spatial_hexplot(x, labels, varName, **kwargs)
    
    return ax

def read_results(
    input_dir: str,
    dataset: str,
    region: str,
    celltype: str
) -> pd.DataFrame:
    """
    Read and summarize isoform traversal results from scisorseqr output.

    Parses `input_dir/dataset/res_scisorseqr/CellTypes_{celltype}_{region}/TreeTraversal_Iso/*/results.csv`,
    counts total tests and significant hits per subdirectory.

    Parameters
    ----------
    input_dir : str
        Base directory for results.
    dataset : str
        Name of dataset subfolder.
    region : str
        Region identifier.
    celltype : str
        Cell type identifier.

    Returns
    -------
    pandas.DataFrame
        Columns: ['reg1','reg2','tested','sig','perc'].
    """

    comp = f'{celltype}_{region}'

    res_dir = f'{input_dir}/{dataset}/res_scisorseqr/CellTypes_{comp}'
    res_dir += '/TreeTraversal_Iso'

    res = pd.DataFrame(columns=['reg1','reg2','tested','sig'])

    for subdir in os.listdir(res_dir):

        res_dir2 = f'{res_dir}/{subdir}/'
        files = np.array(os.listdir(res_dir2))

        try:
            res_file = f"{res_dir2}/{files[np.char.endswith(files, 'results.csv')][0]}"

            x = pd.read_csv(res_file, sep='\t')
            sig = ((x['FDR'] <= 0.05) & (np.abs(x['dPI']) >= 0.1)).sum()
            n = len(x)
            temp = np.array(subdir.split('_'))
            temp = temp[temp != 'ML']
            reg1 = temp[0]
            reg2 = temp[1]

            res2 = pd.DataFrame(data=np.reshape([reg1, reg2, n, sig], (1,4)), 
                                columns=['reg1','reg2','tested','sig'])
            res = pd.concat((res,res2), axis=0)
        except:
            pass

    res = res.astype({"tested": int, "sig": int})
    res['perc'] = 100*res['sig']/res['tested']
    
    return res

def get_countmatrix(
    res: pd.DataFrame,
    region_map: dict = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build symmetric count and percent matrices from pairwise results.

    Parameters
    ----------
    res : pandas.DataFrame
        Output of `read_results`, with columns ['reg1','reg2','sig','perc'].
    region_map : dict, optional
        Mapping from raw region codes to display labels.

    Returns
    -------
    count_matrix, percent_matrix : tuple of pandas.DataFrame
        Square DataFrames indexed and columned by region names.
    """

    # Get sorted list of unique regions
    regions = sorted(set(res['reg1']) | set(res['reg2']))

    # Initialize matrices
    count_matrix = pd.DataFrame(index=regions, columns=regions, dtype=float)
    percent_matrix = pd.DataFrame(index=regions, columns=regions, dtype=float)

    # Fill matrices
    for _, row in res.iterrows():
        r1, r2 = row['reg1'], row['reg2']
        count_matrix.loc[r1, r2] = row['sig']
        count_matrix.loc[r2, r1] = row['sig']
        percent_matrix.loc[r1, r2] = row['perc']
        percent_matrix.loc[r2, r1] = row['perc']

    if region_map != None:
        # Rename both axes of your matrices
        count_matrix.rename(index=region_map, columns=region_map, inplace=True)
        percent_matrix.rename(index=region_map, columns=region_map, inplace=True)

    return count_matrix, percent_matrix

def get_barplot_counts(
    allinfo: pd.DataFrame,
    ct_comp_file: str
) -> pd.Series:
    """
    Sum read counts per group based on a cell-type composition file.

    Parameters
    ----------
    allinfo : pandas.DataFrame
        Raw info with subgroup IDs in column 2.
    ct_comp_file : str
        Path to tab-separated file mapping groups and subgroups.

    Returns
    -------
    pandas.Series
        Index = group names, values = total counts.
    """

    allinfo_grouped = allinfo.groupby(2).size()
    allinfo_grouped = allinfo_grouped.reset_index()
    allinfo_grouped.columns = ['subgroup', 'count']

    ct_comp = pd.read_csv(ct_comp_file, 
                        sep='\t', header=None)
    ct_mapping = pd.concat((ct_comp[[0,1]], 
                            ct_comp[[2,3]].rename(columns={2:0, 3:1})), axis=0).drop_duplicates()
    ct_mapping.columns = ['group', 'subgroups']
    ct_mapping['subgroups'] = ct_mapping['subgroups'].str.split(',')
    mapping_long = ct_mapping.explode('subgroups')
    mapping_long['subgroups'] = mapping_long['subgroups'].str.strip()

    merged = allinfo_grouped.merge(mapping_long, left_on='subgroup', right_on='subgroups', how='left')

    group_counts = merged.groupby('group')['count'].sum()

    return group_counts


# Adaptive text color based on background brightness
def get_text_color(rgba) -> str:
    """
    Choose black or white text based on background brightness.

    Parameters
    ----------
    rgba : tuple
        RGBA color tuple.

    Returns
    -------
    str
        'white' if brightness < 0.5 else 'black'.
    """
    r, g, b, _ = rgba
    brightness = 0.299*r + 0.587*g + 0.114*b
    return 'white' if brightness < 0.5 else 'black'

def plot_heatmap(
    input_dir: str,
    dataset: str,
    region: str,
    celltype: str,
    allinfo: pd.DataFrame,
    region_map: dict = None,
    region_map2: dict = None,
    figsize: tuple = (4, 4),
    cmap_count=cm.Oranges,
    cmap_percent=cm.Blues,
    fontsize_tiles: int = 11,
    fontsize_ticks: int = 12,
    fn: str = None,
    vmax_count=None,
    vmax_perc=None
) -> None:
    """
    Draw a lower-triangle heatmap with % and count triangles plus side barplot.

    Parameters
    ----------
    input_dir : str
        Base results directory.
    dataset : str
        Dataset subfolder.
    region : str
        Region identifier.
    celltype : str
        Cell type identifier.
    allinfo : pandas.DataFrame
        Raw info for barplot counts.
    region_map, region_map2 : dict, optional
        Mapping for region labels.
    figsize : tuple
        Figure size.
    cmap_count, cmap_percent : Colormap
        Colormaps for count and percent triangles.
    fontsize_tiles, fontsize_ticks : int
        Font sizes for tile text and ticks.
    fn : str, optional
        Path to save figure; if None, figure is shown and not saved.
    vmax_count, vmax_perc : float, optional
        Manual maximum for color normalization.
    """

    res = read_results(input_dir=input_dir, 
                       dataset=dataset, 
                       region=region, 
                       celltype=celltype)
    count_matrix, percent_matrix = get_countmatrix(res, region_map)
    read_counts_df = get_barplot_counts(allinfo=allinfo, 
                                        ct_comp_file=f'{input_dir}/ct_files/CellTypes_{celltype}_{region}')
    if region_map2 != None:
        read_counts_df = read_counts_df.rename(index=region_map2)

    regions = count_matrix.index.tolist()
    n = len(regions)

    fig, ax = plt.subplots(figsize=figsize)

    # Color normalization and maps
    if vmax_count == None:
        norm_count = Normalize(vmin=0, vmax=np.ceil(np.nanmax(count_matrix.values)/10)*10)
    else:
        norm_count = Normalize(vmin=0, vmax=vmax_count)
    if vmax_perc == None:
        norm_percent = Normalize(vmin=0, vmax=np.ceil(np.nanmax(percent_matrix.values)))
    else:
        norm_percent = Normalize(vmin=0, vmax=vmax_perc)
        
    # Plotting only lower triangle
    for i in range(n):
        for j in range(n):
            if i < j:
                continue  # skip upper triangle

            x, y = j, i

            # Tile corners (clockwise)
            square = [(x, y), (x+1, y), (x+1, y+1), (x, y+1)]
            lower_tri = [square[0], square[1], square[3]]  # bottom-left triangle
            upper_tri = [square[1], square[2], square[3]]  # top-right triangle

            val_count = count_matrix.iloc[i, j]
            val_percent = percent_matrix.iloc[i, j]

            # Colors
            color_count = cmap_count(norm_count(val_count)) if not pd.isna(val_count) else (1,1,1,1)
            color_percent = cmap_percent(norm_percent(val_percent)) if not pd.isna(val_percent) else (1,1,1,1)

            # Draw triangles
            ax.add_patch(Polygon(lower_tri, facecolor=color_count, edgecolor='white'))
            ax.add_patch(Polygon(upper_tri, facecolor=color_percent, edgecolor='white'))
            
            dy = 0.05
            # Add text
            if not pd.isna(val_count):
                x_l = np.mean([p[0] for p in lower_tri])
                y_l = np.mean([p[1] for p in lower_tri]) - dy
                ax.text(x_l, y_l, f"{int(val_count)}", color=get_text_color(color_count),
                        ha='center', va='center', fontsize=fontsize_tiles)

            if not pd.isna(val_percent):
                x_u = np.mean([p[0] for p in upper_tri])
                y_u = np.mean([p[1] for p in upper_tri]) + dy
                ax.text(x_u, y_u, f"{val_percent:.2f}", color=get_text_color(color_percent),
                        ha='center', va='center', fontsize=fontsize_tiles)

    # Axis ticks and labels
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(regions, fontsize=fontsize_ticks)
    ax.set_yticklabels(regions, fontsize=fontsize_ticks)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='both', length=0)  # remove tick lines
    for spine in ax.spines.values():
        spine.set_visible(False)  # remove axis lines

    # Grid lines as white borders
    ax.set_xticks(np.arange(n+1), minor=True)
    ax.set_yticks(np.arange(n+1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)

    sm1 = plt.cm.ScalarMappable(norm=norm_count, cmap=cmap_count)
    sm1.set_array([])
    sm2 = plt.cm.ScalarMappable(norm=norm_percent, cmap=cmap_percent)
    sm2.set_array([])

    # Get position of main heatmap to place colorbars and barplot
    pos = ax.get_position()
    
    #### Barplots
    # Align read counts with heatmap regions
    read_counts = read_counts_df.reset_index().set_index('group').reindex(regions)['count']

    # Determine dynamic scaling
    max_read = read_counts.max()
    exp = int(np.floor(np.log10(max_read)))
    scale_factor = 10**exp
    read_counts_scaled = read_counts / scale_factor
    
    # Create new axis for bar plot to the **left** of heatmap
    bar_width = 0.15
    bar_pad = 0.05  # increased padding to avoid overlap
    bar_x = pos.x1 + bar_pad # right

    ax_bar = fig.add_axes([bar_x, pos.y0, bar_width, pos.height])

    # Plot horizontal bars (right-aligned, extend left)
    max_read = read_counts_scaled.max()
    y_pos = np.arange(n) + 0.5  # center bars in same Y as heatmap tiles

    ax_bar.barh(y_pos, read_counts_scaled.values, color='darkgray', align='center')

    # left-aligned bars
    ax_bar.set_xlim(0,max_read * 1.05)  # add buffer to avoid overlap with heatmap

    # Align y-ticks with heatmap
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])           # hide redundant region names
    ax_bar.set_ylim(0, n)                # match heatmap vertical scale
    ax_bar.invert_yaxis()                # same order as heatmap

    # Labeling
    ax_bar.set_xlabel(f'# Reads (1e{exp})', labelpad=4, fontsize=fontsize_ticks)
    ax_bar.tick_params(axis='x', labelsize=fontsize_ticks)
    ax_bar.xaxis.set_label_position('bottom')

    # Clean look
    ax_bar.tick_params(axis='y', length=0)
    for spine_name, spine in ax_bar.spines.items():
        if spine_name in ['top', 'right']:
            spine.set_visible(False)

    #### Colorbars heatmap
    # Define width and height of colorbars (shorter height)
    cbar_width = 0.2
    cbar_height = 0.03  # 40% height of the heatmap each
    cbar_x = pos.x1 - 0.3  # just right of heatmap

    # First colorbar (counts) - upper on the right
    cax1 = fig.add_axes([cbar_x, pos.y0 + pos.height - 5*cbar_height, cbar_width, cbar_height])

    # Second colorbar (percent) - below the first
    cax2 = fig.add_axes([cbar_x, pos.y0 + pos.height - 10*cbar_height, cbar_width, cbar_height])

    # Create colorbars (vertical)
    cb1 = fig.colorbar(sm1, cax=cax1, orientation='horizontal')
    cb2 = fig.colorbar(sm2, cax=cax2, orientation='horizontal')
    cb1.set_label('# Sig. Genes', labelpad=-38, loc='center', fontsize=fontsize_tiles)
    cb2.set_label('% Sig. Genes', labelpad=-38, loc='center', fontsize=fontsize_tiles)
    
    cb1.set_ticks([sm1.norm.vmin, (sm1.norm.vmin + sm1.norm.vmax)/2, sm1.norm.vmax])
    cb2.set_ticks([sm2.norm.vmin, (sm2.norm.vmin + sm2.norm.vmax)/2, sm2.norm.vmax])
    cb1.ax.set_xticklabels([int(sm1.norm.vmin), int((sm1.norm.vmin + sm1.norm.vmax)/2), int(sm1.norm.vmax)], fontsize=fontsize_tiles)
    cb2.ax.set_xticklabels([f"{sm2.norm.vmin:.1f}", f"{(sm2.norm.vmin + sm2.norm.vmax)/2:.1f}", f"{sm2.norm.vmax:.1f}"], fontsize=fontsize_tiles)
    
    if fn != None:
        plt.savefig(fn, bbox_inches='tight')
    plt.show()

def barplot_psi(
        x_sparse,
        labels,
        var_info,
        varName,
        celltype="", 
        figsize=(6, 4),
        color="gray"
    ) -> plt.Figure:
    """
    Function to plot a barplot with mean PSI value per brain region.

    Extracts the specified transcript column from the sparse PSI matrix
    and computes mean PSI per region for singlets only and optionally
    a given cell type.

    Parameters
    ----------
    x_sparse : scipy.sparse.csr_matrix
        Sparse PSI matrix (cells × isoforms).

    labels : pandas.DataFrame
        Cell metadata indexed by cell IDs. Must contain at least ['region', 'spot_class', 'first_type'].
    
    var_info : pandas.DataFrame
        Isoform metadata with at least a 'Transcript ID' column that matches x_sparse columns.

    varName : str
        Name of the transcript to plot (must match a value in var_info['Transcript ID']).

    celltype : str, optional
        Filter for a specific cell type (matches labels['first_type']).
        Default is "", meaning no filtering by cell type.

    **kwargs :
        Additional keyword arguments passed to psi_region_barplot,
        e.g. figsize=(6,4), color='gray', etc.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A bar plot showing mean PSI per region for singlets (and optionally per cell type).
    """
    
    # Find transcript index
    idx_var = np.where(var_info["Transcript ID"] == varName)[0]
    if len(idx_var) == 0:
        raise ValueError(f"Transcript '{varName}' not found in var_info['Transcript ID'].")

    x_sparse_transcript = x_sparse[:, idx_var]
    x_coo = x_sparse_transcript.tocoo()
    psi_values = x_coo.data
    idx_notNaN = x_coo.row
    labels_nonzero = labels.iloc[idx_notNaN].copy()
    labels_nonzero["PSI"] = psi_values

    # Apply cell type filter (optional)
    if celltype:
        # Filter singlets
        labels_nonzero = labels_nonzero[labels_nonzero["spot_class"] == "singlet"]
        labels_nonzero = labels_nonzero[labels_nonzero["first_type"] == celltype]
        if labels_nonzero.empty:
            raise ValueError(f"No cells found for cell type '{celltype}'.")
    if "region" not in labels_nonzero.columns:
        raise ValueError("The labels DataFrame must contain a 'region' column.")

    # Compute mean PSI per region and count of cells
    df_summary = (
        labels_nonzero.groupby("region", dropna=False)
        .agg(
            mean_PSI=("PSI", "mean"),
            n_cells=("PSI", "size")
        )
        .reset_index()
    )

     # Keep only regions with >=10 cells
    df_filtered = df_summary[df_summary["n_cells"] >= 10].sort_values("mean_PSI", ascending=False)
    if df_filtered.empty:
        raise ValueError("No regions with ≥10 singlet cells to plot.")
    
    # Draw bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(df_filtered["region"], df_filtered["mean_PSI"], color=color, alpha=0.85)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean PI")
    title_suffix = f" ({celltype})" if celltype else ""
    ax.set_title(f"Mean PI per region — {varName}{title_suffix}", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.close(fig)
    return fig