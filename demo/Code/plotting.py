import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.colors import Normalize
from matplotlib import cm

import seaborn as sns

def spatial_hexplot(x, labels, imarray, varName, celltype, 
                    hexsize=120, fig_size=(5,5), ax=None, plot_lim=None,
                    alpha=1, cmap='viridis', show_colorbar=True,
                    staining_max='grey', staining_min='white',
                    linewidths=0.1):

    # Subset data
    idx_var = np.where(x.columns == varName)[0][0]
    if celltype != '':
        idx_ct = np.where((labels['first_type'] == celltype) &
                          (labels['spot_class'] == 'singlet'))[0]
        x = x.iloc[idx_ct, idx_var]
        labels = labels.iloc[idx_ct]
    else:
        x = x.iloc[:, idx_var]

    if len(x.dropna()) == 0:
        print('No data')
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
            ax.imshow(imarray, alpha=alpha, cmap=staining_cmap)
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

def read_results(input_dir, dataset, region, celltype):

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

def get_countmatrix(res, region_map=None):
    
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

    # Masking
    mask_upper = np.triu(np.ones_like(count_matrix, dtype=bool), k=1)
    mask_lower = np.tril(np.ones_like(count_matrix, dtype=bool), k=-1)

    if region_map != None:
        # Rename both axes of your matrices
        count_matrix.rename(index=region_map, columns=region_map, inplace=True)
        percent_matrix.rename(index=region_map, columns=region_map, inplace=True)

    return count_matrix, percent_matrix

def get_barplot_counts(allinfo, ct_comp_file):
    
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
def get_text_color(rgba):
    r, g, b, _ = rgba
    brightness = 0.299*r + 0.587*g + 0.114*b
    return 'white' if brightness < 0.5 else 'black'

def plot_heatmap(input_dir, dataset, region, celltype, allinfo,
                 region_map=None, region_map2=None,
                 figsize=(4,4), cmap_count=cm.Oranges,
                 cmap_percent = cm.Blues, fontsize_tiles=11, 
                 fontsize_ticks=12,
                 fn=None, vmax_count=None,
                 vmax_perc=None):

    res = read_results(input_dir=input_dir, 
                       dataset=dataset, 
                       region=region, 
                       celltype=celltype)
    count_matrix, percent_matrix = get_countmatrix(res, region_map)
    read_counts_df = get_barplot_counts(allinfo=allinfo, 
                                        ct_comp_file=f'{input_dir}/ct_files/CellTypes_{celltype}_{region}')
    if region_map != None:
        read_counts_df = read_counts_df.rename(index=region_map)

    
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
