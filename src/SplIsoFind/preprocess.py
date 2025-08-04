# SPDX-License-Identifier: MIT
"""
Utilities for annotating reads in an allinfo file with cell types and constructing sparse isoform matrices.

This module provides functions to:

- Label reads with celltype-region labels from segmentation data
- Generate auxiliary CSV outputs for scisorseqr (isoform IDs, counts per cluster)
- Build sparse cell-by-isoform matrices for downstream analysis
- Combine two-slide experiments
- Load sparse matrices into DataFrames
"""
from pathlib import Path
import pandas as pd 
import numpy as np
import scanpy as sc
from tqdm.notebook import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack

def allinfo_addct(
    fn_allinfo: str,
    fn_CIDmap: str,
    fn_adata: str
) -> None:
    """
    Annotate raw read allinfo file with cell-type-region labels and save filtered file.

    Parameters
    ----------
    fn_allinfo : str
        Path to raw AllInfo TSV file (index_col=0, no header).
    fn_CIDmap : str
        Path to TSV file with mapping of barcodes to original CellIDs.
    fn_adata : str
        Path to AnnData .h5ad file with obs containing 'first_type' (cell type), 
        'spot_class' (whether a cell is a singlet), and 'subregion'.

    Returns
    -------
    None
        Writes `<fn_allinfo>.filtered.labeled.gz` with same format.
    """

    x = pd.read_csv(fn_allinfo, sep='\t', header=None, index_col=0)
    print('Number of reads:')
    print(len(x))

    # Remove genes with 0 exons in intron chain
    x = x[x[10]>0]
    print('Number of reads with at least 1 exon in intron chain:')
    print(len(x))

    # Filter for reads overlapping segmented cells
    CIDmap = pd.read_csv(fn_CIDmap, sep='\t')
    CIDmap['CellID-original'] = CIDmap['CellID-original'].astype(int)
    bc_set = set(CIDmap['barcode'])

    x_filt = x[[xx in bc_set for xx in tqdm(x[3])]].copy()
    print('Number of reads overlapping segmented cells:')
    print(len(x_filt))

    # Read labels file
    adata = sc.read_h5ad(fn_adata)
    labels = adata.obs
    # labels['CellID-original'] = range(len(labels))
    labels['celltype'] = labels['first_type']
    labels['celltype'] = labels['celltype'].cat.add_categories(['other'])
    labels['celltype'][labels['spot_class'] != 'singlet'] = 'other'
    labels['celltype'][np.isin(labels['celltype'], ['ExciteNeuron', 'InhibNeuron', 'Astro', 'Oligo']) == False] = 'other'
    labels['ct_reg'] = labels['celltype'].astype(str) + '_' + labels['subregion'].astype(str)
    
    # Filter otherhemisphere
    labels = labels[labels['subregion'] != 'OtherHemisphere']

    # Add ct label to CIDmap
    CIDmap = CIDmap.merge(labels[['ct_reg', 'CellID-original']], 
                          on=['CellID-original'], 
                          how='left')  
    
    # Add ct label to allinfo
    x_filt.loc[:,2] = x_filt.merge(CIDmap[['barcode','ct_reg']], left_on=3, 
                             right_on='barcode', how='left')['ct_reg'].values

    # Remove NaN values
    x_filt = x_filt[x_filt[2].notna()]
    print('Number of reads with label:')
    print(len(x_filt))
    
    x_filt.to_csv(fn_allinfo[:-3]+'.filtered.labeled.gz', header=False, 
                  sep='\t', index=True, compression='gzip')

    return 

def create_auxiliary_files(
    fn_allinfo: str,
    output_dir: str
) -> None:
    """
    Generate isoform ID mapping and counts-per-cluster files as input for scisorseqr.

    Parameters
    ----------
    fn_allinfo : str
        Path to labeled AllInfo TSV file (index_col=0, no header).
    output_dir : str
        Directory to write 'Iso-IsoID.csv' and 'NumIsoPerCluster' files.

    Returns
    -------
    None
    """

    ## Create output dir if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    allinfo = pd.read_csv(fn_allinfo,
                          sep='\t', header=None, index_col=0)
    print('Number of reads:')
    print(len(allinfo))
    
    # Remove reads without isoform assigned
    allinfo = allinfo[allinfo[11] != 'None']
    print('Number of reads with isoform assigned:')
    print(len(allinfo))
    
    # Filter out genes with 1 transcript
    gene_count = allinfo.groupby([1, 11]).size().groupby(1).size()
    genes_tokeep = gene_count[gene_count > 1].index

    # Filter out genes with <10 counts
    gene_count2 = allinfo.groupby(1).size()
    genes_tokeep2 = gene_count2[gene_count2 >= 10].index
    
    genes_tokeep3 = np.intersect1d(genes_tokeep, genes_tokeep2)

    # Create Iso-IsoID file
    gene_tr = allinfo.groupby([1, 11]).size().reset_index()
    gene_tr = gene_tr[np.isin(gene_tr[1], genes_tokeep3)]
    gene_tr = gene_tr.sort_values([1,0], ascending=[True, False])
    gene_tr['isoID'] = gene_tr.groupby(1).cumcount()
    gene_tr.columns = ['Gene', 'Isoform', 'n', 'iso.id']
    gene_tr.to_csv(f'{output_dir}/Iso-IsoID.csv', sep='\t',
                    index=None)

    # Create counts per celltype
    gene_reg_tr_count = allinfo.groupby([1,2,11]).size().reset_index()
    gene_reg_tr_count = gene_reg_tr_count[np.isin(gene_reg_tr_count[1], genes_tokeep3)]
    gene_reg_tr_count.columns = ['Gene', 'Celltype', 'Isoform', 'n']
    gene_reg_tr_count = gene_reg_tr_count.merge(gene_tr[['Gene','Isoform','iso.id']], 
                                                on=['Gene','Isoform'], how='left')
    gene_reg_tr_count['combi'] = gene_reg_tr_count['Gene'] + "::" + gene_reg_tr_count['iso.id'].astype('str')
    gene_reg_tr_count.iloc[:,[5,1,3]].to_csv(f'{output_dir}/NumIsoPerCluster',
                                             index=None, header=None, sep='\t')
    
def readAllInfo(
    fn_allinfo: str
) -> pd.DataFrame:
    """
    Load and filter AllInfo table (remove 'None' isoforms).

    Parameters
    ----------
    fn_allinfo : str
        Path to AllInfo TSV (index_col=0, no header).

    Returns
    -------
    pandas.DataFrame
    """

    allinfo = pd.read_csv(fn_allinfo, index_col=0, 
                          sep='\t', header=None)

    # Filter out transcripts annotated as 'None'
    allinfo = allinfo[allinfo[11] != 'None']

    return allinfo

def constructSparseMatrix(
    allinfo: pd.DataFrame,
    fn_CIDmap: str,
    gene_isoform_count: pd.DataFrame
) -> tuple[csr_matrix, list]:
    """
    Build a cell-by-isoform sparse matrix of relative expression values.

    Parameters
    ----------
    allinfo : pandas.DataFrame
        Filtered AllInfo with reads.
    fn_CIDmap : str
        Path to barcode-to-CellID mapping TSV.
    gene_isoform_count : pandas.DataFrame
        DataFrame with columns [Gene, Isoform, count].

    Returns
    -------
    x_sparse : csr_matrix
        Shape (n_cells, n_transcripts), values = PSI per cell.
    gene_isoform_list : list of tuple
        List of (Gene, Isoform) pairs matching matrix columns.
    """
    
    CIDmap = pd.read_csv(fn_CIDmap, sep='\t').set_index('barcode')
    max_CellID = int(CIDmap['CellID-original'].max())

    # Filter CIDmap to only barcodes in allinfo
    CIDmap = CIDmap[CIDmap.index.isin(allinfo[3])]

    # Map barcodes to Cell IDs
    barcode_to_cid = CIDmap['CellID-original'].to_dict()

    # Create isoform mapping
    gene_isoform_list = list(gene_isoform_count.set_index([1, 11]).index)
    gene_isoform_to_idx = {pair: j for j, pair in enumerate(gene_isoform_list)}
    
    n_cells = max_CellID + 1
    n_transcripts = len(gene_isoform_list)

    # Group by gene, isoform, barcode
    grouped = allinfo.groupby([1, 11, 3]).size().reset_index(name='count')

    # Build inc/exc dictionaries
    inc_dict = defaultdict(dict)
    exc_dict = defaultdict(dict)
    
    for g, t, bc, count in grouped.itertuples(index=False):
        if bc in barcode_to_cid:
            inc_dict[(g, t)][bc] = count
    
    for (g, t), bc_counts in inc_dict.items():
        other_ts = [tt for (gg, tt) in inc_dict if gg == g and tt != t]
        exc_counts = defaultdict(int)
        for tt in other_ts:
            for bc, c in inc_dict[(g, tt)].items():
                exc_counts[bc] += c
        exc_dict[(g, t)] = exc_counts

    # Build sparse matrix data
    x_data, x_row, x_col = [], [], []
    
    for (g, t), inc_bc_dict in tqdm(inc_dict.items(), total=len(inc_dict)):
        if (g, t) in gene_isoform_to_idx:
            j = gene_isoform_to_idx[(g, t)]
            exc_bc_dict = exc_dict.get((g, t), {})
        
            # Map barcode counts → CID counts
            inc_cid_counts = defaultdict(int)
            exc_cid_counts = defaultdict(int)
        
            for bc, count in inc_bc_dict.items():
                if bc in barcode_to_cid:
                    cid = barcode_to_cid[bc]
                    inc_cid_counts[cid] += count
        
            for bc, count in exc_bc_dict.items():
                if bc in barcode_to_cid:
                    cid = barcode_to_cid[bc]
                    exc_cid_counts[cid] += count
        
            all_cids = set(inc_cid_counts) | set(exc_cid_counts)
        
            for cid in all_cids:
                inc = inc_cid_counts.get(cid, 0)
                exc = exc_cid_counts.get(cid, 0)
                total = inc + exc
                if total > 0:
                    x_data.append(inc / total)
                    x_row.append(cid)
                    x_col.append(j)
            
    # Build sparse matrices
    x_sparse = csr_matrix((x_data, (x_row, x_col)), shape=(n_cells, n_transcripts))

    return x_sparse, gene_isoform_list

def create_isoform_matrix(
    fn_allinfo: str,
    fn_CIDmap: str,
    fn_adata: str,
    output: str,
    mincells: int = 50,
    mincellspergroup: int = 20
) -> None:
    """
    Build and save a filtered cell by isoform sparse matrix for a single slide.

    This function:
    1. Loads read-level data via `readAllInfo`.
    2. Filters genes with at least `mincells` total reads.
    3. Filters isoforms with at least `mincellspergroup` reads.
    4. Retains genes with ≥2 isoforms after filtering.
    5. Constructs a cell-by-isoform sparse PSI matrix.
    6. Filters matrix columns by the same cell count thresholds.
    7. Saves:
       - `X_sparse.npz`: CSR matrix of PSI values.
       - `genes_isoforms.csv`: List of (Gene, Isoform) pairs.
       - `labels.csv`: Cell metadata from AnnData.

    Parameters
    ----------
    fn_allinfo : str
        Path to labeled AllInfo TSV (filtered by `allinfo_addct`), index_col=0.
    fn_CIDmap : str
        Path to TSV mapping barcodes to original CellIDs.
    fn_adata : str
        Path to AnnData .h5ad file for cell ordering and labels.
    output : str
        Directory to write output files (`X_sparse.npz`, etc.).
    mincells : int, default=50
        Minimum total reads per isoform to keep a gene.
    mincellspergroup : int, default=20
        Minimum reads in high/low group per isoform to include.

    Returns
    -------
    None
    """
    allinfo = readAllInfo(fn_allinfo)

    # Filter for genes with enough counts
    gene_counts = allinfo.groupby([1]).size()
    genes_tokeep = gene_counts[gene_counts >= mincells].index
    allinfo = allinfo[allinfo[1].isin(genes_tokeep)]
    
    # Filter for potentially interesting isoforms
    gene_isoform_count = allinfo.groupby([1, 11]).size().reset_index(name='count')
    gene_isoform_count = gene_isoform_count[gene_isoform_count['count'] >= mincellspergroup]
    
    # Keep only genes with ≥2 isoforms left
    multi_iso_genes = gene_isoform_count.groupby(1).filter(lambda x: len(x) >= 2)[1].unique()
    gene_isoform_count = gene_isoform_count[gene_isoform_count[1].isin(multi_iso_genes)]
    allinfo = allinfo[allinfo[1].isin(multi_iso_genes)]
    
    print(f'Potentially interesting isoforms (total): {len(gene_isoform_count)}')
    num_novel_isoforms = gene_isoform_count[11].str.split('.', expand=True)[2].notna().sum()
    print(f'Potentially interesting isoforms (novel): {num_novel_isoforms}')

    x_sparse, gene_isoform_list = constructSparseMatrix(allinfo, fn_CIDmap, gene_isoform_count)

    # Read adata to get the labels
    adata = sc.read_h5ad(fn_adata)
    labels = adata.obs
    x_sparse = x_sparse[labels['CellID-original'].values.astype(int)]

    # Filter for mincells and mincellspergroup
    data = x_sparse.data
    high_mask = csr_matrix((data >= 0.5, x_sparse.indices, x_sparse.indptr), shape=x_sparse.shape)
    low_mask = csr_matrix((data < 0.5, x_sparse.indices, x_sparse.indptr), shape=x_sparse.shape)
    
    high_counts = np.array(high_mask.sum(axis=0)).ravel()
    low_counts = np.array(low_mask.sum(axis=0)).ravel()
    total_counts = np.array(x_sparse.getnnz(axis=0)).ravel()
    
    keep = (high_counts >= mincellspergroup) & (low_counts >= mincellspergroup) & (total_counts >= mincells)

    gene_isoform = np.array(gene_isoform_list)[keep]
    x_sparse = x_sparse[:,keep]

    Path(output).mkdir(parents=True, exist_ok=True)
    
    save_npz(f"{output}/X_sparse.npz", x_sparse)
    
    # Save cell/exon labels
    pd.DataFrame(gene_isoform).to_csv(f"{output}/genes_isoforms.csv", index=False, header=False)

    # Save cell labels
    labels.to_csv(f"{output}/labels.csv")
    
    return     

def create_isoform_matrix_twoslides(
    fn_allinfo_S1: str,
    fn_allinfo_S2: str,
    fn_CIDmap_S1: str,
    fn_CIDmap_S2: str,
    fn_adata_S1: str,
    fn_adata_S2: str,
    output: str,
    mincells: int = 50,
    mincellspergroup: int = 20
) -> None:
    """
    Build and save a filtered cell by isoform sparse matrix combining two slides.

    This function performs the same filtering and matrix construction as
    `create_isoform_matrix`, but for two separate slides whose results
    are then vertically concatenated:
    1. Load and concatenate `readAllInfo` outputs for slide1 and slide2.
    2. Apply gene and isoform read count filters.
    3. Construct separate sparse matrices via `constructSparseMatrix`.
    4. Subset each matrix by cell order from respective AnnData.
    5. Vertically stack matrices and concatenate labels.
    6. Apply cell-count filters on the combined matrix.
    7. Save outputs similarly to single-slide version.

    Parameters
    ----------
    fn_allinfo_S1, fn_allinfo_S2 : str
        Paths to filtered AllInfo TSVs for slide1 and slide2.
    fn_CIDmap_S1, fn_CIDmap_S2 : str
        Paths to barcode-to-CellID TSVs for each slide.
    fn_adata_S1, fn_adata_S2 : str
        Paths to AnnData .h5ad files for each slide.
    output : str
        Directory to write output files (`X_sparse.npz`, etc.).
    mincells : int, default=50
        Minimum total reads per isoform to keep a gene.
    mincellspergroup : int, default=20
        Minimum reads in high/low group per isoform to include.

    Returns
    -------
    None
    """

    allinfo_S1 = readAllInfo(fn_allinfo_S1)
    allinfo_S2 = readAllInfo(fn_allinfo_S2)
    allinfo = pd.concat((allinfo_S1, allinfo_S2), axis=0)

    # Filter for genes with enough counts
    gene_counts = allinfo.groupby([1]).size()
    genes_tokeep = gene_counts[gene_counts >= mincells].index
    allinfo = allinfo[allinfo[1].isin(genes_tokeep)]
    
    # Filter for potentially interesting isoforms
    gene_isoform_count = allinfo.groupby([1, 11]).size().reset_index(name='count')
    gene_isoform_count = gene_isoform_count[gene_isoform_count['count'] >= mincellspergroup]
    
    # Keep only genes with ≥2 isoforms left
    multi_iso_genes = gene_isoform_count.groupby(1).filter(lambda x: len(x) >= 2)[1].unique()
    gene_isoform_count = gene_isoform_count[gene_isoform_count[1].isin(multi_iso_genes)]
    allinfo = allinfo[allinfo[1].isin(multi_iso_genes)]
    
    print(f'Potentially interesting isoforms (total): {len(gene_isoform_count)}')
    num_novel_isoforms = gene_isoform_count[11].str.split('.', expand=True)[2].notna().sum()
    print(f'Potentially interesting isoforms (novel): {num_novel_isoforms}')

    x_sparse_S1, gene_isoform_list = constructSparseMatrix(allinfo_S1, fn_CIDmap_S1, gene_isoform_count)
    x_sparse_S2, gene_isoform_list = constructSparseMatrix(allinfo_S2, fn_CIDmap_S2, gene_isoform_count)

    # Read adata to get the labels
    adata_S1 = sc.read_h5ad(fn_adata_S1)
    labels_S1 = adata_S1.obs
    x_sparse_S1 = x_sparse_S1[labels_S1['CellID-original'].values.astype(int)]

    adata_S2 = sc.read_h5ad(fn_adata_S2)
    labels_S2 = adata_S2.obs
    x_sparse_S2 = x_sparse_S2[labels_S2['CellID-original'].values.astype(int)]
    
    x_sparse = vstack([x_sparse_S1, x_sparse_S2])
    labels = pd.concat((labels_S1, labels_S2))

    # Filter for mincells and mincellspergroup
    data = x_sparse.data
    high_mask = csr_matrix((data >= 0.5, x_sparse.indices, x_sparse.indptr), shape=x_sparse.shape)
    low_mask = csr_matrix((data < 0.5, x_sparse.indices, x_sparse.indptr), shape=x_sparse.shape)
    
    high_counts = np.array(high_mask.sum(axis=0)).ravel()
    low_counts = np.array(low_mask.sum(axis=0)).ravel()
    total_counts = np.array(x_sparse.getnnz(axis=0)).ravel()
    
    keep = (high_counts >= mincellspergroup) & (low_counts >= mincellspergroup) & (total_counts >= mincells)

    gene_isoform = np.array(gene_isoform_list)[keep]
    x_sparse = x_sparse[:,keep]

    Path(output).mkdir(parents=True, exist_ok=True)
    
    save_npz(f"{output}/X_sparse.npz", x_sparse)
    
    # Save cell/exon labels
    pd.DataFrame(gene_isoform).to_csv(f"{output}/genes_isoforms.csv", index=False, header=False)

    # Save cell labels
    labels.to_csv(f"{output}/labels.csv")
    
    return     


def sparse2df(
    input_dir: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a saved sparse isoform matrix into dense DataFrame form.

    This function:
    1. Reads `labels.csv` and `genes_isoforms.csv`.
    2. Loads `X_sparse.npz` as a CSR matrix.
    3. Converts zeros to NaN via a mask.
    4. Returns:
       - `x`: DataFrame of PSI values (cells × isoforms).
       - `labels`: DataFrame of cell metadata.

    Parameters
    ----------
    input_dir : str
        Directory containing output files from `create_isoform_matrix*`.

    Returns
    -------
    x : pandas.DataFrame
        Cell-by-isoform relative expression matrix (NaN where zero counts).
    labels : pandas.DataFrame
        Cell metadata indexed by cell identifier.
    """

    # Labels
    labels = pd.read_csv(f'{input_dir}/labels.csv',
                         index_col=0)
    labels.index = labels.index.astype('str') + '_' + labels['sample']

    # Isoforms 
    isoforms=pd.read_csv(f'{input_dir}/genes_isoforms.csv',header=None).values.squeeze()
    
    # Data matrix
    x_sparse = load_npz(f'{input_dir}/X_sparse.npz')
    mask = csr_matrix(([1]*len(x_sparse.data), x_sparse.indices, x_sparse.indptr), shape=np.shape(x_sparse)).toarray().astype(float)
    mask[mask == 0] = np.nan

    # Convert to dataframe
    x = pd.DataFrame(x_sparse.toarray()*mask, 
                     index=labels.index,
                     columns=isoforms[:,1])
        
    return x, labels