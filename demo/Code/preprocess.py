from pathlib import Path
import pandas as pd 
import numpy as np
import scanpy as sc
from tqdm.notebook import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix, save_npz, load_npz



def allinfo_addct(fn_allinfo, fn_CIDmap, fn_adata):
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

def create_auxiliary_files(fn_allinfo, output_dir):

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
    
def readAllInfo(fn_allinfo):

    allinfo = pd.read_csv(fn_allinfo, index_col=0, 
                          sep='\t', header=None)

    # Filter out transcripts annotated as 'None'
    allinfo = allinfo[allinfo[11] != 'None']

    return allinfo

def constructSparseMatrix(allinfo, fn_CIDmap, gene_isoform_count):
    
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

def create_isoform_matrix(fn_allinfo, fn_CIDmap, fn_adata, output, mincells=50, mincellspergroup=20):
    
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

    # Read adata and filter for correct cells
    adata = sc.read_h5ad(fn_adata)
    labels = adata.obs
    x_sparse = x_sparse[labels['CellID-original'].values.astype(int)]

    Path(output).mkdir(parents=True, exist_ok=True)
    
    save_npz(f"{output}/X_sparse.npz", x_sparse)
    
    # Save cell/exon labels
    pd.DataFrame(gene_isoform).to_csv(f"{output}/genes_isoforms.csv", index=False, header=False)

    # Save cell labels
    labels.to_csv(f"{output}/labels.csv")
    
    return     

def sparse2df(input_dir):

    # Labels
    labels = pd.read_csv(f'{input_dir}/labels.csv',
                         index_col=0)

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