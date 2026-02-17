# SPDX-License-Identifier: MIT
"""
Spatial autocorrelation functions using Moran's I.

This module provides functions to compute Moran's I for long-read spatial transcriptomics data,
including permutation-based significance testing and FDR correction.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib

from statsmodels.stats.multitest import fdrcorrection

from scipy.sparse import csr_matrix

from sklearn.neighbors import NearestNeighbors
from libpysal.weights import W
import esda

def moransI(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    nperm: int = 100000,
    k: int = 10,
    mincells: int = 50,
    imb: float = 0.05,
    mincellspergroup: int = 20,
    celltypes: list = ['All','ExciteNeuron','InhibNeuron','Astro','Oligo'],
    x_coord: str = 'x',
    y_coord: str = 'y',
    output_dir: str = '',
    n_jobs: int = 1,
    seed: int = 0,
    test: str = 'moransI'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Moran's I scores, p-values, and q-values (Benjaminini-Yekutieli FDR-corrected) for each isoform and cell type.

    Parameters
    ----------
    x : pandas.DataFrame
        Feature matrix of shape (n_cells, n_isoforms), with relative expression values between 0 and 1.
    labels : pandas.DataFrame
        Cell metadata, must contain columns for `x_coord`, `y_coord`,
        `spot_class`, and `first_type`.`spot_class` and `first_type` are used to filter cells per cell type.
    nperm : int, default=100000
        Number of permutations for significance testing.
    k : int, default=10
        Number of nearest neighbors for spatial weight matrix.
    mincells : int, default=50
        Minimum number of cells required to have relative expression values for an isoform.
    imb : float, default=0.05
        Minimum ratio of minority group to total.
    mincellspergroup : int, default=20
        Minimum number of cells per binary group (x > 0.5 or ≤ 0.5).
    celltypes : list of str
        List of cell type labels to test; 'All' computes on all cells.
    x_coord, y_coord : str
        Column names in `labels` for spatial coordinates.
    output_dir : str
        Directory path to save Moran's I scores, p-values, and q-values as CSV files.
    n_jobs : int
        Number of parallel workers.
    seed : int
        Base seed used during permutation testing.
        
    Returns
    -------
    mI : pandas.DataFrame
        Moran's I scores, index = isoforms, columns = celltypes.
    pval : pandas.DataFrame
        Permutation p-values, index = isoforms, columns = celltypes.
    qval : pandas.DataFrame
        FDR-corrected q-values, index = isoforms, columns = celltypes.
    """
    
    mI = pd.DataFrame(np.nan, index=x.columns, columns=celltypes)
    pval = pd.DataFrame(np.nan, index=x.columns, columns=celltypes)
    qval = pd.DataFrame(np.nan, index=x.columns, columns=celltypes)

    # Iterate over isoforms
    if n_jobs == 1:
        results = [
            _compute_single_isoform(
                    x.columns[i], x.iloc[:, i],
                    labels, celltypes,
                    nperm, k, mincells, mincellspergroup, imb,
                    x_coord, y_coord, seed+i, test
            )
            for i in tqdm(range(x.shape[1]), desc="Computing Moran's I")
        ]
    else:
        with tqdm_joblib(desc="Computing Moran's I", total=x.shape[1]):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_compute_single_isoform)(
                    x.columns[i], x.iloc[:, i],
                    labels, celltypes,
                    nperm, k, mincells, mincellspergroup, imb,
                    x_coord, y_coord, seed+i, test
                )
                for i in range(x.shape[1])
            )

    # Collect results
    for isoform, iso_dict in results:
        for ct, (I_val, p_val) in iso_dict.items():
            mI.loc[isoform, ct] = I_val
            pval.loc[isoform, ct] = p_val    
    
    # BY FDR correction
    for ct in celltypes:
        tocorrect = pval[ct][pval[ct].notna()]
        _, q = fdrcorrection(tocorrect, method='n')
        qval[ct][pval[ct].notna()] = q

    # Save files
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_mI = f"{output_dir}/MoransI_scores_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
        output_pval = f"{output_dir}/MoransI_pval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
        output_qval = f"{output_dir}/MoransI_qval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
    
        mI.to_csv(output_mI)    
        pval.to_csv(output_pval)
        qval.to_csv(output_qval)
            
    return mI, pval, qval

def moransI_sparse(
    x_sparse: csr_matrix,
    labels: pd.DataFrame,
    isoform_ids: pd.DataFrame,
    nperm: int = 100000,
    k: int = 10,
    mincells: int = 50,
    imb: float = 0.05,
    mincellspergroup: int = 20,
    celltypes: list = ['All','ExciteNeuron','InhibNeuron','Astro','Oligo'],
    x_coord: str = 'x',
    y_coord: str = 'y',
    output_dir: str = '',
    n_jobs: int = 1,
    seed: int = 0,
    test: str = 'moransI'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute Moran's I spatial autocorrelation scores for each isoform using a
    **sparse matrix** (csr_matrix) as input. This function mirrors
    `moransI_parallel` but avoids densifying the full matrix.

    Each isoform is extracted as a 1D sparse vector, converted to dense (only
    that column), filtered, and analysed using `_compute_single_isoform`.

    Parameters
    ----------
    x_sparse : csr_matrix
        Sparse matrix of shape (n_cells, n_isoforms). Expected to contain PSI or
        relative expression values with explicit zeros; missing values should be NaN.
    labels : pandas.DataFrame
        Cell metadata indexed identically to the rows of `x_sparse`.
        Must contain `x_coord`, `y_coord`, `spot_class`, and `first_type`.
    isoform_ids : pandas.DataFrame
        Isoform names corresponding to the columns of `x_sparse`. 
        Must contain one column called 'Transcript ID'
    nperm : int
        Number of permutations for significance testing.
    k : int
        Number of nearest neighbors in spatial weight matrix.
    mincells : int
        Minimum number of cells expressing an isoform.
    imb : float
        Minimum fraction for minority group.
    mincellspergroup : int
        Minimum number of cells per binary group (x > 0.5).
    celltypes : list
        Cell types to analyze. 'All' analyzes all cells.
    x_coord, y_coord : str
        Spatial coordinate column names.
    output_dir : str
        Directory to write CSV output files.
    n_jobs : int
        Number of parallel workers.
    seed : int
        Base seed used during permutation testing.

    Returns
    -------
    mI : pandas.DataFrame
        Moran's I values.
    pval : pandas.DataFrame
        Permutation p-values.
    qval : pandas.DataFrame
        BY-FDR corrected q-values.
    """

    mI = pd.DataFrame(np.nan, index=isoform_ids['Transcript ID'], columns=celltypes)
    pval = pd.DataFrame(np.nan, index=isoform_ids['Transcript ID'], columns=celltypes)
    qval = pd.DataFrame(np.nan, index=isoform_ids['Transcript ID'], columns=celltypes)

    # Iterate over isoforms
    def _prepare_inputs_for_isoform(idx_iso):
        """
        Internal helper to extract sparse column, convert to temporary dense vector,
        filter NaNs, and run `_compute_single_isoform`.
        """
        seed_iso = int(seed + idx_iso)
        
        col = x_sparse.getcol(idx_iso)  # returns a sparse column vector
        nonzero_idx = col.tocoo().row          # FAST: row indices directly
        values = col.data                  # FAST: corresponding PSI values
        labels_sub = labels.iloc[nonzero_idx]
        x_sub = pd.DataFrame(
            values.reshape(-1,1),
            index=labels_sub.index,
            columns=[isoform_ids['Transcript ID'].values[idx_iso]]
        )

        # Call same helper as dense pipeline
        return _compute_single_isoform(
            isoform_ids['Transcript ID'].values[idx_iso],
            x_sub.iloc[:, 0], labels_sub,
            celltypes, nperm,
            k, mincells, mincellspergroup,
            imb, x_coord, y_coord,
            seed_iso, test
        )

    if n_jobs == 1:
        results = [
            _prepare_inputs_for_isoform(i)
            for i in tqdm(range(x_sparse.shape[1]), desc="Computing Moran's I (serial)")
        ]
    else:
        with tqdm_joblib(desc="Computing Moran's I", total=x_sparse.shape[1]):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_prepare_inputs_for_isoform)(i)
                for i in range(x_sparse.shape[1])
            )

    # Collect results
    for isoform, iso_dict in results:
        for ct, (I_val, p_val) in iso_dict.items():
            mI.loc[isoform, ct] = I_val
            pval.loc[isoform, ct] = p_val
    
    # BY FDR correction
    for ct in celltypes:
        tocorrect = pval[ct][pval[ct].notna()]
        _, q = fdrcorrection(tocorrect, method='n')
        qval[ct][pval[ct].notna()] = q

    # Save files
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_mI = f"{output_dir}/MoransI_scores_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
        output_pval = f"{output_dir}/MoransI_pval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
        output_qval = f"{output_dir}/MoransI_qval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}_seed{seed}.csv"
    
        mI.to_csv(output_mI)    
        pval.to_csv(output_pval)
        qval.to_csv(output_qval)
            
    return mI, pval, qval


def _compute_single_isoform(
    isoform_name, x_i, labels, celltypes,
    nperm, k, mincells, mincellspergroup, imb,
    x_coord, y_coord, seed, test
):
    """
    Compute Moran's I and permutation p-values for a single isoform 
    across all requested cell types.
    Returns (isoform_name, results_dict) where:
        results_dict[celltype] = (I, pval)
    """
    results = {}
    x_i = x_i.dropna()

    # Global filtering 
    if len(x_i) < mincells or x_i.var() == 0:
        return isoform_name, results

    # Loop over cell types 
    for ct_idx, ct in enumerate(celltypes):

        seed_ct = seed + 10000*ct_idx

        # ---------------------- ALL CELLS --------------------------------
        if ct == "All":

            labels_i = labels.loc[x_i.index]
            _, counts = np.unique(x_i > 0.5, return_counts=True)
        
            # Check for enough cells per group 
            if np.any(counts < mincellspergroup) or ((np.min(counts)/len(x_i)) < imb):
                continue

            coords = labels_i[[x_coord, y_coord]].to_numpy()
            w = _calculate_weight_matrix_sklearn(coords, k) 

            np.random.seed(seed_ct)
            if test == 'moransI':
                res = esda.Moran(x_i, w, permutations=nperm, transformation="b") 
                pval = res.p_sim if res.I > 0 else 1  
                results[ct] = (res.I, pval)
                
            elif test == 'gearyC':
                res = esda.Geary(x_i, w, permutations=nperm, transformation="b") 
                pval = res.p_sim if res.C < 1 else 1  
                results[ct] = (res.C, pval)
                
            continue

        # ---------------------- SPECIFIC CELL TYPES -----------------------
        tokeep = (labels_i["spot_class"] == "singlet") & (labels_i["first_type"] == ct)
        x_ct = x_i[tokeep]

        if len(x_ct) < mincells or x_ct.var() == 0:
            continue

        _, counts_ct = np.unique(x_ct > 0.5, return_counts=True)
        if np.any(counts_ct < mincellspergroup) or (np.min(counts_ct) / len(x_ct) < imb):
            continue

        coords_ct = labels_i.loc[tokeep, [x_coord, y_coord]].to_numpy()
        w = _calculate_weight_matrix_sklearn(coords_ct, k) 
        
        np.random.seed(seed_ct)
        if test == 'moransI':
            res = esda.Moran(x_ct, w, permutations=nperm, transformation="b") 
            pval = res.p_sim if res.I > 0 else 1  
            results[ct] = (res.I, pval)
            
        elif test == 'gearyC':
            res = esda.Geary(x_ct, w, permutations=nperm, transformation="b") 
            pval = res.p_sim if res.C < 1 else 1  
            results[ct] = (res.C, pval)
        
    return isoform_name, results





def _calculate_weight_matrix_sklearn(
    locations: pd.DataFrame, 
    k: int
) -> W:
    """
    Build a libpysal spatial weights matrix via k-nearest neighbors.

    Parameters
    ----------
    locations : pandas.DataFrame
        Coordinates of cells, shape (n_cells, 2), columns = [x, y].
    k : int
        Number of neighbors to include (self excluded).

    Returns
    -------
    W
        libpysal.weights.W object with equal weights to k neighbors.
    """

    # Fit nearest neighbors model
    # k+1 cause we will remove the diagonal afterwards
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(locations)
    
    # Find k-nearest neighbors (distances and indices)
    _, indices = nn.kneighbors(locations)
    
    # Convert to libpysal object
    neighbors = {i: list(indices[i, 1:]) for i in range(indices.shape[0])}  # Remove self-reference
    weights = {i: [1] * len(neighbors[i]) for i in neighbors}  # Assign equal weights

    return W(neighbors, weights)


def moransI_ctperm(
    x: pd.DataFrame,
    labels: pd.DataFrame,
    var_totest: list,
    nperm: int = 10000,
    k: int = 10,
    x_coord: str = 'x',
    y_coord: str = 'y',
    output_dir: str = '',
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Compute Moran's I with cell-type constrained permutation.

    Performs a two-phase test: original Moran's I permutation test (shuffling all cells) and 
    a permutation test while shuffling cells only within the same cell type.

    Parameters
    ----------
    x : pandas.DataFrame
        Feature matrix of shape (n_cells, n_isoforms), with relative expression values between 0 and 1.
    labels : pandas.DataFrame
        Cell metadata, must contain columns for `x_coord`, `y_coord`,
        'spot_class','first_type','first_type_weight','second_type'.
    var_totest : list of str
        Subset of columns in `x` to test.
    nperm : int, default=10000
        Number of random permutations for cell-type assignment.
    k : int, default=10
        Number of neighbors for spatial weight matrix.
    x_coord, y_coord : str
        Column names in `labels` for spatial coordinates.
    output_dir : str
        Directory path to save result as CSV files.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    res : pandas.DataFrame
        Index = tested isoforms, columns = ['morans I','p-value (original)',
        'p-value (new)','Num cells','Imbalance'].
    """

    var_order = x[var_totest].notna().sum().sort_values(ascending=False).index

    # Iterate over isoforms
    if n_jobs == 1:
        results = [
            _compute_ctperm_for_variable(
                    i, x[i],
                    labels, nperm, 
                    k, x_coord, y_coord
            )
            for i in tqdm(var_order, desc="Moran's I with ct-constrained perm.")
        ]
    else:
        with tqdm_joblib(desc="Moran's I with ct-constrained perm.", total=len(variables)):
            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_compute_ctperm_for_variable)(
                    i, x[i],
                    labels, nperm, 
                    k, x_coord, y_coord
                )
                for i in var_order
            )


    # Build DataFrame
    res = pd.DataFrame(
        results,
        columns=['variable', 'morans I', 'p-value (original)', 'p-value (new)', 'Num cells', 'Imbalance']
    ).set_index('variable')
    
    # Save results
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_res = f"{output_dir}/MoransI_pval_nperm{nperm}_k{k}_ctconstrained.csv"
        res.to_csv(output_res) 

    return res


def moransI_ctperm_sparse(
    x_sparse: csr_matrix,
    labels: pd.DataFrame,
    isoform_ids: pd.DataFrame,
    var_totest: list,
    nperm: int = 10000,
    k: int = 10,
    x_coord: str = 'x',
    y_coord: str = 'y',
    output_dir: str = '',
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Compute Moran's I with cell-type constrained permutation using sparse input matrix.

    Performs a two-phase test: original Moran's I permutation test (shuffling all cells) and 
    a permutation test while shuffling cells only within the same cell type.

    Parameters
    ----------
    x_sparse : csr_matrix
        Sparse matrix of shape (n_cells, n_isoforms). Expected to contain PSI or
        relative expression values with explicit zeros; missing values should be NaN.
    labels : pandas.DataFrame
        Cell metadata indexed identically to the rows of `x_sparse`.
        Must contain `x_coord`, `y_coord`, `spot_class`, and `first_type`.
    isoform_ids : pandas.DataFrame
        Isoform names corresponding to the columns of `x_sparse`. 
        Must contain one column called 'Transcript ID'
    var_totest : list of str
        Subset of columns in `x` to test.
    nperm : int, default=10000
        Number of random permutations for cell-type assignment.
    k : int, default=10
        Number of neighbors for spatial weight matrix.
    x_coord, y_coord : str
        Column names in `labels` for spatial coordinates.
    output_dir : str
        Directory path to save result as CSV files.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    res : pandas.DataFrame
        Index = tested isoforms, columns = ['morans I','p-value (original)',
        'p-value (new)','Num cells','Imbalance'].
    
    """

    name_to_index = {name: idx for idx, name in enumerate(isoform_ids["Transcript ID"])}
    var_idx = [(v, name_to_index[v]) for v in var_totest]

    # Rank variables by non-zero count (descending)
    col_nnz = np.diff(x_sparse.tocsc()[:, [idx for _, idx in var_idx]].indptr)
    sorted_vars = [v for _, v in sorted(zip(col_nnz, var_idx), reverse=True, key=lambda t: t[0])]

    def _prepare_inputs_for_isoform(var_name: str, col_idx: int):
        col = x_sparse.getcol(col_idx).tocoo()
        nonzero_idx = col.row
        values = col.data

        labels_sub = labels.iloc[nonzero_idx]
        x_sub = pd.Series(values, index=labels_sub.index)

        return _compute_ctperm_for_variable(
            var_name, x_sub, labels_sub,
            nperm, k, x_coord, y_coord
        )

    if n_jobs == 1:
        results = [
            _prepare_inputs_for_isoform(var_name, col_idx)
            for var_name, col_idx in tqdm(sorted_vars, desc="Moran's I with ct-constrained perm.")
        ]
    else:
        with tqdm_joblib(desc="Moran's I with ct-constrained perm.", total=len(sorted_vars)):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_prepare_inputs_for_isoform)(var_name, col_idx)
                for var_name, col_idx in sorted_vars
            )

    res = pd.DataFrame(
        results,
        columns=[
            'variable', 'morans I', 'p-value (original)',
            'p-value (new)', 'Num cells', 'Imbalance'
        ]
    ).set_index('variable')

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_fn = f"{output_dir}/MoransI_sparse_pval_nperm{nperm}_k{k}_ctconstrained.csv"
        res.to_csv(out_fn)

    return res

    

def _compute_ctperm_for_variable(
    i, x_i, labels, nperm, k, x_coord, y_coord
):
    perm_I = []

    x_i = x_i.dropna()
    labels_i = labels.loc[x_i.index]

    # Keep only singlets and confident doublets
    keep_mask = (labels_i['spot_class'] == 'singlet') | (labels_i['spot_class'] == 'doublet_certain')
    labels_i = labels_i[keep_mask]
    x_i = x_i.loc[labels_i.index]

    imbalance = (x_i < 0.5).sum() / len(x_i)

    # Compute original Moran's I and permutation p-value
    w_i = _calculate_weight_matrix_sklearn(labels_i[[x_coord, y_coord]], k)
    res = esda.Moran(x_i, w_i, permutations=nperm, transformation='b', two_tailed=False)
    I_obs = res.I
    p_orig = res.p_sim

    # Pre-load label arrays for faster reuse
    first_type = labels_i['first_type'].values.copy()
    spot_class = labels_i['spot_class'].values.copy()
    first_type_weight = labels_i['first_type_weight'].values.copy()
    second_type = labels_i['second_type'].values.copy()
    doublet_certain_mask = spot_class == 'doublet_certain'
    cts = np.unique(first_type)

    for _ in range(nperm):
        # Probabilistic reassignment of doublets
        rand_vals = np.random.rand(np.sum(doublet_certain_mask))
        first_type[doublet_certain_mask] = np.where(
            rand_vals > first_type_weight[doublet_certain_mask],
            second_type[doublet_certain_mask],
            first_type[doublet_certain_mask]
        )

        # Cell-type constrained permutation
        x_i_temp = x_i.copy()
        for ct in cts:
            idx_ct = np.where(first_type == ct)[0]
            x_i_temp.iloc[idx_ct] = np.random.permutation(x_i_temp.iloc[idx_ct])

        # Moran without internal permutations
        res_perm = esda.Moran(x_i_temp, w_i, permutations=0, transformation='b')
        perm_I.append(res_perm.I)

    # Compute constrained p-value
    perm_I = np.array(perm_I)
    p_ct = (np.sum(perm_I >= I_obs) + 1) / (nperm + 1)

    return i, I_obs, p_orig, p_ct, len(x_i), imbalance
