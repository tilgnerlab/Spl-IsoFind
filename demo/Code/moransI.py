from pathlib import Path
import pandas as pd
import numpy as np

from tqdm.notebook import tqdm

from statsmodels.stats.multitest import fdrcorrection

from sklearn.neighbors import NearestNeighbors
from libpysal.weights import W
import esda

def moransI(x, labels, nperm=100000, k=10, 
            mincells=50, imb=0.05, mincellspergroup=20, 
            celltypes=['All','ExciteNeuron','InhibNeuron','Astro','Oligo'],
            x_coord = 'x', y_coord = 'y', output_dir=''):

    pval = pd.DataFrame(np.nan, index=x.columns, columns=celltypes)
    qval = pd.DataFrame(np.nan, index=x.columns, columns=celltypes)
    
    for i in tqdm(range(x.shape[1])):
    
        x_i = x.iloc[:,i].dropna()
        labels_i = labels.loc[x_i.index]
        
        _, counts = np.unique(x_i > 0.5, return_counts=True) # Check if we have enough high and low values
        
        if (len(x_i) >= mincells) & np.all(counts >= mincellspergroup) & (x_i.var() > 0):
        
            for ct in celltypes:
                if ct == 'All':
                    # Check imbalance
                    if (np.min(counts)/len(x_i) >= imb):
                        w = calculate_weight_matrix_sklearn(labels_i[[x_coord, y_coord]], k)
                        res = esda.Moran(x_i, w, permutations=nperm, transformation='b')
                        pval.loc[x.columns[i], ct] = res.p_sim
                else:
                    tokeep_ct = (labels_i['spot_class'] == 'singlet') & (labels_i['first_type'] == ct)
                    x_i_ct = x_i[tokeep_ct]
                    _, counts_ct = np.unique(x_i_ct > 0.5, return_counts=True)
                    
                    if (np.sum(tokeep_ct) >= mincells) & np.all(counts_ct >= mincellspergroup) & (x_i_ct.var() > 0):
                        if (np.min(counts_ct)/len(x_i_ct) >= imb):
                            w = calculate_weight_matrix_sklearn(labels_i.loc[tokeep_ct.values,[x_coord, y_coord]], k)
                            res = esda.Moran(x_i_ct, w, permutations=nperm, transformation='b')
                            pval.loc[x.columns[i], ct] = res.p_sim

    # BY FDR correction
    for ct in celltypes:
        tocorrect = pval[ct][pval[ct].notna()]
        _, q = fdrcorrection(tocorrect, method='n')
        qval[ct][pval[ct].notna()] = q

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_pval = f"{output_dir}/MoransI_pval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}.csv"
    output_qval = f"{output_dir}/MoransI_qval_nperm{nperm}_k{k}_mincells{mincells}_mincellspergroup{mincellspergroup}_imb{imb}.csv"
        
    pval.to_csv(output_pval)
    qval.to_csv(output_qval)
            
    return pval, qval


def calculate_weight_matrix_sklearn(locations, k):
    # Fit nearest neighbors model
    # k+1 cause we will remove the diagonal afterwards
    nn = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(locations)
    
    # Find k-nearest neighbors (distances and indices)
    _, indices = nn.kneighbors(locations)
    
    # Convert to libpysal object
    neighbors = {i: list(indices[i, 1:]) for i in range(indices.shape[0])}  # Remove self-reference
    weights = {i: [1] * len(neighbors[i]) for i in neighbors}  # Assign equal weights

    return W(neighbors, weights)

def moransI_ctperm(x, labels, variables, nperm = 10000, k = 10):

    p_original = []
    p_new = []
    num_cells = []
    imb = []

    for e in tqdm(variables):

        perm_I = []

        # Select exon of interest
        x_i = x[e].dropna()
        labels_i = labels.loc[x_i.index]

        # Focus on singlets and doublet certain
        labels_i = labels_i[((labels_i['spot_class'] == 'singlet') | 
                             (labels_i['spot_class'] == 'doublet_certain'))]
        x_i = x_i.loc[labels_i.index]

        num_cells.append(len(x_i))
        imb.append((x_i < 0.5).sum()/(len(x_i)))

        # Original moran's I and p-value
        w_i = calculate_weight_matrix_sklearn(labels_i[['x','y']], k)
        res = esda.Moran(x_i, w_i, permutations=nperm, transformation='b')
        
        p_original.append(res.p_sim)
        
        for _ in range(nperm):

            # Convert to NumPy arrays for faster access
            first_type = labels_i['first_type'].values.copy()
            spot_class = labels_i['spot_class'].values.copy()
            first_type_weight = labels_i['first_type_weight'].values.copy()
            second_type = labels_i['second_type'].values.copy()
            
            # Define our temp. cell-type labels for the doublets
            doublet_certain_mask = spot_class == 'doublet_certain'
            rand_vals = np.random.rand(np.sum(doublet_certain_mask))
            first_type[doublet_certain_mask] = np.where(
                rand_vals > first_type_weight[doublet_certain_mask],
                second_type[doublet_certain_mask],
                first_type[doublet_certain_mask]
            )
        
            # Permutate each cell-type separately
            cts = np.unique(first_type)
            x_i_temp = x_i.copy()
            for ct in cts:
                idx_ct = np.where(first_type == ct)[0]
                x_i_temp.iloc[idx_ct] = np.random.permutation(x_i_temp.iloc[idx_ct])
            
            # Calculate new statistic
            res_perm = esda.Moran(x_i_temp, w_i, permutations=0, transformation='b')
            perm_I.append(res_perm.I)
        
        # New p-value
        p = (np.sum(np.array(perm_I) >= res.I)+1)/(nperm+1)
        p_new.append(p)

    df = pd.DataFrame([p_original, p_new, num_cells, imb], columns=variables, 
            index=['Original', 'New', 'Num cells', 'Imbalance']).T
    
    return df