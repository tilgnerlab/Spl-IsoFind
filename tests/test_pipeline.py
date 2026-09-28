import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import SplIsoFind

CELLTYPES = ['All', 'ExciteNeuron', 'Astro']


def read_allinfo(fn):
    return pd.read_csv(fn, sep='\t', header=None, index_col=0)


# preprocessing ---------------------------------------------------------------

def test_allinfo_addct_labels(dataset, labeled):
    out = read_allinfo(labeled)
    obs = dataset['obs']

    # unspliced reads, reads outside cells, and OtherHemisphere cells are gone
    assert (out[10] > 0).all()
    assert not out[3].eq('NOTACELL').any()
    other_hemi = set(obs.index[obs['subregion'] == 'OtherHemisphere'].str[4:].astype(int))
    cells = out[3].str[2:6].astype(int)
    assert not cells.isin(other_hemi).any()

    # labels are celltype_subregion; doublets become "other"
    assert out[2].notna().all()
    first = cells.map(obs.set_index('CellID-original')['first_type'].astype(str))
    doublet = cells.map(obs.set_index('CellID-original')['spot_class'].astype(str)) != 'singlet'
    assert out.loc[doublet.values, 2].str.startswith('other_').all()
    assert (out.loc[~doublet.values, 2].str.split('_').str[0] == first[~doublet.values]).all()


def test_create_auxiliary_files(labeled, tmp_path):
    SplIsoFind.pp.create_auxiliary_files(str(labeled), str(tmp_path))
    iso = pd.read_csv(tmp_path / 'Iso-IsoID.csv', sep='\t')
    assert set(iso['Gene']) == {'G1', 'G2'}  # single-isoform G3 is dropped
    assert (tmp_path / 'NumIsoPerCluster').stat().st_size > 0


def test_isoform_matrix_values(dataset, labeled, matrix):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    assert x.shape == (len(dataset['obs']), len(isoforms))
    assert set(isoforms['Gene ID']) == {'G1', 'G2'}
    assert (x.data == 0).any(), 'explicit zeros must be kept'

    # recompute the PSI of every stored entry directly from the reads
    reads = read_allinfo(labeled)
    cid = pd.read_csv(dataset['cidmap'], sep='\t').set_index('barcode')['CellID-original']
    reads['cell'] = reads[3].map(cid).astype(int)
    per_cell = reads.groupby(['cell', 1, 11]).size()
    x = x.tocsc()
    for j in range(x.shape[1]):
        gene, tr = isoforms.iloc[j][['Gene ID', 'Transcript ID']]
        col = x[:, j].tocoo()
        for row, val in zip(col.row, col.data):
            cell = int(labels['CellID-original'].iloc[row])
            counts = per_cell.loc[cell, gene]
            assert val == pytest.approx(counts.get(tr, 0) / counts.sum())


def test_sparse2df_matches_load_sparse(matrix):
    x_df, labels_df = SplIsoFind.pp.sparse2df(str(matrix))
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    dense = x.toarray()
    stored = x.copy()
    stored.data[:] = 1
    dense[stored.toarray() == 0] = np.nan
    np.testing.assert_array_equal(x_df.to_numpy(), dense)
    assert list(x_df.columns) == list(isoforms['Transcript ID'])


# spatial statistics ----------------------------------------------------------

@pytest.fixture(scope='module')
def moran(matrix):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    return SplIsoFind.sv.moransI_sparse(x, labels, isoforms, nperm=199,
                                        celltypes=CELLTYPES, n_jobs=1)


def test_moransI_detects_spatial_isoform(moran):
    mI, pval, qval = moran
    assert mI.loc['T1.1', 'All'] > 0.3
    assert qval.loc['T1.1', 'All'] < 0.05
    assert qval.loc['T2.1', 'All'] > 0.05


def test_qvalues_are_filled(moran):
    # regression: chained assignment left q-values all NaN under pandas copy-on-write
    mI, pval, qval = moran
    for ct in CELLTYPES:
        assert qval[ct].notna().sum() == pval[ct].notna().sum() > 0
    assert (qval.fillna(1) >= pval.fillna(1) - 1e-12).all().all()


def test_moransI_dense_matches_sparse(matrix, moran):
    x_df, labels_df = SplIsoFind.pp.sparse2df(str(matrix))
    mI_d, pval_d, qval_d = SplIsoFind.sv.moransI(x_df, labels_df, nperm=199,
                                                 celltypes=CELLTYPES, n_jobs=1)
    mI, pval, qval = moran
    pd.testing.assert_frame_equal(mI_d, mI, check_names=False)
    pd.testing.assert_frame_equal(pval_d, pval, check_names=False)


def test_moransI_parallel_matches_serial(matrix, moran):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    mI, pval, _ = SplIsoFind.sv.moransI_sparse(x, labels, isoforms, nperm=199,
                                               celltypes=CELLTYPES, n_jobs=2)
    pd.testing.assert_frame_equal(mI, moran[0])
    pd.testing.assert_frame_equal(pval, moran[1])


def test_moransI_ctperm_sparse(matrix):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    np.random.seed(0)
    res = SplIsoFind.sv.moransI_ctperm_sparse(x, labels, isoforms, var_totest=['T1.1', 'T2.1'],
                                              nperm=49, n_jobs=1)
    assert list(res.columns) == ['morans I', 'p-value (original)', 'p-value (new)',
                                 'Num cells', 'Imbalance']
    assert res.loc['T1.1', 'p-value (new)'] < 0.05


def test_moransI_ctperm_dense_parallel(matrix):
    # regression: n_jobs > 1 referenced an undefined variable
    x_df, labels_df = SplIsoFind.pp.sparse2df(str(matrix))
    res = SplIsoFind.sv.moransI_ctperm(x_df, labels_df, var_totest=['T1.1', 'T2.1'],
                                       nperm=19, n_jobs=2)
    assert set(res.index) == {'T1.1', 'T2.1'}


# plotting --------------------------------------------------------------------

def test_spatial_hexplot_sparse(matrix):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    ax = SplIsoFind.pl.spatial_hexplot_sparse(x, labels, isoforms, varName='T1.1', hexsize=20)
    assert ax is not None
    ax = SplIsoFind.pl.spatial_hexplot_sparse(x, labels, isoforms, varName='T1.1', hexsize=20,
                                              celltype='ExciteNeuron')
    assert ax.get_title().startswith('ExciteNeuron')


def test_barplot_pi(matrix):
    # regression: column was created as "PI" but aggregated as "PSI"
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    ax = SplIsoFind.pl.barplot_pi(x, labels, isoforms, 'T1.1')
    assert len(ax.patches) == 2  # Cortex and HPC


def test_read_results(tmp_path):
    base = tmp_path / 'demo' / 'res_scisorseqr' / 'CellTypes_All_All' / 'TreeTraversal_Iso'
    for sub, fdr in [('A_B', [0.01, 0.5]), ('A_C', [0.01, 0.01])]:
        (base / sub).mkdir(parents=True)
        pd.DataFrame({'FDR': fdr, 'dPI': [0.2, 0.2]}).to_csv(base / sub / 'x_results.csv', sep='\t')
    (base / 'empty_dir').mkdir()
    res = SplIsoFind.pl.read_results(str(tmp_path), 'demo', 'All', 'All').set_index(['reg1', 'reg2'])
    assert res.loc[('A', 'B'), 'sig'] == 1
    assert res.loc[('A', 'C'), 'perc'] == 100
    assert len(res) == 2


# cell-type constrained permutation -------------------------------------------

def test_doublet_reassignment_is_redrawn_each_permutation():
    # regression: reassignments used to accumulate, so after ~20 permutations
    # nearly every doublet carried its second type
    from SplIsoFind.spatially_variable import _draw_cell_types
    n = 2000
    first = np.array(['A'] * n, dtype=object)
    second = np.array(['B'] * n, dtype=object)
    weight = np.full(n, 0.7)
    doublet = np.arange(n) % 2 == 0
    rng = np.random.default_rng(0)
    fractions = []
    for _ in range(200):
        ct = _draw_cell_types(first, second, weight, doublet, rng)
        assert (ct[~doublet] == 'A').all()      # singlets never change
        fractions.append((ct[doublet] == 'B').mean())
    assert np.mean(fractions) == pytest.approx(0.3, abs=0.01)
    assert fractions[-1] == pytest.approx(0.3, abs=0.05)
    assert (first == 'A').all()                 # inputs are not modified


def test_permutation_covers_types_only_seen_as_second_type():
    # regression: types were fixed from first_type before reassignment, so
    # cells reassigned to a type absent from first_type were never shuffled
    from SplIsoFind.spatially_variable import _permute_within_types
    x = np.arange(100, dtype=float)
    cell_type = np.array(['A'] * 50 + ['Rare'] * 50, dtype=object)
    rng = np.random.default_rng(0)
    xp = _permute_within_types(x, cell_type, rng)
    assert sorted(xp[:50]) == list(x[:50])      # values stay within their type
    assert sorted(xp[50:]) == list(x[50:])
    assert not np.array_equal(xp[50:], x[50:])  # but 'Rare' cells are shuffled


def test_moransI_ctperm_reproducible(matrix):
    x, labels, isoforms = SplIsoFind.pp.load_sparse(str(matrix))
    kw = dict(var_totest=['T1.1', 'T2.1'], nperm=49)
    a = SplIsoFind.sv.moransI_ctperm_sparse(x, labels, isoforms, n_jobs=1, seed=3, **kw)
    b = SplIsoFind.sv.moransI_ctperm_sparse(x, labels, isoforms, n_jobs=2, seed=3, **kw)
    c = SplIsoFind.sv.moransI_ctperm_sparse(x, labels, isoforms, n_jobs=1, seed=3,
                                            var_totest=['T2.1', 'T1.1'], nperm=49)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(a.sort_index(), c.sort_index())

    x_df, labels_df = SplIsoFind.pp.sparse2df(str(matrix))
    d = SplIsoFind.sv.moransI_ctperm(x_df, labels_df, n_jobs=1, seed=3, **kw)
    pd.testing.assert_frame_equal(a.sort_index(), d.sort_index())


# spatial weights ---------------------------------------------------------------

def _brute_force_knn(loc, k):
    """Reference: k nearest by (squared distance, index), self excluded."""
    out = []
    for i in range(len(loc)):
        d2 = ((loc - loc[i]) ** 2).sum(axis=1)
        order = sorted((d, j) for j, d in enumerate(d2) if j != i)
        out.append(sorted(j for _, j in order[:k]))
    return out


@pytest.mark.parametrize('layout', ['grid', 'random', 'duplicates'])
def test_knn_weights_break_ties_deterministically(layout):
    # regression: a kd-tree resolves equal distances differently across
    # machines, which changed Moran's I in the 4th decimal between platforms
    from SplIsoFind.spatially_variable import _calculate_weight_matrix_sklearn
    rng = np.random.default_rng(1)
    if layout == 'grid':            # integer grid: nearly every cell has ties
        gx, gy = np.meshgrid(np.arange(15), np.arange(15))
        loc = np.c_[gx.ravel(), gy.ravel()].astype(float)
    elif layout == 'random':
        loc = rng.random((300, 2)) * 100
    else:                           # several cells sharing coordinates
        loc = np.repeat(rng.integers(0, 20, size=(60, 2)), 3, axis=0).astype(float)
    k = 10
    w = _calculate_weight_matrix_sklearn(pd.DataFrame(loc), k)
    expected = _brute_force_knn(loc, k)
    for i in range(len(loc)):
        assert i not in w.neighbors[i]
        assert sorted(w.neighbors[i]) == expected[i]
