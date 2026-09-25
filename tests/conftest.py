"""
Synthetic stereo-seq-like dataset shared by the tests.

400 cells on a 20 x 20 grid, two barcodes per cell. Gene G1 switches isoform
between the left and right half of the tissue (spatially variable); gene G2
picks its isoform at random (not spatially variable); gene G3 has a single
isoform and should always be filtered out.
"""
import anndata as ad
import numpy as np
import pandas as pd
import pytest

N_SIDE = 20
READS_PER_GENE = 4


def _write_dataset(d):
    rng = np.random.default_rng(0)
    n = N_SIDE * N_SIDE
    gx, gy = np.meshgrid(np.arange(N_SIDE), np.arange(N_SIDE), indexing='ij')
    x = gx.ravel() * 10
    y = gy.ravel() * 10

    obs = pd.DataFrame({
        'x': x,
        'y': y,
        'CellID-original': np.arange(n),
        'first_type': pd.Categorical(np.where(np.arange(n) % 2 == 0, 'ExciteNeuron', 'Astro')),
        'second_type': pd.Categorical(np.where(np.arange(n) % 2 == 0, 'Astro', 'ExciteNeuron')),
        'spot_class': pd.Categorical(np.where(np.arange(n) % 10 == 0, 'doublet_certain', 'singlet')),
        'first_type_weight': 0.8,
        'region': pd.Categorical(np.where(y < 100, 'Cortex', 'HPC')),
        'subregion': pd.Categorical(np.where(y < 100, 'Cortex', 'HPC')),
        'sample': pd.Categorical(['S1'] * n),
    }, index=[f'cell{i}' for i in range(n)])
    # a few cells in the other hemisphere, which allinfo_addct drops
    obs['subregion'] = obs['subregion'].cat.add_categories(['OtherHemisphere'])
    obs.loc[obs.index[-5:], 'subregion'] = 'OtherHemisphere'
    ad.AnnData(obs=obs).write_h5ad(d / 'cells.h5ad')

    barcodes = [f'BC{i:04d}{k}' for i in range(n) for k in 'AB']
    cidmap = pd.DataFrame({
        'x': np.repeat(x, 2), 'y': np.repeat(y, 2),
        'CellID-original': np.repeat(np.arange(n), 2).astype(float),
        'barcode': barcodes,
    })
    cidmap.to_csv(d / 'cidmap.tsv.gz', sep='\t', index=False)

    rows = []
    for cell in range(n):
        left = x[cell] < N_SIDE * 5
        for r in range(READS_PER_GENE):
            bc = barcodes[2 * cell + (r % 2)]
            p1 = 0.9 if left else 0.1
            rows.append(('G1', bc, 'T1.1' if rng.random() < p1 else 'T1.2', 3))
            rows.append(('G2', bc, 'T2.1' if rng.random() < 0.5 else 'T2.2', 2))
            rows.append(('G3', bc, 'T3.1', 1))
    # unspliced reads (filtered by allinfo_addct) and reads outside any cell
    rows += [('G1', barcodes[0], 'T1.1', 0)] * 10
    rows += [('G1', 'NOTACELL', 'T1.1', 3)] * 10

    allinfo = pd.DataFrame({
        1: [r[0] for r in rows],
        2: 'None',
        3: [r[1] for r in rows],
        4: 'UMI',
        5: 'chain',
        6: 'NoTSS',
        7: 'NoPolyA',
        8: 'introns',
        9: 'known',
        10: [r[3] for r in rows],
        11: [r[2] for r in rows],
        12: 'protein_coding',
    }, index=[f'read{i}' for i in range(len(rows))])
    allinfo.to_csv(d / 'allinfo.gz', sep='\t', header=False, compression='gzip')
    return allinfo, cidmap, obs


@pytest.fixture(scope='session')
def dataset(tmp_path_factory):
    d = tmp_path_factory.mktemp('data')
    allinfo, cidmap, obs = _write_dataset(d)
    return {
        'dir': d,
        'allinfo': d / 'allinfo.gz',
        'cidmap': d / 'cidmap.tsv.gz',
        'adata': d / 'cells.h5ad',
        'obs': obs,
    }


@pytest.fixture(scope='session')
def labeled(dataset):
    import SplIsoFind
    fn = dataset['dir'] / 'allinfo.labeled.gz'
    SplIsoFind.pp.allinfo_addct(str(dataset['allinfo']), str(dataset['cidmap']),
                                str(dataset['adata']), str(fn))
    return fn


@pytest.fixture(scope='session')
def matrix(dataset, labeled):
    import SplIsoFind
    out = dataset['dir'] / 'isoform_matrix'
    SplIsoFind.pp.create_isoform_matrix(str(labeled), str(dataset['cidmap']),
                                        str(dataset['adata']), str(out))
    return out
