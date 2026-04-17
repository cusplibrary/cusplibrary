import numpy as np
import pytest
import scipy.sparse as sp

import pycusp


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_small_matches_csr_raw_example(dtype, small_csr):
    row_offsets, column_indices, values = small_csr
    A = pycusp.CsrMatrix(
        num_rows=4,
        num_cols=3,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values.astype(dtype),
    )
    x = np.ones(3, dtype=dtype)
    y = np.zeros(4, dtype=dtype)
    pycusp.spmv(A, x, y)
    np.testing.assert_allclose(y, [30, 0, 30, 150])


def test_matrix_properties(small_csr):
    row_offsets, column_indices, values = small_csr
    A = pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
    assert A.num_rows == 4
    assert A.num_cols == 3
    assert A.num_entries == 6
    assert A.shape == (4, 3)
    assert A.dtype == "float64"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", [0, 1, 42])
def test_random_sparse_matches_scipy(dtype, seed):
    rng = np.random.default_rng(seed)
    n_rows, n_cols, density = 57, 43, 0.2
    S = sp.random(n_rows, n_cols, density=density, format="csr", dtype=dtype,
                  random_state=rng)
    S.sort_indices()

    A = pycusp.CsrMatrix(
        n_rows, n_cols,
        S.indptr.astype(np.int32),
        S.indices.astype(np.int32),
        S.data,
    )
    x = rng.standard_normal(n_cols).astype(dtype)
    y = np.zeros(n_rows, dtype=dtype)
    pycusp.spmv(A, x, y)

    atol = 1e-5 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(y, S @ x, atol=atol)


def test_empty_matrix_zero_nnz():
    row_offsets = np.zeros(4, dtype=np.int32)
    column_indices = np.zeros(0, dtype=np.int32)
    values = np.zeros(0, dtype=np.float64)
    A = pycusp.CsrMatrix(3, 5, row_offsets, column_indices, values)
    x = np.arange(5, dtype=np.float64)
    y = np.full(3, 99.0)
    pycusp.spmv(A, x, y)
    np.testing.assert_array_equal(y, np.zeros(3))


def test_dtype_mismatch_raises(small_csr):
    row_offsets, column_indices, values = small_csr
    A = pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
    x = np.ones(3, dtype=np.float32)
    y = np.zeros(4, dtype=np.float64)
    with pytest.raises(TypeError):
        pycusp.spmv(A, x, y)


def test_wrong_x_shape_raises(small_csr):
    row_offsets, column_indices, values = small_csr
    A = pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
    x = np.ones(4, dtype=np.float64)
    y = np.zeros(4, dtype=np.float64)
    with pytest.raises(ValueError):
        pycusp.spmv(A, x, y)


def test_wrong_y_shape_raises(small_csr):
    row_offsets, column_indices, values = small_csr
    A = pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
    x = np.ones(3, dtype=np.float64)
    y = np.zeros(5, dtype=np.float64)
    with pytest.raises(ValueError):
        pycusp.spmv(A, x, y)


def test_row_offsets_wrong_length_raises(small_csr):
    row_offsets, column_indices, values = small_csr
    bad = row_offsets[:-1].copy()
    with pytest.raises(ValueError):
        pycusp.CsrMatrix(4, 3, bad, column_indices, values)


def test_unsupported_dtype_raises(small_csr):
    row_offsets, column_indices, _ = small_csr
    values = np.ones(6, dtype=np.int64)
    with pytest.raises(TypeError):
        pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
