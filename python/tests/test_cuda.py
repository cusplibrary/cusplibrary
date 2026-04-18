import numpy as np
import pytest

import pycusp

cp = pytest.importorskip("cupy")
sp = pytest.importorskip("scipy.sparse")

if not pycusp.HAS_CUDA:
    pytest.skip("pycusp was built without CUDA support",
                allow_module_level=True)


@pytest.fixture
def small_csr_gpu():
    row_offsets = cp.asarray([0, 2, 2, 3, 6], dtype=cp.int32)
    column_indices = cp.asarray([0, 2, 2, 0, 1, 2], dtype=cp.int32)
    values = cp.asarray([10, 20, 30, 40, 50, 60], dtype=cp.float64)
    return row_offsets, column_indices, values


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_small_gpu_matches_csr_raw_example(dtype, small_csr_gpu):
    row_offsets, column_indices, values = small_csr_gpu
    A = pycusp.CsrMatrix(
        num_rows=4,
        num_cols=3,
        row_offsets=row_offsets,
        column_indices=column_indices,
        values=values.astype(dtype),
    )
    assert A.device == "cuda"
    x = cp.ones(3, dtype=dtype)
    y = cp.zeros(4, dtype=dtype)
    pycusp.spmv(A, x, y)
    np.testing.assert_allclose(cp.asnumpy(y), [30, 0, 30, 150])


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("seed", [0, 7, 123])
def test_random_sparse_gpu_matches_scipy(dtype, seed):
    rng = np.random.default_rng(seed)
    n_rows, n_cols, density = 83, 64, 0.15
    S = sp.random(n_rows, n_cols, density=density, format="csr",
                  dtype=dtype, random_state=rng)
    S.sort_indices()

    A = pycusp.CsrMatrix(
        n_rows, n_cols,
        cp.asarray(S.indptr.astype(np.int32)),
        cp.asarray(S.indices.astype(np.int32)),
        cp.asarray(S.data),
    )
    x_host = rng.standard_normal(n_cols).astype(dtype)
    x = cp.asarray(x_host)
    y = cp.zeros(n_rows, dtype=dtype)
    pycusp.spmv(A, x, y)

    atol = 1e-4 if dtype == cp.float32 else 1e-11
    np.testing.assert_allclose(cp.asnumpy(y), S @ x_host, atol=atol)


def test_mixed_device_raises(small_csr):
    host_ro, host_ci, host_vals = small_csr
    A = pycusp.CsrMatrix(4, 3, host_ro, host_ci, host_vals)
    x_gpu = cp.ones(3, dtype=cp.float64)
    y_host = np.zeros(4, dtype=np.float64)
    with pytest.raises(TypeError):
        pycusp.spmv(A, x_gpu, y_host)


def test_mixed_device_in_matrix_construction_raises():
    host_ro = np.array([0, 2, 2, 3, 6], dtype=np.int32)
    gpu_ci = cp.asarray([0, 2, 2, 0, 1, 2], dtype=cp.int32)
    host_vals = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    with pytest.raises(TypeError):
        pycusp.CsrMatrix(4, 3, host_ro, gpu_ci, host_vals)


def test_empty_gpu_matrix_zero_nnz():
    row_offsets = cp.zeros(4, dtype=cp.int32)
    column_indices = cp.zeros(0, dtype=cp.int32)
    values = cp.zeros(0, dtype=cp.float64)
    A = pycusp.CsrMatrix(3, 5, row_offsets, column_indices, values)
    x = cp.arange(5, dtype=cp.float64)
    y = cp.full(3, 99.0)
    pycusp.spmv(A, x, y)
    np.testing.assert_array_equal(cp.asnumpy(y), np.zeros(3))


def test_device_property(small_csr_gpu):
    row_offsets, column_indices, values = small_csr_gpu
    A = pycusp.CsrMatrix(4, 3, row_offsets, column_indices, values)
    assert A.device == "cuda"
    assert A.dtype == "float64"
    assert A.shape == (4, 3)
