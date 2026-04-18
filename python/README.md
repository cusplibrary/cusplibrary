# pycusp

Python bindings for [CUSP](https://github.com/cusplibrary/cusplibrary), a C++
sparse linear algebra library.

CSR sparse matrix-vector multiply on CPU (NumPy) or CUDA
(CuPy / any array exposing `__cuda_array_interface__`).

## Install from source

`pycusp` is header-driven and compiles against CUSP and CCCL.
Set `CCCL_PATH` to the CCCL location before installing:

```bash
CCCL_PATH=~/repos/cccl pip install ./python
```

If CUDA is found, then CUDA is built
and `pycusp.HAS_CUDA` is  set to `True`.  The modes can be forced with
`-Cset.PYCUSP_CUDA=ON` or `-Cset.PYCUSP_CUDA=OFF`:

```bash
CCCL_PATH=~/repos/cccl pip install ./python -Cset.PYCUSP_CUDA=ON
```

For development with the test extras:

```bash
CCCL_PATH=~/repos/cccl pip install -e './python[test]' -v
pytest python/tests -v
```

CUDA tests require `cupy`, otherwise they are skipped.

## Example (CPU, NumPy)

```python
import numpy as np
import pycusp

# CSR for
#   [10  0 20]
#   [ 0  0  0]
#   [ 0  0 30]
#   [40 50 60]
A = pycusp.CsrMatrix(
    num_rows=4,
    num_cols=3,
    row_offsets=np.array([0, 2, 2, 3, 6], dtype=np.int32),
    column_indices=np.array([0, 2, 2, 0, 1, 2], dtype=np.int32),
    values=np.array([10, 20, 30, 40, 50, 60], dtype=np.float64),
)

x = np.ones(3, dtype=np.float64)
y = np.zeros(4, dtype=np.float64)

pycusp.spmv(A, x, y)
# y == [30., 0., 30., 150.]
```

## Example (CUDA, CuPy)

```python
import cupy as cp
import pycusp

assert pycusp.HAS_CUDA

A = pycusp.CsrMatrix(
    num_rows=4,
    num_cols=3,
    row_offsets=cp.asarray([0, 2, 2, 3, 6], dtype=cp.int32),
    column_indices=cp.asarray([0, 2, 2, 0, 1, 2], dtype=cp.int32),
    values=cp.asarray([10, 20, 30, 40, 50, 60], dtype=cp.float64),
)
assert A.device == "cuda"

x = cp.ones(3, dtype=cp.float64)
y = cp.zeros(4, dtype=cp.float64)
pycusp.spmv(A, x, y)
# cp.asnumpy(y) == [30., 0., 30., 150.]
```

All three arrays (`A`, `x`, `y`) should share one device and one dtype --- mixing
host and device storage will raise a `TypeError`.

## Supported types

- Indices: `int32`
- Values: `float32`, `float64`
- Memory: host (NumPy) or CUDA (CuPy / `__cuda_array_interface__`)
