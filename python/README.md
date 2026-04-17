# pycusp

Python bindings for [CUSP](https://github.com/cusplibrary/cusplibrary), a C++
sparse linear algebra library.

**Currently**: host-only CSR sparse matrix-vector multiply. Accepts NumPy arrays
and calls the sequential CPU `multiply` in CUSP. CUDA / CuPy
support is next.

## Install from source

`pycusp` is header-driven and compiles against CUSP and CCCL.
Set `CCCL_PATH` to the CCCL location before installing:

```bash
CCCL_PATH=~/repos/cccl pip install ./python
```

For development with the test extras:

```bash
CCCL_PATH=~/repos/cccl pip install -e './python[test]' -v
pytest python/tests -v
```

## Example

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

## Supported types

- Indices: `int32`
- Values: `float32`, `float64`
- Memory: host (NumPy) only
