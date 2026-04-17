import numpy as np
import pytest


@pytest.fixture
def small_csr():
    """The 4x3 example matrix from examples/Views/csr_raw.cu:
        [10  0 20]
        [ 0  0  0]
        [ 0  0 30]
        [40 50 60]
    """
    row_offsets = np.array([0, 2, 2, 3, 6], dtype=np.int32)
    column_indices = np.array([0, 2, 2, 0, 1, 2], dtype=np.int32)
    values = np.array([10, 20, 30, 40, 50, 60], dtype=np.float64)
    return row_offsets, column_indices, values
