from typing import Literal, Tuple

import numpy as np
from numpy.typing import NDArray

__version__: str

class CsrMatrix:
    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 row_offsets: NDArray[np.int32],
                 column_indices: NDArray[np.int32],
                 values: NDArray[np.floating],
                ) -> None: ...
    @property
    def num_rows(self) -> int: ...
    @property
    def num_cols(self) -> int: ...
    @property
    def num_entries(self) -> int: ...
    @property
    def shape(self) -> Tuple[int, int]: ...
    @property
    def dtype(self) -> Literal["float32", "float64"]: ...

def spmv(A: CsrMatrix,
         x: NDArray[np.floating],
         y: NDArray[np.floating],
        ) -> None: ...
