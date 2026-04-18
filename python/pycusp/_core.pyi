from typing import Any, Literal, Tuple

import numpy as np
from numpy.typing import NDArray

__version__: str
HAS_CUDA: bool

class CsrMatrix:
    def __init__(self,
                 num_rows: int,
                 num_cols: int,
                 row_offsets: NDArray[np.int32] | Any,
                 column_indices: NDArray[np.int32] | Any,
                 values: NDArray[np.floating] | Any,
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
    @property
    def device(self) -> Literal["cpu", "cuda"]: ...

def spmv(A: CsrMatrix,
         x: NDArray[np.floating] | Any,
         y: NDArray[np.floating] | Any,
        ) -> None: ...
