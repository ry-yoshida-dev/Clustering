from numpy.typing import ArrayLike, DTypeLike

class coo_array:
    shape: tuple[int, int]
    nnz: int

    def __init__(
        self,
        arg1: tuple[ArrayLike, tuple[ArrayLike, ArrayLike]],
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None: ...
