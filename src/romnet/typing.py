from typing import Any, Callable, Union

import numpy as np
import numpy.typing as npt

# Vector = Union[npt.NDArray[np.float64], npt.NDArray[np.complex64]]
# T = TypeVar("T", np.float64, np.complex64)
# Vector = npt.NDArray[T]
# Vector = npt.NDArray[np.float64]
# Vector = npt.NDArray[Any]
Vector = np.ndarray[Any, Any]
VectorField = Callable[[Vector], Vector]
VectorList = Union[list[Vector], npt.NDArray[np.float64]]
