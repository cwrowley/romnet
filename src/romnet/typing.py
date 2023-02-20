from typing import Callable, Protocol
from numpy.typing import ArrayLike

# class Vector(ArrayLike, Protocol):
#     def __mul__(self, alpha: float) -> "Vector":
#         ...

#     def dot(self, x: "Vector") -> "Vector":
#         ...

Vector = ArrayLike
VectorField = Callable[[Vector], Vector]
