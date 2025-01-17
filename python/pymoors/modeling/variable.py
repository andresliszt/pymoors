from typing import ClassVar, TypeVar

from pymoors.expression import Expression


MatrixLike = TypeVar("MatrixLike")


class Variable(Expression):
    is_variable: ClassVar[bool] = True

    _name: str
    _length: int

    __slots__ = ("_name", "_length")

    def __new__(cls, name: str, length: int = 1):
        # Variable expression doesn't have any other extra arg
        # pylint: disable=E0237
        obj = Expression.__new__(cls)
        obj._name = name
        obj._length = length
        return obj

    @property
    def name(self) -> str:
        return self._name

    @property
    def length(self) -> int:
        return self._length

    def __str__(self) -> str:
        return f"Variable({self.name}, {self.length})"

    def __repr__(self):
        return self.__str__()

    def __matmul__(self, other) -> Expression:
        return NotImplementedError

    def _hashable_content(self) -> tuple[str, int, int]:
        return (self.name, self.length)

    def __getitem__(self, index: int):
        if self.length == 1:
            raise ValueError
        if not -self.length < index < self.length:
            raise ValueError
        # TODO: Cache property?
        return Variable(name=f"{self.name}_{index}", length=1)
