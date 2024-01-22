from __future__ import annotations
import abc
from collections import Counter
from typing import TypeVar, List, Union

from pydantic import BaseModel, Field


from pymoors.core import helpers

ExpressionLike = TypeVar("ExpressionLike")


class IDGenerator:
    """Simple id generator for expression ids"""

    def __init__(self):
        self.counter = Counter()

    def generate_id(self):
        self.counter.update(["id"])
        return self.counter["id"]


id_generator = IDGenerator()


class Expression(BaseModel, abc.ABC):
    expression_id: int = Field(default_factory=id_generator.generate_id)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Expression name"""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        "Expression size"

    @property
    def is_constant(self) -> bool:
        return self.size == 0

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.name

    def __le__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    def __lt__(self, other: ExpressionLike):
        return helpers.inequeality()(lhs=self, rhs=other)

    def __ge__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    def __gt__(self, other: ExpressionLike):
        return helpers.inequeality()(lhs=other, rhs=self)

    def __getitem__(self, index: Union[int, List[int]]):
        return helpers.index()(
            index=index, expression=self, expression_id=self.expression_id
        )

    @helpers.cast_other_to_constant
    def __add__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return helpers.add_expression()(self, other)

    def __radd__(self, other: ExpressionLike):
        return self.__add__(other)

    @helpers.cast_other_to_constant
    def __mul__(self, other: float):
        if not isinstance(other, helpers.constant()):
            raise TypeError("Currently multiplication by scalar is supported only")
        return helpers.multiply_expression()(scalar=other.value, expression=self)

    def __rmul__(self, other: float):
        return self.__mul__(other)

    @helpers.cast_other_to_constant
    def __sub__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return self + (-1 * other)

    @helpers.cast_other_to_constant
    def __rsub__(self, other: ExpressionLike):
        return -1 * (self - other)
