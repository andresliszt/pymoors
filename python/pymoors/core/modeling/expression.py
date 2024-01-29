from __future__ import annotations
import abc
from collections import Counter
from typing import TypeVar, List, Union

from pydantic import BaseModel, Field, ConfigDict


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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Expression name"""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        "Expression size"

    @property
    def expressions(self) -> List[ExpressionLike]:
        return [self]

    @property
    def constant(self):
        return helpers.constant()(value=0)

    @property
    def non_constant_expressions(self) -> List[Expression]:
        return [expr for expr in self.expressions if not isinstance(expr, helpers.constant())]

    @property
    def is_constant(self) -> bool:
        return False

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
        return helpers.index()(index=index, expression=self, expression_id=self.expression_id)

    @helpers.cast_other_to_constant
    def __add__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return helpers.add_expression()(self, other)

    def __radd__(self, other: ExpressionLike):
        return self + other

    @helpers.cast_other_to_constant
    def __mul__(self, other: float):
        if not any(
            [isinstance(self, helpers.constant()), isinstance(other, helpers.constant())]
        ):
            raise TypeError("Currently multiplication by scalar is supported only")

        if not isinstance(other, Expression):
            raise TypeError(
                f"Trying to multiply by a non-expression object of type {type(other)}"
            )

        return helpers.multiply_expression()(other.value, self)

    def __rmul__(self, other: float):
        return self * other

    @helpers.cast_other_to_constant
    def __sub__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return self + (-1 * other)

    @helpers.cast_other_to_constant
    def __rsub__(self, other: ExpressionLike):
        return -1 * (self - other)
