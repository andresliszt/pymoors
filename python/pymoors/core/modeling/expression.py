from __future__ import annotations
import abc
from collections import Counter
from typing import TypeVar, Tuple, List, Union

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr, model_validator


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
    _shape: Tuple[int, ...] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_shape(self):
        self._shape = self._shape_from_expressions()
        return self

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Expression name"""

    @abc.abstractmethod
    def _shape_from_expressions(self) -> Tuple[int, ...]:
        "Expression shape"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def is_scalar(self) -> bool:
        return self.shape == ()

    @property
    def is_constant(self) -> bool:
        return False

    @helpers.cast_other_to_expression
    def __eq__(self, other):
        return helpers.equality()(lhs=self, rhs=other)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.name

    def __lt__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    @helpers.cast_other_to_expression
    def __le__(self, other: ExpressionLike):
        return helpers.inequeality()(lhs=self, rhs=other, type="<=")

    def __gt__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    @helpers.cast_other_to_expression
    def __ge__(self, other: ExpressionLike):
        return helpers.inequeality()(lhs=self, rhs=other, type=">=")

    def __getitem__(self, index: Union[int, List[int]]):
        return helpers.index()(index=index, expression=self, expression_id=self.expression_id)

    @helpers.cast_other_to_expression
    def __add__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return helpers.add_expression()(self, other)

    def __radd__(self, other: ExpressionLike):
        return self + other

    @helpers.cast_other_to_expression
    def __mul__(self, other: float):
        if not any(
            [isinstance(self, helpers.constant()), isinstance(other, helpers.constant())]
        ):
            raise TypeError("Currently multiplication by scalar is supported only")

        if not isinstance(other, Expression):
            raise TypeError(
                f"Trying to multiply by a non-expression object of type {type(other)}"
            )

        return helpers.multiply_expression()(other, self)

    def __rmul__(self, other: float):
        return self * other

    # @helpers.cast_other_to_expression
    # def __matmul__(self, other )

    @helpers.cast_other_to_expression
    def __sub__(self, other: ExpressionLike):
        if isinstance(other, helpers.constant()) and other.is_zero:
            return self
        return self + (-1 * other)

    @helpers.cast_other_to_expression
    def __rsub__(self, other: ExpressionLike):
        return -1 * (self - other)
