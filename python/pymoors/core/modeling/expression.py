from __future__ import annotations
import abc
from collections import Counter
from typing import TypeVar, List, Union

from pydantic import BaseModel, Field


from pymoors.core.modeling import helpers
from pymoors.core.inequality import Inequality


ExpressionLike = TypeVar("ExpressionLike")


class IDGenerator:
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

    def is_constant(self) -> bool:
        return self.size == 0

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.name

    def __le__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    def __lt__(self, other: ExpressionLike) -> Inequality:
        return Inequality(lhs=self, rhs=other)

    def __ge__(self, other: ExpressionLike) -> None:
        return NotImplementedError

    def __gt__(self, other: ExpressionLike) -> Inequality:
        return Inequality(lhs=other, rhs=self)

    def __getitem__(self, index: Union[int, List[int]]):
        return helpers.index()(
            index=index, expression=self, expression_id=self.expression_id
        )

    def __add__(self, other):
        return helpers.add_expression()(self, other)

    def __radd__(self, other):
        return helpers.add_expression()(other, self)
