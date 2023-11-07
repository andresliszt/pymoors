from __future__ import annotations
import abc
from functools import partial
from typing import TypeVar, List, Union

from pydantic import BaseModel, model_validator, field_validator

from pymoors.core.inequality import Inequality


ExpressionLike = TypeVar("ExpressionLike")


class Expression(BaseModel, abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Expression name"""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        "Expression size"

    def __le__(self, other: ExpressionLike):
        return Inequality(lhs=self, rhs=other)

    def __lt__(self, other: ExpressionLike):
        return Inequality(lhs=self, rhs=other)

    def __ge__(self, other: ExpressionLike):
        return Inequality(lhs=other, rhs=self)

    def __gt__(self, other: ExpressionLike):
        return Inequality(lhs=other, rhs=self)

    def __getitem__(self, index: Union[int, List[int]]) -> Index:
        return Index(index=index)


class Index(Expression):
    index: Union[List[int], int]

    @staticmethod
    def __check_key_against_size(index: int, size: int) -> None:
        if (index <= -size) or (0 <= index <= size):
            return
        raise IndexError(f"index {index} is out of bounds for expression with size {size}")

    @field_validator("index", mode="before")
    def cast_index_to_list(cls, index: Union[List[int], int]):
        return index if isinstance(index, list) else [index]

    @model_validator(mode="after")
    def validate_key_and_size(self) -> Index:
        # Check against all elements on the list
        list(map(partial(self.__check_key_against_size, size=self.size), self.index))
        return self

    @property
    def name(self) -> str:
        """Name representation of the Index"""
        return f"index[{str(self.index).replace('[', '').replace(']', '')}]"
