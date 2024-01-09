from __future__ import annotations
from typing import List, Union
from functools import partial

from pydantic import field_validator, model_validator

from pymoors.core.modeling.expression import Expression


class Index(Expression):
    index: List[int]
    expression: Expression

    @property
    def size(self):
        return len(self.index)

    @staticmethod
    def __check_key_against_size(index: int, size: int) -> None:
        if (index <= -size) or (0 <= index <= size):
            return
        raise IndexError(
            f"index {index} is out of bounds for expression with size {size}"
        )

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
