from __future__ import annotations

import numbers
from typing import ClassVar
from pymoors.expression import Expression

from pymoors.helpers import cast_other_to_constant


class Constant(Expression):
    is_constant: ClassVar[bool] = True

    _value: numbers.Number

    __slots__ = ("_value",)
    __hash__ = Expression.__hash__

    def __new__(cls, value: numbers.Number):
        obj = Expression.__new__(cls)
        obj._value = value
        return obj

    def __repr__(self) -> str:
        return str(self.value)

    def _hashable_content(self) -> tuple[Constant]:
        return (self.value,)

    @property
    def value(self) -> numbers.Number:
        return self._value

    @cast_other_to_constant
    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.value == other.value
        return False

    @cast_other_to_constant
    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value + other.value)
        return super().__add__(other)

    @cast_other_to_constant
    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(self.value * other.value)
        return super().__mul__(other)
