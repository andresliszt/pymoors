from typing import Union

from pymoors.core.modeling.expression import Expression
from pymoors.core.helpers import cast_other_to_constant


class Constant(Expression):
    value: float

    def __init__(self, value: Union[float, "Constant"], **kwargs) -> None:
        if isinstance(value, Constant):
            value = value.value
        super().__init__(value=value, **kwargs)

    @property
    def size(self) -> int:
        """Constant size is always zero"""
        return 0

    @property
    def is_zero(self) -> bool:
        return self.value == 0

    @property
    def name(self) -> int:
        return f"constant(value = {self.value})"

    @cast_other_to_constant
    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value + other.value)
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    @cast_other_to_constant
    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value - other.value)
        return super().__sub__(other)

    @cast_other_to_constant
    def __rsub__(self, other):
        return other.__sub__(self)

    @cast_other_to_constant
    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value * other.value)
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
