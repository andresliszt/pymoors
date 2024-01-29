from typing import Union, Iterable


import numpy as np


from pymoors.core.modeling.expression import Expression
from pymoors.core.helpers import cast_other_to_constant

from pymoors.typing import OneDArray


class Constant(Expression):
    value: Union[float, OneDArray]

    def __init__(self, value: Union[float, "Constant", Iterable], **kwargs) -> None:
        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, (int, float)):
            pass
        else:
            value = np.array(value)
        super().__init__(value=value, **kwargs)

    @property
    def size(self) -> int:
        return 1 if isinstance(self.value, float) else len(self.value)

    @property
    def is_zero(self) -> bool:
        return self.value == 0 if isinstance(self.value, float) else np.all(self.value == 0)

    @property
    def is_constant(self) -> bool:
        return True

    @property
    def constant(self) -> "Constant":
        return self

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
