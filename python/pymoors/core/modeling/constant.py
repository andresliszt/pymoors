from typing import Union, Iterable, Tuple


import numpy as np


from pymoors.core.modeling.expression import Expression
from pymoors.core.helpers import cast_other_to_expression

from pymoors.typing import OneDArray, TwoDArray


class Constant(Expression):
    value: Union[OneDArray, TwoDArray]

    def __init__(
        self, value: Union[float, "Constant", Iterable, OneDArray, TwoDArray], **kwargs
    ) -> None:
        if isinstance(value, Constant):
            value = value.value
        value = np.array(value)
        super().__init__(value=value, **kwargs)

    @property
    def size(self) -> int:
        return 1 if isinstance(self.value, float) else len(self.value)

    def _shape_from_expressions(self) -> Tuple[int, ...]:
        return self.value.shape

    @property
    def is_zero(self) -> bool:
        return self.value == 0 if isinstance(self.value, float) else np.all(self.value == 0)

    @property
    def is_constant(self) -> bool:
        return True

    @property
    def name(self) -> int:
        return f"constant(value = {self.value})"

    @cast_other_to_expression
    def __add__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value + other.value)
        return super().__add__(other)

    def __radd__(self, other):
        return self.__add__(other)

    @cast_other_to_expression
    def __sub__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value - other.value)
        return super().__sub__(other)

    @cast_other_to_expression
    def __rsub__(self, other):
        return other.__sub__(self)

    @cast_other_to_expression
    def __mul__(self, other):
        if isinstance(other, Constant):
            return Constant(value=self.value * other.value)
        return super().__mul__(other)

    def __rmul__(self, other):
        return self.__mul__(other)
