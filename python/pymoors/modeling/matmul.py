from __future__ import annotations
from typing import ClassVar, Type
import numbers
from collections import defaultdict

from pymoors.expression import MathOperator, Expression, ExpressionLike

from pymoors.modeling.constant import Constant
from pymoors.modeling.pow import Pow
from pymoors.helpers import cast_other_to_expression

# pylint: disable=C0415

class Indentity:
    pass

class NullMatrix:
    pass


class MatMul(MathOperator):
    is_mul: ClassVar[bool] = True
    identity: ClassVar[Type[Indentity]] = Indentity
    anihilator: ClassVar[Type[NullMatrix]] = NullMatrix

    _coeff: Constant

    @cast_other_to_expression
    def check_eq(self, other):
        return super().check_eq(other)