# pylint:disable=C0415
import functools
from typing import Any


def add_expression():
    from pymoors.core.modeling.add import AddExpression

    return AddExpression


def constant():
    from pymoors.core.modeling.constant import Constant

    return Constant


def cast_to_constant(value):
    if isinstance(value, (int, float, list)):
        if isinstance(value, list):
            if not all(isinstance(v, (int, float)) for v in value):
                raise TypeError("If value is a list, all elements must be numbers")
        from pymoors.core.modeling.constant import Constant

        return Constant(value=value)
    return value


def cast_other_to_expression(meth):
    @functools.wraps(meth)
    def wrapper_cast_to_constant(self, other):
        return meth(self, cast_to_constant(other))

    return wrapper_cast_to_constant


def equality():
    from pymoors.core.constraints.inequality import Equality

    return Equality


def index():
    from pymoors.core.modeling.index import Index

    return Index


def inequeality():
    from pymoors.core.constraints.inequality import Inequality

    return Inequality


def multiply_expression():
    from pymoors.core.modeling.multiply import MultiplyExpression

    return MultiplyExpression
