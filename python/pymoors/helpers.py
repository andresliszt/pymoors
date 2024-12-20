import functools
import numbers

# pylint: disable=C0415


def add_expression():
    from pymoors.modeling.add import Add

    return Add


def mul_expression():
    from pymoors.modeling.mul import Mul

    return Mul


def pow_expression():
    from pymoors.modeling.pow import Pow

    return Pow


def cast_to_constant(value):
    if isinstance(value, numbers.Number):
        # if isinstance(value, list):
        #     if not all(isinstance(v, (int, float)) for v in value):
        #         raise TypeError("If value is a list, all elements must be numbers")
        from pymoors.modeling.constant import Constant

        return Constant(value=value)
    return value


def cast_other_to_expression(meth):
    @functools.wraps(meth)
    def wrapper_cast_to_constant(self, other):
        return meth(self, cast_to_constant(other))

    return wrapper_cast_to_constant
