from __future__ import annotations
import abc
from typing import Any, ClassVar, TypeVar

from pymoors.helpers import cast_other_to_expression


# pylint: disable=C0415

ExpressionLike = TypeVar("ExpressionLike")


class Expression(abc.ABC):
    is_constant: ClassVar[bool] = False
    is_variable: ClassVar[bool] = False
    is_math_operator: ClassVar[bool] = False
    is_add: ClassVar[bool] = False
    is_mul: ClassVar[bool] = False
    is_pow: ClassVar[bool] = False

    # Instance attrs
    _args: tuple[Expression, ...]
    _coeff: float

    __slots__ = ("_args", "_coeff")

    def __new__(cls, *args):
        cls._validate_args(args)
        obj = object.__new__(cls)
        obj._args = args
        return obj

    def _hashable_content(self) -> tuple:
        return self.args

    @staticmethod
    def _validate_args(args: tuple[Expression, ...]) -> None:
        return None

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def coeff(self) -> int:
        return 1

    @cast_other_to_expression
    def __eq__(self, other: ExpressionLike) -> bool:
        if self is other:
            return True
        if not isinstance(other, Expression):
            return NotImplemented
        return self._hashable_content() == other._hashable_content()

    @cast_other_to_expression
    def check_eq(self, other: ExpressionLike) -> bool:
        if self is other:
            return True
        if not isinstance(other, Expression):
            return False
        return self._hashable_content() == other._hashable_content()


    def __hash__(self) -> int:
        return hash(self._hashable_content())

    @property
    def args(self) -> tuple[Expression, ...]:
        return self._args

    @cast_other_to_expression
    def __add__(self, other: ExpressionLike) -> Expression:
        from pymoors.helpers import add_expression

        return add_expression()(self, other)

    @cast_other_to_expression
    def __radd__(self, other: ExpressionLike) -> Expression:
        """Default add is conmutative"""
        return self.__add__(other)

    @cast_other_to_expression
    def __sub__(self, other: ExpressionLike) -> Expression:
        return self.__add__(-1 * other)

    @cast_other_to_expression
    def __rub__(self, other: ExpressionLike) -> Expression:
        return (-1) * self.__sub__(other)

    @cast_other_to_expression
    def __mul__(self, other: ExpressionLike) -> Expression:
        from pymoors.helpers import mul_expression

        return mul_expression()(self, other)

    @cast_other_to_expression
    def __rmul__(self, other: ExpressionLike) -> Expression:
        """Default mul is conmutative"""
        return self.__mul__(other)

    @cast_other_to_expression
    def __pow__(self, other: ExpressionLike) -> Expression:
        from pymoors.helpers import pow_expression

        return pow_expression()(self, other)

    @cast_other_to_expression
    def __truediv__(self, other):
        from pymoors.modeling.constant import Constant

        if other == Constant(value=0):
            raise ZeroDivisionError("division by zero")
        return self.__mul__(other.__pow__(-1))



class MathOperator(Expression):
    identity: ClassVar[Any] = None
    anihilator: ClassVar[Any] = None
    is_math_operator: ClassVar[bool] = True

    @classmethod
    def _simplfy_expressions(cls, *args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        return args

    def __new__(cls, *args):
        args = [a for a in args if not a.check_eq(cls.identity)]
        args = cls._simplfy_expressions(*args)

        if not args:
            return cls.identity
        if len(args) == 1:
            return args[0]

        if cls.anihilator is not None:
            if any(a == cls.anihilator for a in args):
                return cls.identity

        obj = Expression.__new__(cls, *args)

        if not obj.args:
            return cls.identity

        return obj

    def expand(self) -> Expression:
        return self
