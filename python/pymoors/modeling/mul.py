from __future__ import annotations
from typing import ClassVar
import numbers
from collections import defaultdict

from pymoors.expression import MathOperator, Expression, ExpressionLike

from pymoors.modeling.constant import Constant
from pymoors.modeling.pow import Pow
from pymoors.helpers import cast_other_to_expression

# pylint: disable=C0415


class Mul(MathOperator):
    is_mul: ClassVar[bool] = True
    identity: ClassVar[Constant] = Constant(value=1)
    anihilator: ClassVar[Constant] = Constant(value=0)

    _coeff: Constant

    def __new__(cls, *args):
        coeff = 1
        new_args = []
        for a in args:
            if isinstance(a, (numbers.Number, Constant)):
                coeff *= a
            else:
                new_args.append(a)
        obj = MathOperator.__new__(cls, Constant(value=coeff), *new_args)
        obj._coeff = coeff
        return obj

    def __hash__(self) -> int:
        return hash(self._hashable_content())

    @cast_other_to_expression
    def __eq__(self, other: ExpressionLike):
        expanded_self = self.expand()
        if other.is_mul:
            other = other.expand()
        if expanded_self.is_mul:
            return super().__eq__(other)
        return type(expanded_self).__eq__(expanded_self, other)

    def __repr__(self) -> str:
        return "*".join(f"({str(a)})" if a.is_add else str(a) for a in self.args)

    @property
    def coeff(self) -> Constant:
        return self._coeff

    @property
    def symbolic_factors(self) -> tuple[Expression, ...]:
        return self.args[1:]

    @staticmethod
    def _group_args(args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        grouped_terms = defaultdict(lambda: 0)

        for a in args:
            # We collect exponents
            if a.is_pow:
                grouped_terms[a.base] += a.exp
            else:
                grouped_terms[a] += 1

        # Set up new coefficients/exponents
        new_args = []
        constant = Constant(value=1)
        for a, exp in grouped_terms.items():
            if exp == 0:
                continue
            if a.is_constant:
                constant *= a
                continue
            if exp != 1:
                a = Pow(base=a, exp=exp)
            new_args.append(a)
        if constant != 1:
            return tuple([constant, *new_args])
        return tuple(new_args)

    @classmethod
    def _simplfy_expressions(cls, *args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        new_args = []
        for a in args:
            if a.is_mul:
                new_args.extend(cls._group_args(a.args))
            # TODO: Complete for is_add, is_pow, etc
            else:
                new_args.append(a)
        return tuple(cls._group_args(new_args))

    def expand(self) -> Expression:
        from pymoors.helpers import add_expression

        Add = add_expression()

        def expand_two_expr(expr1: Expression, expr2: Expression) -> Expression:
            if expr1.is_add and not expr2.is_add:
                return Add(*(expand_two_expr(a, expr2) for a in expr1.args))
            if expr2.is_add and not expr1.is_add:
                return Add(*(expand_two_expr(expr1, a) for a in expr2.args))
            if expr1.is_add and expr2.is_add:
                # Distribute over both additions recursively
                return Add(*(expand_two_expr(a, b) for a in expr1.args for b in expr2.args))
            # If neither is an Add, return their product
            return expr1 * expr2

        # Recursively expand over self.args
        result = self.args[0]
        for arg in self.args[1:]:
            result = expand_two_expr(result, arg)
        return result
