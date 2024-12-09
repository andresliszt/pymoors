from __future__ import annotations
from typing import ClassVar
import numbers
from collections import defaultdict

from pymoors.expression import MathOperator, Expression

from pymoors.modeling.constant import Constant
from pymoors.modeling.pow import Pow


class Mul(MathOperator):
    is_mul: ClassVar[bool] = True
    identity: ClassVar[Constant] = Constant(value=1)
    anihilator: ClassVar[Constant] = Constant(value=0)

    _coeff: Constant = None

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

    def __repr__(self) -> str:
        return "*".join(str(a) for a in self.args)

    @property
    def coeff(self) -> Constant:
        return self._coeff

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
            a = Pow(base=a, exp=exp)
            new_args.append(a)
        if constant != 1:
            return tuple([constant, *new_args])
        return tuple(new_args)

    @classmethod
    def _simplfy_expressions(cls, args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        new_args = []
        for a in args:
            if a.is_mul:
                new_args.extend(cls._group_args(a.args))
            # TODO: Complete for is_add, is_pow, etc
            else:
                new_args.append(a)
        return tuple(cls._group_args(new_args))
