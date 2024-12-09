from __future__ import annotations

from typing import ClassVar
from collections import defaultdict

from pymoors.expression import Expression, MathOperator
from pymoors.modeling.mul import Mul
from pymoors.modeling.constant import Constant


class Add(MathOperator):
    is_add: ClassVar[bool] = True
    identity: ClassVar[Constant] = Constant(value=0)

    def __repr__(self) -> str:
        return "+".join(str(a) for a in self.args)

    @staticmethod
    def _group_args(args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        """Group coeff args of equivalent expressions"""
        grouped_terms = defaultdict(lambda: 0)
        for a in args:
            grouped_terms[a] += a.coeff
        # Set up new coefficients
        new_args = []
        for a, coeff in grouped_terms.items():
            if coeff == 0:
                continue
            if coeff != 1:
                a = Mul(coeff, a)
            new_args.append(a)
        return tuple(new_args)

    @classmethod
    def _simplfy_expressions(cls, args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        new_args = []
        for a in args:
            if a.is_add:
                new_args.extend(cls._group_args(a.args))
            # TODO: Complete for is_mul, is_pow, etc
            else:
                new_args.append(a)
        return tuple(cls._group_args(new_args))
