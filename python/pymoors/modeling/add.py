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
        # this is safe because Add instances are created only if there are at least 2 args
        terms = [str(self.args[0])]
        for a in self.args[1:]:
            if a.coeff < 0:
                terms.append(f"- {str(a).replace('-', '', 1)}")
            else:
                terms.append(f"+ {str(a)}")
        return " ".join(terms)

    @staticmethod
    def _validate_args(args: tuple[Expression, ...]) -> None:
        length = None
        for a in args:
            if a.is_variable:
                if length is not None:
                    if length != a.length:
                        raise ValueError("Can't add variables with different length")
                length = a.length
        return None

    @staticmethod
    def _group_args(args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        """Group coeff args of equivalent expressions"""
        grouped_terms = defaultdict(lambda: 0)
        for a in args:
            if a.is_mul:
                new_a = Mul(*a.symbolic_factors)
                grouped_terms[new_a] += a.coeff
            else:
                grouped_terms[a] += 1
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
    def _simplfy_expressions(cls, *args: tuple[Expression, ...]) -> tuple[Expression, ...]:
        new_args = []
        for a in args:
            if a.is_math_operator:
                a = a.expand()
            if a.is_add:
                new_args.extend(cls._group_args(a.args))
            # TODO: Complete for is_mul, is_pow, etc
            else:
                new_args.append(a)
        return tuple(cls._group_args(new_args))
