# from typing import List

# from pydantic import PrivateAttr

# from pymoors.core.modeling.expression import Expression
# from pymoors.core.modeling.constant import Constant


# class MatMultiplyExpression(Expression):
#     """Currently it only supports multiply by constant"""

#     _expressions: List[Expression] = PrivateAttr(default=None)
#     _size: int = PrivateAttr(default=None)

#     def __init__(self, matrix: Constant, var2: Expression, **kwargs):
#         if not any((isinstance(mat, Constant), isinstance(var2, Constant))):
#             raise TypeError("Matmul is only supported ")
#         # Get constant and expression
#         constant = var1 if isinstance(var1, Constant) else var2
#         expression = var1 if isinstance(var2, Constant) else var2
#         # Only scalar multiplication or inner dot product is supported
#         if constant.shape[0] > 1 and constant.shape[0] != expression.shape[0]:
#             raise ValueError(
#                 "An expression can only be multiplied by a scalar constant or by an array constant. "
#                 "The last implies that the inner dot product will be performed, therefore the array size "
#                 f"must be equal to expression size. Got constant size {constant.size} and expression size {expression.size}"
#             )
#         # Now simplify constants
#         if isinstance(expression, MultiplyExpression):
#             constant = expression.constant * constant

#         super().__init__(**kwargs)
#         self._expressions = [constant, *expression.non_constant_expressions]
#         # We set the size of this expression.
#         self._size = expression.size

#     @property
#     def size(self) -> int:
#         """All expressions have the same size"""
#         return self._size

#     @property
#     def name(self) -> str:
#         return f"{self.constant.value}*({self.non_constant_expressions[0].name})"

#     @property
#     def expressions(self) -> List[Expression]:
#         return self._expressions

#     @property
#     def constant(self) -> Constant:
#         """Constant is always placed in the first position of expressions"""
#         return self.expressions[0]
