from typing import Literal, Tuple

from pydantic import Field

from pymoors.core.modeling.expression import Expression


class Variable(Expression):
    length: int = Field(default=1, ge=1)
    # dtype: Literal["float", "int", "binnary"]

    def _shape_from_expressions(self) -> Tuple[int]:
        """Currently 1D variables are supported"""
        return (self.length,)

    @property
    def name(self) -> str:
        return f"var_{self.expression_id}(shape={self.shape})"
