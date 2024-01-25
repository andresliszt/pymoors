from __future__ import annotations

from pydantic import Field

from pymoors.core.modeling.expression import Expression


class Variable(Expression):
    length: int = Field(default=1, ge=1)

    @property
    def size(self) -> int:
        return self.length

    @property
    def name(self) -> str:
        return f"var_{self.expression_id}(size={self.size})"
