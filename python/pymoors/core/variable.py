from pydantic import Field

from pymoors.core.expression import Expression


class Variable(Expression):
    number_variables: Field(ge=1)

    @property
    def size(self) -> int:
        return self.number_variables

    @property
    def name(self) -> str:
        return f"Var(size={self.size})"
