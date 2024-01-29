from pydantic import BaseModel, model_validator, field_validator

from pymoors.core.modeling.expression import Expression
from pymoors.core.modeling.add import AddExpression
from pymoors.core import helpers


class Inequality(BaseModel):
    lhs: Expression
    rhs: Expression

    @field_validator("lhs", "rhs", mode="before")
    @classmethod
    def cast_to_constant(cls, value):
        return helpers.cast_to_constant(value)

    @model_validator(mode="after")
    def at_most_one_constant(self):
        if isinstance(self.lhs, helpers.constant()) and isinstance(
            self.rhs, helpers.constant()
        ):
            raise ValueError("Can't be both constant 'lhs' and 'rhs'")

        return self

    @property
    def expression(self) -> AddExpression:
        return self.lhs - self.rhs
