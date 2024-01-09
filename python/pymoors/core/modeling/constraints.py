from pydantic import BaseModel

from pymoors.typing import TwoDArray
from pymoors.core.modeling.expression import Index


class Constraint(BaseModel):
    def affine_constraints(self, population: TwoDArray, index: Index) -> TwoDArray:
        ...
