from typing import List, Dict, Union, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator
import numpy as np

from pymoors.core.modeling.variable import Variable
from pymoors.core.constraints.constraints import Constraints
from pymoors.core.constraints.inequality import Inequality, Equality
from pymoors.typing import TwoDArray, PopulationCallable


class Problem(BaseModel):
    objectives: List[PopulationCallable]
    variables: Union[Variable, List[Variable]]
    constraints: Optional[List[Union[Inequality, Equality, PopulationCallable]]] = None
    _constraints_handler: Constraints = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator
    def _set_constraints_handler(self):
        if self.constraints is not None:
            self._constraints_handler = Constraints(
                variables=self.variables, constraints=self.constraints
            )
        return self

    @property
    def constraints_handler(self) -> Constraints:
        return self._constraints_handler

    @property
    def number_variables(self) -> int:
        return (
            self.variables.size
            if isinstance(self.variables, Variable)
            else sum(var.size for var in self.variables)
        )

    @property
    def number_objectives(self) -> int:
        return len(self.objectives)

    def evaluate(self, population: TwoDArray) -> Dict[str, TwoDArray]:
        # Keep this dict definition out of pydantic.
        result = {"objectives": np.column_stack([obj(population) for obj in self.objectives])}
        if self.constraints:
            result["constraints"] = self.constraints_handler.function(population)
        return result
