import abc
from typing import List, Dict, Union, Optional

from pydantic import BaseModel, ConfigDict

from pymoors.core.modeling.variable import Variable
from pymoors.core.constraints.constraints import Constraints, AffineConstraints
from pymoors.core.constraints.inequality import Inequality
from pymoors.typing import TwoDArray, PopulationCallable


class Problem(BaseModel, abc.ABC):
    objectives: PopulationCallable
    variables: Union[Variable, List[Variable]]
    constraints: Optional[Constraints]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        variables: Union[Variable, List[Variable]],
        constraints: Optional[
            Union[
                PopulationCallable,
                List[Inequality],
                List[Union[Inequality, PopulationCallable]],
            ]
        ] = None,
        **kwargs,
    ):
        if isinstance(constraints, PopulationCallable):
            constraints = Constraints(constraints_function=constraints)

        elif isinstance(constraints, list):
            # Check that if there is a callable in the list, this is unique.
            # Also sets the constraint function used in `evaluate`
            _custom_function = None
            _affine_constr = []
            for constr in self.constraints:
                if isinstance(constr, PopulationCallable):
                    if _custom_function is not None:
                        raise ValueError("Only one custom constraint callable must be given")
                    _custom_function = constr
                else:
                    _affine_constr.append(constr)
            # Create constraints object
            constraints = Constraints(
                constraints_function=_custom_function,
                affine_constraints=AffineConstraints(
                    variables=variables, affine_constraints=_affine_constr
                ),
            )
        super().__init__(variables=variables, constraints=constraints, **kwargs)

    def evaluate(self, population: TwoDArray) -> Dict[str, TwoDArray]:
        result = {"objectives": self.objectives(population)}
        if self.constraints:
            result["constraints"] = self.constraints.evaluate(population)
        return result
