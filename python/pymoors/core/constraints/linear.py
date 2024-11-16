from typing import List, Union

import numpy as np
from pydantic import BaseModel, PrivateAttr

from pymoors.typing import OneDArray, TwoDArray
from pymoors.core.constraints.inequality import Inequality
from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.decoder import LinearDecoder
from pymoors.typing import PopulationCallable


class AffineConstraints(BaseModel):
    variables: Union[Variable, List[Variable]]
    affine_constraints: List[Inequality]
    _G: TwoDArray = PrivateAttr(default=None)
    _B: OneDArray = PrivateAttr(default=None)
    _decoder: LinearDecoder = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set privates
        self._G, self._B = self.decoder.decode(inequalities=self.affine_constraints)

    @property
    def decoder(self) -> LinearDecoder:
        if self._decoder is None:
            self._decoder = LinearDecoder(variables=self.variables)
        return self._decoder

    @property
    def G(self) -> TwoDArray:
        return self._G

    @property
    def B(self) -> OneDArray:
        return self._B

    @property
    def function(self) -> PopulationCallable:
        G, B = self.G, self.B

        def _function(population: TwoDArray) -> TwoDArray:
            return np.dot(population, G.T) - B

        return _function
