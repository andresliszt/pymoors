import abc
from typing import final

import numpy as np
from pydantic import BaseModel, Field

from pymoors.typing import OneDArray, TwoDArray


class Mutation(BaseModel, abc.ABC):
    mutation_probability: float = Field(ge=0, le=1)

    def _selector(self, population: TwoDArray) -> TwoDArray:
        """Select individuals for which the mutation will be applied"""
        return population[
            np.random.random(population.shape[0]) < self.mutation_probability
        ]

    @abc.abstractmethod
    def mutation(self, individual: OneDArray) -> OneDArray:
        """Mutates a single individual

        `individual` is a 1D `numpy.ndarray` and the output `individual`
        must have the same length than the original.

        """

    @final
    def operate(self, population: TwoDArray) -> TwoDArray:
        """Main method to apply the mutation operator in the population

        `population` is a 2D `numpy.ndarray` with shape
        `(number_matings, number_variables)`. This method maintains the shape
        of the input `population`

        """
        population = self._selector(population)
        if population.empty:
            # NOTE: empty case will maintain the dimension (will have shape = (0, number_variables))
            return population
        return np.apply_along_axis(self.mutation, 1, population)
