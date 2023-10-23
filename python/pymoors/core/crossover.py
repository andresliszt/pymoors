import abc
from typing import final, Union, ClassVar

import numpy as np
from pydantic import BaseModel, Field

from pymoors._typing import OneDArray, TwoDArray, ThreeDArray
from pymoors.errors import CrossoverDimensionError


class Crossover(BaseModel, abc.ABC):
    number_parents: ClassVar[int]
    number_offsprings: ClassVar[int]
    crossover_probability: float = Field(ge=0, le=1)

    def _selector(self, population: ThreeDArray) -> ThreeDArray:
        """Select individual for which the crossover will be applied"""
        return population[:, np.random.random(population.shape[1]) < self.crossover_probability]

    @abc.abstractmethod
    def crossover(self, parents: TwoDArray) -> Union[OneDArray, TwoDArray]:
        """Defines single crossover operation

        `parents` is a 2D dimensional `numpy.ndarray` with number of rows equal
        to `number_parents`. This method must return either a one dimensional `numpy.ndarray`
        if `number_offsprings = 1` or a two dimensional `numpy.ndarray`
        if `number_offsprings` is greater than 1. In the last case each
        row is an offspring.

        """

    @final
    def operate(self, population: ThreeDArray) -> ThreeDArray:
        """Main method to apply the crossover

        `population` is a 3D `numpy.ndarray` with shape
        `(number_parents, number_matings, number_variables)`. This method
        will return offsprings with shape `(number_offsprings, number_matings, number_variables)`

        """
        # First get individuals based on crossover probability
        population = self._selector(population)
        # Get dimensions
        _, number_matings, number_variables = population.shape
        # Create a container where offsprings will be saved
        offsprings = np.empty(shape=(self.number_offsprings, number_matings, number_variables))
        # Iterate over mating dimension
        for mating_index in range(number_matings):
            # Set the offsprings for the current mating index
            off = self.crossover(population[:, mating_index, :])
            # Validate that the operator returns expected shape
            if off.shape[0] != self.number_offsprings:
                raise CrossoverDimensionError(expected_dimension=self.number_offsprings, recived_dimension=off.shape[0])
            offsprings[:, mating_index, :] = off

        return offsprings
