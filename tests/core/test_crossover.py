from typing import ClassVar

import pytest
import numpy as np

from pymoors._typing import OneDArray, TwoDArray
from pymoors.core.crossover import Crossover
from pymoors.errors import CrossoverOutputError


def test_crossover():
    class DummyCrossover(Crossover):
        number_offsprings: ClassVar[int] = 1
        number_parents: ClassVar[int] = 2

        def crossover(self, parents: TwoDArray) -> OneDArray:
            return parents.mean(axis=0)

    # 2 parents, 10 matings and 5 variables
    population = np.ones((2, 10, 5))
    # In this tests all individuals are being used in the crossover due the probability = 1
    offsprings = DummyCrossover(crossover_probability=1).operate(population=population)
    np.testing.assert_array_equal(offsprings, np.ones((1, 10, 5)), strict=False)


def test_invalid_crossover():
    class DummyCrossover(Crossover):
        number_offsprings: ClassVar[int] = 2
        number_parents: ClassVar[int] = 2

        def crossover(self, parents: TwoDArray) -> OneDArray:
            return parents.mean(axis=0)

    # 2 parents, 10 matings and 5 variables
    population = np.ones((2, 10, 5))
    # This test should fail because mating expects two offsprings per crossover
    with pytest.raises(
        CrossoverOutputError, match="Expected number of offsprings is `2` for the Crossover operator. Got `1`"
    ):
        DummyCrossover(crossover_probability=1).operate(population=population)
