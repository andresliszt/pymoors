from typing import List

import pytest
import numpy as np

from pymoors.moo_algorithms import fast_non_dominated_sorting


@pytest.mark.parametrize(
    "population_fitness, expected_fronts",
    [
        (np.array([[1, 0], [0, 1]], dtype=float), [[0, 1]]),
        (np.array([[0, 0, 0], [1, 1, 1]], dtype=float), [[0], [1]]),
        # (np.array([[0, 0, 0], [1, 1, 0]], dtype=float), [[0], [1]]),
    ],
)
def test_fast_non_dominated_sorting(population_fitness: np.ndarray, expected_fronts):
    assert fast_non_dominated_sorting(population_fitness) == expected_fronts
