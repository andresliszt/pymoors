import pytest
import numpy as np

from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    ExponentialCrossover,
    CloseDuplicatesCleaner,
)
from pymoors.typing import TwoDArray


def f1(x: float, y: float) -> float:
    """Objective 1: x^2 + y^2."""
    return x**2 + y**2


def f2(x: float, y: float) -> float:
    """Objective 2: (x - 1)^2 + (y - 1)^2."""
    return (x - 1) ** 2 + (y - 1) ** 2


def fitness_biobjective(population_genes: TwoDArray) -> TwoDArray:
    """
    Multi-objective fitness for a population of real-valued vectors [x, y].

    Parameters
    ----------
    population_genes: np.ndarray of shape (pop_size, 2)
      Each row is (x,y).

    Returns
    -------
    fitness: np.ndarray of shape (pop_size, 2)
      Each row is [f1, f2], the two objectives to minimize.
    """
    pop_size = population_genes.shape[0]
    fitness = np.zeros((pop_size, 2), dtype=float)
    for i in range(pop_size):
        x, y = population_genes[i]  # shape (2,)
        fitness[i, 0] = f1(x, y)
        fitness[i, 1] = f2(x, y)
    return fitness


def constraints_biobjective(population_genes: TwoDArray) -> TwoDArray:
    x = population_genes[:, 0]
    y = population_genes[:, 1]
    constraints = np.column_stack((-x, x - 1, -y, y - 1))
    return constraints


##############################################################################
# 2. TEST
##############################################################################


def test_small_real_biobjective_nsag2():
    """
    Test a 2D real-valued problem:
      f1 = x^2 + y^2
      f2 = (x-1)^2 + (y-1)^2
    with x,y in [0,1].

    The real front is (x, y) in (0,1): x = y

    """

    algorithm = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=ExponentialCrossover(exponential_crossover_rate=0.5),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        constraints_fn=constraints_biobjective,
        n_vars=2,  # We have 2 variables: x,y
        pop_size=200,
        n_offsprings=200,
        num_iterations=200,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-16),
        keep_infeasible=False,
    )
    algorithm.run()

    final_population = algorithm.population

    best = final_population.best

    for i in best:  # FIXME: Fix the abs in the tests
        assert i.genes[0] == pytest.approx(i.genes[1], abs=0.5)

    assert len(final_population) == 200
