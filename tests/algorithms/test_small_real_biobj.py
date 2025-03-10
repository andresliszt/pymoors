import pytest
import numpy as np

from pymoors import (
    RandomSamplingFloat,
    GaussianMutation,
    CloseDuplicatesCleaner,
    SimulatedBinaryCrossover,
    AgeMoea,
    Nsga2,
    Nsga3,
    Rnsga2,
    DanAndDenisReferencePoints,
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
    population_genes: np.ndarray of shape (population_size, 2)
      Each row is (x,y).

    Returns
    -------
    fitness: np.ndarray of shape (population_size, 2)
      Each row is [f1, f2], the two objectives to minimize.
    """
    population_size = population_genes.shape[0]
    fitness = np.zeros((population_size, 2), dtype=float)
    for i in range(population_size):
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


@pytest.mark.parametrize(
    "algorithm_class, extra_kw",
    [
        (Nsga2, {"seed": 42}),
        (Nsga2, {"seed": None}),
        (AgeMoea, {"seed": 42}),
        (
            Nsga3,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=200, n_objectives=2
                )
            },
        ),
        (
            Nsga3,
            {
                "reference_points": DanAndDenisReferencePoints(
                    n_reference_points=200, n_objectives=2
                ).generate()
            },
        ),
        (
            Rnsga2,
            {"reference_points": np.array([[0.5, 0.5], [0.1, 0.1]]), "epsilon": 0.1},
        ),
        (
            Rnsga2,
            {"reference_points": np.array([[0.8, 0.8], [0.9, 0.9]]), "epsilon": 0.001},
        ),
    ],
)
def test_small_real_biobjective(algorithm_class, extra_kw):
    """
    Test a 2D real-valued problem:
      f1 = x^2 + y^2
      f2 = (x-1)^2 + (y-1)^2

    with x,y in [0,1].

    The real front is (x, y) in (0,1): x = y

    """

    algorithm = algorithm_class(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=15),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        n_vars=2,  # We have 2 variables: x,y
        population_size=100,
        n_offsprings=100,
        n_iterations=100,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-6),
        keep_infeasible=False,
        lower_bound=0,
        upper_bound=1,
        verbose=True,
        **extra_kw,
    )
    algorithm.run()

    final_population = algorithm.population
    best = final_population.best
    for i in best:
        assert i.genes[0] == pytest.approx(i.genes[1], abs=0.2)
    assert len(final_population) == 100
    # In this test all algorithms have to reach full pareto front
    assert len(np.unique(np.array([b.genes for b in best]), axis=0)) == 100


@pytest.mark.xfail(
    reason="Known issue https://github.com/andresliszt/pymoors/issues/48"
)
def test_same_seed_same_result():
    algorithm1 = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        constraints_fn=constraints_biobjective,
        n_vars=2,  # We have 2 variables: x,y
        population_size=50,
        n_offsprings=50,
        n_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-5),
        keep_infeasible=False,
        seed=1,
    )
    algorithm1.run()

    algorithm2 = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        constraints_fn=constraints_biobjective,
        n_vars=2,  # We have 2 variables: x,y
        population_size=50,
        n_offsprings=50,
        n_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-8),
        keep_infeasible=False,
        seed=100,
    )
    algorithm2.run()

    np.testing.assert_array_equal(
        algorithm1.population.genes, algorithm2.population.genes
    )
