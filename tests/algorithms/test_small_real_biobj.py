import pytest
import numpy as np

from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    CloseDuplicatesCleaner,
    SimulatedBinaryCrossover,
)
from pymoors.typing import TwoDArray

# FIXME: Once RNsga2 is exposed to the users, change the import
from pymoors._pymoors import RNsga2


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


@pytest.mark.parametrize("seed", [42, None], ids=["seed", "no_seed"])
def test_small_real_biobjective_nsag2(seed: int | None):
    """
    Test a 2D real-valued problem:
      f1 = x^2 + y^2
      f2 = (x-1)^2 + (y-1)^2
    with x,y in [0,1].

    The real front is (x, y) in (0,1): x = y

    """
    print(f"{seed=}", flush=True)
    algorithm = Nsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        constraints_fn=constraints_biobjective,
        n_vars=2,  # We have 2 variables: x,y
        pop_size=200,
        n_offsprings=200,
        num_iterations=50,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-5),
        keep_infeasible=False,
        seed=seed,
    )
    algorithm.run()

    final_population = algorithm.population
    best = final_population.best
    for i in best:  # FIXME: Fix the abs in the tests --- Should be 0.05
        assert i.genes[0] == pytest.approx(i.genes[1], abs=0.2)

    assert len(final_population) == 200


def test_small_real_biobjective_rnsga2():
    """
    Test a 2D real-valued problem with RNsga2 using reference points and an epsilon value.
    The objectives are:
        f1 = x^2 + y^2
        f2 = (x-1)^2 + (y-1)^2
    with x, y in [0, 1].

    The true Pareto front is approximately: { (x,y) in [0,1]^2 : x ≈ y }.
    We provide reference points uniformly along the line x=y in [0,1] and set epsilon=0.01.

    """
    # Define reference points uniformly along the line x=y
    r = np.linspace(0, 0.2, 2)
    reference_points = np.column_stack((r, r))  # shape (11, 2)

    algorithm = RNsga2(
        sampler=RandomSamplingFloat(min=0.0, max=1.0),
        crossover=SimulatedBinaryCrossover(distribution_index=2),
        mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.05),
        fitness_fn=fitness_biobjective,
        constraints_fn=constraints_biobjective,
        n_vars=2,  # Two variables: x and y
        pop_size=200,
        n_offsprings=200,
        num_iterations=50,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-8),
        keep_infeasible=False,
        reference_points=reference_points,
        epsilon=1,
    )
    algorithm.run()

    final_population = algorithm.population
    best = final_population.best
    for individual in best:
        # Since the Pareto front is approximately x = y, test that gene[0] is near gene[1]
        assert individual.genes[0] == pytest.approx(individual.genes[1], abs=0.2)

    # Also, the final population size should be exactly 200.
    assert len(final_population) == 200


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
        pop_size=50,
        n_offsprings=50,
        num_iterations=20,
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
        pop_size=50,
        n_offsprings=50,
        num_iterations=20,
        mutation_rate=0.1,
        crossover_rate=0.9,
        duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-5),
        keep_infeasible=False,
        seed=100,
    )
    algorithm2.run()

    np.testing.assert_array_equal(
        algorithm1.population.genes, algorithm2.population.genes
    )
