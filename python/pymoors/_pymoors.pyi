from typing import Optional, TypedDict, Unpack

from pymoors.schemas import Population
from pymoors.typing import (
    FitnessPopulationCallable,
    ConstraintsPopulationCallable,
    TwoDArray,
)

# pylint: disable=W0622, W0231

class SamplingOperator:
    """
    Base class for sampling operators used to initialize or generate new individuals in the population.

    This abstract class defines the interface for different sampling strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the SamplingOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

class MutationOperator:
    """
    Base class for mutation operators used to introduce variations in individuals.

    This abstract class defines the interface for different mutation strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the MutationOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

class CrossoverOperator:
    """
    Base class for crossover operators used to combine two parent individuals to produce offspring.

    This abstract class defines the interface for different crossover strategies.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the CrossoverOperator.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

class RandomSamplingFloat(SamplingOperator):
    """
    Sampling operator for floating-point variables using uniform random distribution.

    Generates random float values within a specified range.

    Args:
        min (float): The minimum value for sampling.
        max (float): The maximum value for sampling.
    """

    def __init__(self, min: float, max: float) -> None: ...

class RandomSamplingInt(SamplingOperator):
    """
    Sampling operator for integer variables using uniform random distribution.

    Generates random integer values within a specified range.

    Args:
        min (int): The minimum integer value for sampling.
        max (int): The maximum integer value for sampling.
    """

    def __init__(self, min: int, max: int) -> None: ...

class RandomSamplingBinary(SamplingOperator):
    """
    Sampling operator for binary variables.

    Generates random binary values (0 or 1).
    """

    def __init__(self) -> None: ...

class PermutationSampling(SamplingOperator):
    """
    Sampling operator for permutation-based variables.

    Generates random permutations of a given set of elements.
    """

    def __init__(self) -> None: ...

class BitFlipMutation(MutationOperator):
    """
    Mutation operator that flips bits in a binary individual with a specified mutation rate.

    Each bit has a probability equal to `gene_mutation_rate` to be flipped.

    Args:
        gene_mutation_rate (float): The probability of flipping each bit.
    """

    def __init__(self, gene_mutation_rate: float) -> None: ...

class SwapMutation(MutationOperator):
    """
    Mutation operator that swaps two genes in a permutation-based individual.

    This operator is useful for permutation-based representations to maintain valid permutations.
    """

    def __init__(self) -> None: ...

class GaussianMutation(MutationOperator):
    """
    Mutation operator that adds Gaussian noise to float variables.

    Each gene is perturbed by a Gaussian-distributed random value.

    Args:
        gene_mutation_rate (float): The probability of mutating each gene.
        sigma (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, gene_mutation_rate: float, sigma: float) -> None: ...

class OrderCrossover(CrossoverOperator):
    """
    Crossover operator for permutation-based individuals using Order Crossover (OX).

    Preserves the relative order of genes from the parents in the offspring.
    """

    def __init__(self) -> None: ...

class SinglePointBinaryCrossover(CrossoverOperator):
    """
    Single-point crossover operator for binary-encoded individuals.

    A single crossover point is selected, and the binary strings are exchanged beyond that point.
    """

    def __init__(self) -> None: ...

class UniformBinaryCrossover(CrossoverOperator):
    def __init__(self) -> None: ...

class ExponentialCrossover(CrossoverOperator):
    """
    Crossover operator that combines parent genes based on an exponential distribution.

    The `exponential_crossover_rate` controls the influence of each parent.

    Args:
        exponential_crossover_rate (float): The rate parameter for the exponential distribution.
    """

    def __init__(self, exponential_crossover_rate: float) -> None: ...

class SimulatedBinaryCrossover(CrossoverOperator):
    """
    Simulated Binary Crossover (SBX) operator for real-coded genetic algorithms.

    SBX is a widely used crossover mechanism for continuous variables, inspired
    by the behavior of single-point crossover in binary-coded GAs. Instead of
    slicing bit strings, SBX generates offspring by interpolating (and potentially
    extrapolating) between two parent solutions (p1 and p2).

    The key parameter `distribution_index` (often called "eta") controls how far
    the offspring can deviate from the parents. A higher distribution index results
    in offspring closer to the parents (exploitation), whereas a lower value
    produces offspring that can be further away (exploration).

    In each crossover event, SBX computes a factor `beta_q`, based on a random
    number in [0,1) and the distribution index `eta`. This factor dictates where
    each child solution lies relative to the parent solutions. If the parent genes
    (p1 and p2) differ minimally, no crossover is performed (i.e., the children
    inherit the parents' values directly).

    Reference:
        - Deb, Kalyanmoy, and R. B. Agrawal. "Simulated binary crossover for
          continuous search space." Complex Systems 9.2 (1995): 115-148.

    """
    def __init__(self, distribution_index: float): ...

class DuplicatesCleaner:
    """
    Base class for cleaning duplicate individuals in the population.

    This abstract class defines the interface for different duplicate cleaning strategies.
    """

class ExactDuplicatesCleaner(DuplicatesCleaner):
    """
    Cleaner that removes exact duplicate individuals from the population.

    Ensures all individuals in the population are unique.
    """

    def __init__(self) -> None: ...

class CloseDuplicatesCleaner(DuplicatesCleaner):
    """
    Cleaner that removes individuals that are close to each other based on a specified epsilon.

    Two individuals are considered duplicates if their distance is less than `epsilon`.

    Args:
        epsilon (float): The distance threshold to consider individuals as duplicates.
    """

    def __init__(self, epsilon: float) -> None: ...

class _MooAlgorithmKwargs(TypedDict, total=False):
    """
    It exists for Multi-Objective Optimization (MOO) algorithms kwargs.

    Provides common functionalities and interfaces for MOO algorithms like NSGA-II and NSGA-III.

    Args:
        sampler (SamplingOperator): Operator to sample initial population.
        crossover (CrossoverOperator): Operator to perform crossover.
        mutation (MutationOperator): Operator to perform mutation.
        fitness_fn (FitnessPopulationCallable): Function to evaluate the fitness of the population.
        n_vars (int): Number of variables in the optimization problem.
        pop_size (int): Population size.
        n_offsprings (int): Number of offsprings generated in each generation.
        num_iterations (int): Number of generations to run the algorithm.
        mutation_rate (float): Probability of mutation.
        crossover_rate (float): Probability of crossover.
        keep_infeasible (bool, optional): Whether to keep infeasible solutions. Defaults to False.
        duplicates_cleaner (Optional[DuplicatesCleaner], optional): Cleaner to remove duplicates. Defaults to None.
        constraints_fn (Optional[ConstraintsPopulationCallable], optional): Function to handle constraints. Defaults to None.
    """

    sampler: SamplingOperator
    crossover: CrossoverOperator
    mutation: MutationOperator
    fitness_fn: FitnessPopulationCallable
    n_vars: int
    pop_size: int
    n_offsprings: int
    num_iterations: int
    mutation_rate: float
    crossover_rate: float
    keep_infeasible: bool
    duplicates_cleaner: Optional[DuplicatesCleaner]
    constraints_fn: Optional[ConstraintsPopulationCallable]

class Nsga2:
    """
    Implementation of the NSGA-II (Non-dominated Sorting Genetic Algorithm II).

    NSGA-II is a popular multi-objective evolutionary algorithm known for its fast non-dominated sorting approach
    and crowding distance mechanism to maintain diversity in the population.

    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm:
        NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.
    """

    def __init__(self, **kwargs: Unpack[_MooAlgorithmKwargs]): ...
    @property
    def population(self) -> Population:
        """
        Get the current population.

        Returns:
            Population: The current population of individuals.
        """

    def run(self) -> None: ...

class _Nsga3Kwargs(_MooAlgorithmKwargs, total=False):
    reference_points: TwoDArray

class Nsga3:
    """
    Implementation of the NSGA-III (Non-dominated Sorting Genetic Algorithm III).

    NSGA-III extends NSGA-II to handle many-objective optimization problems by introducing reference points
    for better diversity maintenance in higher dimensions.

    Reference:
        Deb, K., Jain, H., & Thiele, L. (2014). NSGA-III: A Many-Objective Genetic Algorithm Using Reference-Points
        for Selection. IEEE Transactions on Evolutionary Computation, 18(4), 577-601.
    """

    def __init__(self, **kwargs: Unpack[_Nsga3Kwargs]) -> None: ...
    @property
    def population(self) -> Population:
        """
        Get the current population.

        Returns:
            Population: The current population of individuals.
        """

    def run(self) -> None: ...
