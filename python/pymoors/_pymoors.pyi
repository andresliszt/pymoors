from typing import Callable, Optional

class SamplingOperator:
    def __init__(self, *args, **kwargs): ...

class MutationOperator:
    def __init__(self, *args, **kwargs): ...

class CrossoverOperator:
    def __init__(self, *args, **kwargs): ...

class RandomSamplingFloat(SamplingOperator):
    def __init__(self, min: float, max: float): ...

class RandomSamplingInt(SamplingOperator):
    def __init__(self, min: int, max: int): ...

class RandomSamplingBinary(SamplingOperator):
    def __init__(self): ...

class BitFlipMutation(MutationOperator):
    def __init__(self, gene_mutation_rate: float): ...

class SinglePointBinaryCrossover(CrossoverOperator):
    def __init__(self): ...

class Nsga2:
    def __init__(
        self,
        sampler: SamplingOperator,
        mutation: MutationOperator,
        crossover: CrossoverOperator,
        fitness_fn: Callable,
        n_vars: int,
        pop_size: int,
        n_offsprings: int,
        num_iterations: int,
        mutation_rate: float,
        crossover_rate: float,
        constraints_fn: Optional[Callable],
    ): ...
