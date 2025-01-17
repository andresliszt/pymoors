from typing import Callable, Optional

class SamplingOperator:
    def __init__(self, *args, **kwargs): ...

class MutationOperator:
    def __init__(self, *args, **kwargs): ...

class CrossoverOperator:
    def __init__(self, *args, **kwargs): ...


class Nsga2:
    def __init__(
        self,
        sampler: SamplingOperator,
        mutation: MutationOperator,
        crossover: CrossoverOperator,
        fitness_function: Callable,
        pop_size: int,
        n_offsprings: int,
        num_iterations: int,
        mutation_rate: float,
        crossover_rate: float,
        constraints_fn: Optional[Callable],
    ): ...
