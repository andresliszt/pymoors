from typing import List

from attrs import define

from pymoors._typing import NDArray2x2
from pymoors.population.individual import Individual


@define
class Population:
    individuals: List[Individual]

    def population_fitness(self) -> NDArray2x2:
        pass
