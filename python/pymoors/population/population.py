from typing import List

from attrs import define

from pymoors.typing import NDArray2x2
from pymoors.population.individual import Genes


@define
class PopulationGenes:
    individuals: List[Genes]

    def population_fitness(self) -> NDArray2x2:
        pass
