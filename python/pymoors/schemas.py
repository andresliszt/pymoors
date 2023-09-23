from typing import Optional, Union, List, overload, Iterator

import numpy as np
from pydantic import BaseModel, ConfigDict, PrivateAttr

from pymoors.typing import OneDArray, TwoDArray


class Individual(BaseModel):
    genes: OneDArray
    fitness: OneDArray
    rank: int
    constraints: Optional[OneDArray]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def belongs_to_pareto_front(self) -> bool:
        return self.rank == 0

    @property
    def is_feasible(self) -> bool:
        if self.constraints is None:
            return True
        return np.all(np.array(self.constraints) <= 0)


class Population(BaseModel):
    genes: TwoDArray
    fitness: TwoDArray
    rank: OneDArray
    constraints: Optional[TwoDArray]

    _best: List[Individual] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @overload
    def __getitem__(self, index: int) -> Individual:
        ...

    @overload
    def __getitem__(self, index: slice) -> List[Individual]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Individual, List[Individual]]:
        if isinstance(index, int):
            return Individual(
                genes=self.genes[index],
                fitness=self.fitness[index],
                rank=self.rank[index],
                constraints=self.constraints[index] if self.constraints is not None else None,
            )
        if isinstance(index, slice):
            return [
                Individual(
                    genes=self.genes[i],
                    fitness=self.fitness[i],
                    rank=self.rank[i],
                    constraints=self.constraints[i] if self.constraints is not None else None,
                )
                for i in range(*index.indices(len(self.genes)))
            ]
        raise TypeError(f"indices must be integers or slices, not {type(index)}")

    def __iter__(self) -> Iterator[Individual]:
        for i in range(len(self.genes)):
            yield self[i]

    @property
    def best(self) -> List[Individual]:
        if self._best is None:
            self._best = [i for i in self if i.rank == 0]
        return self._best
