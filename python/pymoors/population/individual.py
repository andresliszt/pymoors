from attrs import define
import numpy as np


from pymoors.typing import NDArray2x2


@define
class Genes:
    fitness: NDArray2x2
    constraints: NDArray2x2
