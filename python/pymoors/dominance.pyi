from typing import List

from pymoors.typing import TwoDArray

def fast_non_dominated_sorting(population_fitness: TwoDArray) -> List[List[int]]:
    """Fast Dominated Soriting Algorithm

    Here `population_fitness` is a matriz of shape `(n,m) = (population number, objectives number)`
    represented as a two dimensional `numpy.ndarray`. It's known that this
    algorithm has complexity order `O(m*n**2)`.

    This function returns `k` ordered fronts from the fitness matrix, each front
    is a list of indexes, for example if we call `fronts: List[List[int]]` the output
    of this function, then if `i in fronts[j]` means that the individual `i` belongs to
    the front number `j`. Note that at least 1 front will be returned.

    Example:

    .. code-block:: python

       population_fitness = np.array([[0,1,2], [1,2,3], [0,0,3]], dtype = float)
       fronts = fast_non_dominated_sorting(population_fitness)
       print(fronts) # Should output [[0, 2], [1]]

    """
