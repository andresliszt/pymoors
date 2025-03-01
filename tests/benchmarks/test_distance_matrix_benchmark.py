import pytest
import numpy as np
from scipy.spatial.distance import cdist

from pymoors._pymoors import cross_euclidean_distances  # type: ignore


@pytest.mark.parametrize("n", [100, 500, 1000, 2000])
def test_compare_scipy_cdist_vs_pymoors(benchmark, n):
    matrix = np.random.rand(n, n)

    result_pymoors = benchmark(cross_euclidean_distances(matrix, matrix))
    result_scipy = benchmark(cdist(matrix, matrix, metric="sqeuclidean"))

    mean_f1 = result_pymoors.stats.mean
    mean_f2 = result_scipy.stats.mean

    assert mean_f1 < mean_f2
