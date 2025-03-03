import time

import pytest
import numpy as np
from scipy.spatial.distance import cdist  # type: ignore

from pymoors._pymoors import cross_euclidean_distances  # type: ignore


@pytest.mark.parametrize("n", [100, 500, 1000, 2000])
def test_compare_scipy_cdist_vs_pymoors(benchmark, n):
    matrix = np.random.rand(n, n)
    result_pymoors = benchmark(cross_euclidean_distances, matrix, matrix)
    # Measure the time for scipy's cdist using a timer
    start_time = time.perf_counter()
    result_scipy = cdist(matrix, matrix, metric="sqeuclidean")
    end_time = time.perf_counter()
    time_scipy = end_time - start_time
    # Check the result is the same
    np.testing.assert_allclose(result_pymoors, result_scipy, rtol=1e-5, atol=1e-8)
    # Check the time
    mean_time_pymoors = benchmark.stats.stats.mean
    assert mean_time_pymoors < time_scipy
