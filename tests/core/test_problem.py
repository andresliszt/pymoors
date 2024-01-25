import numpy as np
from pymoors.core.modeling.variable import Variable
from pymoors.core.problem import Problem
from pymoors.typing import TwoDArray


def test_knapsack_problem():
    """
    Test Knapsack Problem formulation.

    **Problem Formulation**:

    Maximize the total sum of the values of the selected items:

    .. math::
       \max \sum_{i=1}^{n} v_i x_i

    Subject to the constraint that the total sum of the weights does not exceed the capacity of the knapsack:

    .. math::
       \sum_{i=1}^{n} w_i x_i \leq W

    Where:

    - :math:`n` is the total number of items.
    - :math:`v_i` is the value of item :math:`i`.
    - :math:`w_i` is the weight of item :math:`i`.
    - :math:`x_i` is a binary variable indicating whether item :math:`i` is (1) or is not (0) in the knapsack.
    - :math:`W` is the maximum capacity of the knapsack.

    """

    # In this test suite we use n = 4

    x = Variable(length=4)

    # Define weights and values
    V = [1, 3, 1, 2]
    W = [2, 1, 1.5, 3]

    # Objective as a callable
    def objective(population: TwoDArray) -> TwoDArray:
        return np.dot(np.array(V), population.T)

    # Define constraints
    constraints = sum(W[i] * x[i] for i in range(4))

    # Create problem object
    problem = Problem(variables=x, objectives=objective, constraints=constraints)
