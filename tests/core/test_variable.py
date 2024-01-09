import pytest
import numpy as np

from pymoors.core.modeling.variable import Variable


def test_init():
    # Test defaults
    var = Variable()
    assert var.size == 1
    np.testing.assert_array_equal(var.coefficients, np.array([1]))
    # Test non defaults
    var = Variable(number_variables=10)
    assert var.size == 10
    np.testing.assert_array_equal(var.coefficients, np.ones(10))
    # Test coefficients vector
    var = Variable(number_variables=2, coefficients=np.array([2, 4]))
    assert var.size == 2
    np.testing.assert_array_equal(var.coefficients, np.array([2, 4]))
    # Test invalid setup
    with pytest.raises(ValueError):
        # Coefficients vector is larger than number variables
        Variable(number_variables=2, coefficients=np.array([2, 4, 6]))


def test_add_variables():
    x = Variable(number_variables=2)
    y = Variable(number_variables=2)
    z = Variable(number_variables=3)

    assert (x + y).expressions == [x, y]
    assert (y + x).expressions == [y, x]
    assert (x + x + y).expressions == [x, x, y]
    # Invalid sum
    with pytest.raises(
        ValueError,
        match='Two expressions with different sizes cannot be added. Got sizes 2 and 3',
    ):
        _ = x + y + z
