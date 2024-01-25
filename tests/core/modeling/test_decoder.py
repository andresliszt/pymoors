import pytest
import numpy as np

from pymoors.core.modeling.variable import Variable
from pymoors.core.modeling.decoder import LinearDecoder
from pymoors.typing import OneDArray

# pylint: disable=W0123,W0612


@pytest.mark.parametrize(
    "add_expression_str, expected_coefficients",
    [
        ("x[0]", np.array([1.0, 0.0, 0.0])),
        ("x", np.array([1.0, 1.0, 1.0])),
        ("x[0] + x[1] + x[2]", np.array([1.0, 1.0, 1.0])),
        ("(x[0] + x[1] + x[2]) + (x[0] + x[1] + x[2])", np.array([2.0, 2.0, 2.0])),
        ("2*x", np.array([2.0, 2.0, 2.0])),
        ("x[0] - x[0]", np.array([0.0, 0.0, 0.0])),
        ("x[0] + x[1] -2*x[0]", np.array([-1.0, 1.0, 0.0])),
        ("(x[0] + x[1]) -2*(x[0])", np.array([-1.0, 1.0, 0.0])),
        ("0*x", np.array([0.0, 0.0, 0.0])),
    ],
)
def test_linear_decoder_variable_multidimensional(
    add_expression_str: str, expected_coefficients: OneDArray
):
    x = Variable(length=3)
    decoder = LinearDecoder(variables=x)
    result: OneDArray = decoder.decode(eval(add_expression_str))
    np.testing.assert_equal(result, expected_coefficients)


@pytest.mark.parametrize(
    "add_expression_str, expected_coefficients",
    [
        ("x", np.array([1.0, 0.0, 0.0])),
        ("x + y + z", np.array([1.0, 1.0, 1.0])),
        ("(x + y + z) + (x + y + z)", np.array([2.0, 2.0, 2.0])),
        ("2*(x + y + z)", np.array([2.0, 2.0, 2.0])),
        ("x - x", np.array([0.0, 0.0, 0.0])),
        ("x + y - 2*x", np.array([-1.0, 1.0, 0.0])),
        ("(x + y) -2*(x)", np.array([-1.0, 1.0, 0.0])),
        ("0*x + 0*y + z", np.array([0.0, 0.0, 1.0])),
    ],
)
def test_linear_decoder_one_dimensional_variables(
    add_expression_str: str, expected_coefficients: OneDArray
):
    x = Variable(length=1)
    y = Variable(length=1)
    z = Variable(length=1)
    decoder = LinearDecoder(variables=[x, y, z])
    result: OneDArray = decoder.decode(eval(add_expression_str))
    np.testing.assert_equal(result, expected_coefficients)
