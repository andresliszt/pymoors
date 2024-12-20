import pytest

from pymoors.modeling.variable import Variable
from pymoors.modeling.add import Add
from pymoors.modeling.mul import Mul
from pymoors.modeling.constant import Constant
from pymoors.modeling.pow import Pow


def test_add_variables() -> None:
    x = Variable(name='x')
    y = Variable(name='y')
    z = x + x + y
    # Check instance
    assert isinstance(z, Add)
    # Check grouped args
    assert z.args[0] == Mul(2, x) == 2 * x
    assert z.args[1] == y
    # Check the opposite
    assert z - z == 0
    # Check the identity
    assert z + 0 == z + Constant(value=0) == z
    assert 2 * z == z + z == 4 * Variable('x') + 2 * Variable("y") == 4 * x + 2 * y

    # Missmatch length
    with pytest.raises(ValueError):
        w = Variable('w', length=10)
        _ = z + w

    with pytest.raises(ValueError):
        w = Variable('w', length=10)
        _ = w + z


def test_mul_variables() -> None:
    x = Variable(name='x')
    y = Variable(name='y')
    z = x * x * y
    # Check isntance
    assert isinstance(z, Mul)
    # Check grouped args
    assert z.args[0] == Pow(x, 2) == x**2
    assert z.args[1] == y
    # Check expand
    assert (x + 1) * (x + 2) == x**2 + 3 * x + 2 == ((x + 1) * (x + 2)).expand()


def test_pow() -> None:
    x = Variable(name='x')
    z = x**2
    # Check instance
    assert isinstance(z, Pow)
    # Check base and exp
    assert z.base == x
    assert z.exp == 2
    # Check groups
    assert z**2 == Pow(x, 4)
    # Check expand
    assert (x + 1)**2 == x*+2 + 2*x +1 == ((x+1)**2).expand()
