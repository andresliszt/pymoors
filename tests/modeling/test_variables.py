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
    assert isinstance(z, Add)
    assert z.args[0] == Mul(2, x) == 2 * x
    assert z.args[1] == y
    import pdb

    pdb.set_trace()
    assert z + 0 == z
    assert z - z == 0
    # Missmatch length
    with pytest.raises(ValueError):
        w = Variable('w', length=10)
        _ = z + w


def test_mul_variables() -> None:
    x = Variable(name='x')
    y = Variable(name='y')
    z = x * x * y
    import pdb

    pdb.set_trace()
