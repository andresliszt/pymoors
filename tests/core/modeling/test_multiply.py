import pytest

from pymoors.core.modeling.multiply import MultiplyExpression
from pymoors import Variable, Constant

from pydantic import BaseModel


class TestMultiplyExpression:
    @pytest.fixture(autouse=True)
    def mock_eq_method_using_pydantic(self, monkeypatch):
        # Mocks Expression __eq__ to compare instance attributes instead of returning Equality
        monkeypatch.setattr(
            "pymoors.core.modeling.expression.Expression.__eq__", BaseModel.__eq__
        )

    def test_multiply(self):
        mul = Constant(value=10) * Variable(length=1)
        import pdb

        pdb.set_trace()
