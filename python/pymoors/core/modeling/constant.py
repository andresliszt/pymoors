from pymoors.core.modeling.expression import Expression


class Constant(Expression):
    value: float

    @property
    def size(self) -> int:
        """Constant size is always zero"""
        return 0
