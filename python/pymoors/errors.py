import abc
from typing import Any


class ErrorMixin(abc.ABC):
    """Mixing class for custom exceptions.

    Example:

        >>> class MyError(ErrorMixin):
                msg_template = "Value ``{value}`` is not found"
        >>> raise MyError(value="can't touch this")
        (...)
        MyError: Value `can't touch this` is not found
    """

    def __init__(self, **values: Any):
        self.values = values
        super().__init__()

    @property
    @abc.abstractmethod
    def msg_template(self) -> str:
        ...

    def __str__(self) -> str:
        txt = self.msg_template
        for name, value in self.values.items():
            txt = txt.replace("{" + name + "}", str(value))
        txt = txt.replace("`{", "").replace("}`", "")
        return txt


class CrossoverOutputError(ErrorMixin, ValueError):
    @property
    def msg_template(self):
        return "Expected number of offsprings is `{expected_offsprings}` for the Crossover operator. Got `{recived_offsprings}`"
