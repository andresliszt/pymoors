from __future__ import annotations

import numpy as np
from pydantic import Field, ConfigDict, model_validator

from pymoors.core.modeling.expression import Expression
from pymoors.typing import OneDArray


class Variable(Expression):
    number_variables: int = Field(ge=1)
    coefficients: OneDArray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def validate_coefficents(cls, data):
        data.setdefault("number_variables", 1)
        if data.get("coefficients") is None:
            data["coefficients"] = np.ones(data["number_variables"])
        if len(data["coefficients"]) != data["number_variables"]:
            # TODO: Complete this error msg
            raise ValueError("")
        return data

    @property
    def size(self) -> int:
        return self.number_variables

    @property
    def name(self) -> str:
        return f"var_{self.expression_id}(size={self.size})"

    def decode(self) -> OneDArray:
        return self.coefficients
