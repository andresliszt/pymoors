from typing import Annotated, Callable, Literal, TypeVar
import numpy as np
import numpy.typing as npt


DType = TypeVar("DType", bound=np.generic)

OneDArray = Annotated[npt.NDArray[DType], Literal["N"]]
TwoDArray = Annotated[npt.NDArray[DType], Literal["N", "M"]]
ThreeDArray = Annotated[npt.NDArray[DType], Literal["N", "M", "K"]]

PopulationCallable = Callable[[TwoDArray], TwoDArray]
