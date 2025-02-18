use pyo3::create_exception;
use pyo3::exceptions::PyBaseException;

// Raise this error when no feasible individuals are found
create_exception!(
    pymoors,
    NoFeasibleIndividualsError,
    PyBaseException,
    "Raise this error when no feasible individuals are found"
);

// Raised when an invalid parameter value is provided
create_exception!(
    pymoors,
    InvalidParameterError,
    PyBaseException,
    "Raised when an invalid parameter value is provided"
);
