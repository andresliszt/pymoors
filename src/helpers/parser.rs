use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::operators::{MutationOperator, CrossoverOperator, SamplingOperator};

pub fn unwrap_mutation_operator(
    mutation_obj: PyObject
) -> PyResult<Box<dyn MutationOperator>> {
    Python::with_gil(|py| {
        // Try PyBitFlipMutation
        if let Ok(extracted) = mutation_obj.extract::<crate::operators::py_operators::PyBitFlipMutation>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn MutationOperator>)
        }
        // If none match, return an error
        else {
            Err(PyValueError::new_err(
                "Unsupported or unknown mutation operator object",
            ))
        }
    })
}

pub fn unwrap_crossover_operator(
    crossover_obj: PyObject
) -> PyResult<Box<dyn CrossoverOperator>> {
    Python::with_gil(|py| {
        if let Ok(extracted) = crossover_obj.extract::<crate::operators::py_operators::PySinglePointBinaryCrossover>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn CrossoverOperator>)
        }
        else if let Ok(extracted) = crossover_obj.extract::<crate::operators::py_operators::PyUniformBinaryCrossover>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn CrossoverOperator>)
        }
        else {
            Err(PyValueError::new_err(
                "Unsupported or unknown crossover operator object",
            ))
        }
    })
}

pub fn unwrap_sampling_operator(py_obj: PyObject) -> PyResult<Box<dyn SamplingOperator>> {
    Python::with_gil(|py| {
        // 1) Try extracting PyRandomSamplingFloat
        if let Ok(extracted) = py_obj.extract::<crate::operators::py_operators::PyRandomSamplingFloat>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn SamplingOperator>)
        }
        // 2) Try extracting PyRandomSamplingInt
        else if let Ok(extracted) = py_obj.extract::<crate::operators::py_operators::PyRandomSamplingInt>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn SamplingOperator>)
        }
        // 3) Try extracting PyRandomSamplingBinary
        else if let Ok(extracted) = py_obj.extract::<crate::operators::py_operators::PyRandomSamplingBinary>(py) {
            Ok(Box::new(extracted.inner) as Box<dyn SamplingOperator>)
        } else {
            Err(PyValueError::new_err(
                "Unsupported or unknown sampling operator",
            ))
        }
    })
}
