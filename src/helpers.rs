use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray2, IntoPyArray, Element};
use numpy::ndarray::Array2;
use crate::genetic::{PopulationGenes, PopulationFitness};

pub trait FitnessFn {
    fn evaluate(&self, genes: &PopulationGenes) -> PopulationFitness;
}

/// `PyFitnessFn` holds a PyObject (the Python callback).
pub struct PyFitnessFn {
    callback: PyObject,
}

impl PyFitnessFn {
    pub fn new(callback: PyObject) -> Self {
        PyFitnessFn { callback }
    }
}

fn genes_to_numpy<'py>(py: Python<'py>, genes: &Array2<f64>) -> &'py PyArray2<f64> {
    genes.clone().into_pyarray(py)
}

impl FitnessFn for PyFitnessFn {
    fn evaluate(&self, genes: &PopulationGenes) -> PopulationFitness {
        // Acquire the GIL for Python
        Python::with_gil(|py| {
            // 1. Convert `genes` (Array2<f64>) to a `numpy.ndarray`
            let np_genes = genes_to_numpy(py, genes);

            // 2. Call the Python function:
            //    `self.callback( np_genes )`
            let args = PyTuple::new(py, &[np_genes]);
            let result = self.callback.call1(py, args);

            // 3. Handle errors
            let result = match result {
                Ok(res) => res,
                Err(err) => {
                    eprintln!("Error calling Python fitness function: {:?}", err);
                    // You can decide to panic or handle differently
                    panic!("Fitness callback failed");
                }
            };

            // 4. Extract the result as a NumPy array (PyArray2<f64>)
            let np_result = result
                .downcast::<PyArray2<f64>>(py)
                .expect("Expected Python function to return a numpy.ndarray");

            let fitness_values = np_result.to_owned_array();

            // 6. Return it as `PopulationFitness`
            fitness_values
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*; // bring `PyFitnessFn`, `FitnessFn`, etc. into scope
    use pyo3::types::PyModule;
    use numpy::ndarray::array;

    #[test]
    fn test_pyfitness_fn() {
        Python::with_gil(|py| {
            let code = r#"
def add_one(arr):
    import numpy as np
    # We assume `arr` is float64. Just add 1.0 to everything:
    return arr + 1.0
"#;
            let module = PyModule::from_code(py, code, "test_module.py", "test_module")
                .expect("Failed to create module");
            let func = module.getattr("add_one")
                .expect("Couldn't get Python function")
                .to_object(py);

            let pf = PyFitnessFn::new(func);

            // 2x2 array of f64
            let input = array![[1.0f64, 2.0], [3.0, 4.0]];
            let output = pf.evaluate(&input);

            // Expected: everything +1.0
            let expected = array![[2.0, 3.0], [4.0, 5.0]];
            assert_eq!(output, expected);
        });
    }
}
