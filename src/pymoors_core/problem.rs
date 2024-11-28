// use numpy::pyo3::Python;
// use numpy::ndarray::Array2;
// use numpy::{PyArray2, ToPyArray};
// use pyo3::prelude::*;
// use pyo3::types::PyDict;

// pub trait Problem {
//     fn evaluate(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>>;
// }

// #[pyclass(subclass)]
// pub struct PyProblem {
//     #[pyo3(set)]
//     pub n_var: usize,
//     #[pyo3(set)]
//     pub n_obj: usize,
//     #[pyo3(set)]
//     pub n_ieq_constr: usize,
//     #[pyo3(set)]
//     pub xl: f64,
//     #[pyo3(set)]
//     pub xu: f64,

//     #[pyo3(get)]
//     py_obj: PyObject,
// }

// #[pymethods]
// impl PyProblem {
//     #[new]
//     fn new(py_obj: PyObject) -> Self {
//         PyProblem {
//             n_var: 0,
//             n_obj: 0,
//             n_ieq_constr: 0,
//             xl: 0.0,
//             xu: 0.0,
//             py_obj,
//         }
//     }
// }

// impl Problem for PyProblem {
//     fn evaluate(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
//         let result = Python::with_gil(|py| -> PyResult<_> {
//             let py_obj = self.py_obj.as_ref(py);

//             let x_py = x.to_pyarray(py);
//             let result_py = py_obj.call_method("_evaluate", (x_py,), None)?;
//             let result_dict = result_py.downcast::<PyDict>()?;
//             let f_py = result_dict.get_item("F");
//             let g_py = result_dict.get_item("G");
//             let f_pyarray: &PyArray2<f64> = f_py.extract()?;
//             let g_pyarray: &PyArray2<f64> = g_py.extract()?;

//             let f_rust = g_f_pyarray.to_owned_array();
//             let g_rust = g_pyarray.to_owned_array();

//             Ok((f_rust, g_rust))
//         });

//         result.map_err(|e| e.into())
//     }
// }
