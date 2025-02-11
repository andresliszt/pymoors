#[macro_export]
macro_rules! define_multiobj_pyclass {
    ($StructName:ident, $PyClassName:literal) => {
        use numpy::ToPyArray;
        use pyo3::exceptions::PyRuntimeError;
        use pyo3::types::PyDict;
        use pyo3::IntoPyObject;

        use crate::algorithms::MultiObjectiveAlgorithm;

        // The PyO3-exposed struct
        #[pyclass(name = $PyClassName, unsendable)]
        pub struct $StructName {
            pub algorithm: MultiObjectiveAlgorithm,
        }

        // Implement PyO3 methods
        #[pymethods]
        impl $StructName {
            pub fn run(&mut self) -> PyResult<()> {
                self.algorithm
                    .run()
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            }

            // The population getter
            #[getter]
            pub fn population(&self, py: Python) -> PyResult<PyObject> {
                let schemas_module = py.import("pymoors.schemas")?;
                let population_class = schemas_module.getattr("Population")?;
                let population = &self.algorithm.population;

                // For each value we want to pass to Python, call the new conversion method.
                let py_genes = population.genes.to_pyarray(py);
                let py_fitness = population.fitness.to_pyarray(py);
                let py_rank = population.rank.to_pyarray(py);
                // For constraints, if Some then convert the array, else convert py.None()
                let py_constraints = if let Some(ref c) = population.constraints {
                    c.to_pyarray(py).into_py(py)
                } else {
                    py.None().into_py(py)
                };

                // Set up the keyword arguments.
                let kwargs = PyDict::new(py);
                kwargs.set_item("genes", py_genes)?;
                kwargs.set_item("fitness", py_fitness)?;
                kwargs.set_item("rank", py_rank)?;
                kwargs.set_item("constraints", py_constraints)?;

                // Call the python class to create an instance.
                let py_instance = population_class.call((), Some(&kwargs))?;

                // Convert the instance to a PyObject using the new trait.
                let py_instance = py_instance.into_pyobject(py).map_err(|e| {
                    PyRuntimeError::new_err(format!("Error converting instance: {:?}", e))
                })?;
                // Convert the Bound (or Borrowed) into a PyObject.
                Ok(py_instance.into())
            }
        }
    };
}
