use crate::define_multiobj_pyclass;
use crate::helpers::functions::{
    create_population_constraints_closure, create_population_fitness_closure,
};
use crate::helpers::parser::{
    unwrap_crossover_operator, unwrap_duplicates_cleaner, unwrap_mutation_operator,
    unwrap_sampling_operator,
};
use crate::operators::selection::RandomSelection;
use crate::operators::{
    survival::helpers::PyStructuredReferencePointsEnum, survival::nsga3::Nsga3ReferencePoints,
    survival::Nsga3ReferencePointsSurvival,
};

use numpy::{PyArray2, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

// Define the NSGA-III algorithm using your macro.
define_multiobj_pyclass!(Nsga3);

#[pymethods]
impl Nsga3 {
    #[new]
    #[pyo3(signature = (
        reference_points,
        sampler,
        crossover,
        mutation,
        fitness_fn,
        n_vars,
        population_size,
        n_offsprings,
        n_iterations,
        mutation_rate=0.1,
        crossover_rate=0.9,
        keep_infeasible=false,
        verbose=true,
        duplicates_cleaner=None,
        constraints_fn=None,
        lower_bound=None,
        upper_bound=None,
        seed=None
    ))]
    pub fn py_new(
        reference_points: PyObject,
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        n_vars: usize,
        population_size: usize,
        n_offsprings: usize,
        n_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        duplicates_cleaner: Option<PyObject>,
        constraints_fn: Option<PyObject>,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            // First, try to extract the object as our custom type.
            let reference_points_array: Nsga3ReferencePoints = if let Ok(custom_obj) =
                reference_points.extract::<PyStructuredReferencePointsEnum>(py)
            {
                Nsga3ReferencePoints::new(custom_obj.generate(), false)
            } else if let Ok(rp_maybe_array) = reference_points.downcast_bound::<PyArray2<f64>>(py)
            {
                Nsga3ReferencePoints::new(rp_maybe_array.readonly().as_array().to_owned(), true)
            } else {
                return Err(PyTypeError::new_err(
                        "reference_points must be either a custom reference points class or a NumPy array.",
                    ));
            };

            // Unwrap the genetic operators.
            let sampler_box = unwrap_sampling_operator(sampler)?;
            let crossover_box = unwrap_crossover_operator(crossover)?;
            let mutation_box = unwrap_mutation_operator(mutation)?;
            let duplicates_box = if let Some(py_obj) = duplicates_cleaner {
                Some(unwrap_duplicates_cleaner(py_obj)?)
            } else {
                None
            };

            // Build the mandatory population-level fitness closure.
            let fitness_closure = create_population_fitness_closure(fitness_fn)?;
            // Build the optional constraints closure.
            let constraints_closure = if let Some(py_obj) = constraints_fn {
                Some(create_population_constraints_closure(py_obj)?)
            } else {
                None
            };

            // Create instances of the selection and survival structs.
            let selector_box = Box::new(RandomSelection::new());
            let survivor_box = Box::new(Nsga3ReferencePointsSurvival::new(reference_points_array));

            // Build the algorithm.
            let algorithm = MultiObjectiveAlgorithm::new(
                sampler_box,
                selector_box,
                survivor_box,
                crossover_box,
                mutation_box,
                duplicates_box,
                fitness_closure,
                n_vars,
                population_size,
                n_offsprings,
                n_iterations,
                mutation_rate,
                crossover_rate,
                keep_infeasible,
                verbose,
                constraints_closure,
                lower_bound,
                upper_bound,
                seed,
            )?;

            Ok(Nsga3 { algorithm })
        })
    }
}
