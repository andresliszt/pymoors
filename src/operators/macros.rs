// Define the common submacro to implement the constructor, getters, and setters.
macro_rules! impl_py_common {
    ($rust_struct:ident $(, $field:ident : $type:ty )* $(,)?) => {
        paste::paste! {
            #[pymethods]
            impl [<Py $rust_struct>] {
                #[new]
                pub fn new( $( $field: $type ),* ) -> Self {
                    Self {
                        inner: $rust_struct::new( $( $field ),* )
                    }
                }

                $(
                    #[getter]
                    pub fn $field(&self) -> $type {
                        self.inner.$field
                    }
                )*
            }
        }
    }
}

#[macro_export]
macro_rules! impl_py_mutation {
    ($doc:expr, $rust_struct:ident, $py_name:expr $(, $field:ident : $type:ty )* $(,)?) => {
        use pyo3::prelude::*;
        use numpy::{PyArray2, PyReadonlyArrayDyn};
        use ndarray::Ix2;
        use pyo3::exceptions::PyValueError;
        use numpy::{
            PyArrayMethods,
            ToPyArray,
        };
        use crate::random::MOORandomGenerator;

        paste::paste! {
            #[doc = $doc]
            #[pyclass(name = $py_name)]
            #[derive(Clone)]
            pub struct [<Py $rust_struct>] {
                pub inner: $rust_struct,
            }

            // Inject common implementation (constructor, getters, and setters).
            impl_py_common!($rust_struct $(, $field : $type )*);

            // Additional mutation-specific methods.
            #[pymethods]
            impl [<Py $rust_struct>]  {
                #[pyo3(signature = (population, seed=None))]
                pub fn mutate_population<'py>(
                    &self,
                    py: Python<'py>,
                    population: PyReadonlyArrayDyn<'py, f64>,
                    seed: Option<u64>,
                ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                    let owned_population = population.to_owned_array();
                    let mut owned_population = owned_population
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| PyValueError::new_err("Population numpy array must be 2D to use mutate_population."))?;

                    let mut rng = MOORandomGenerator::new_from_seed(seed);
                    self.inner.operate(&mut owned_population, 1.0, &mut rng);

                    Ok(owned_population.to_pyarray(py))
                }
            }
        }
    }
}

#[macro_export]
macro_rules! impl_py_crossover {
    ($doc:expr, $rust_struct:ident, $py_name:expr $(, $field:ident : $type:ty )* $(,)?) => {
        use pyo3::prelude::*;
        use numpy::{PyArray2, PyReadonlyArrayDyn};
        use ndarray::Ix2;
        use pyo3::exceptions::PyValueError;
        use numpy::{PyArrayMethods, ToPyArray};
        use crate::random::MOORandomGenerator;

        // Define the struct with the name generated as "Py" + $rust_struct.
        paste::paste! {
            #[doc = $doc]
            #[pyclass(name = $py_name)]
            #[derive(Clone)]
            pub struct [<Py $rust_struct>] {
                pub inner: $rust_struct,
            }
        }

        // Inject common implementation (constructor, getters, and setters)
        impl_py_common!($rust_struct $(, $field : $type )*);

        // Additional crossover-specific methods.
        paste::paste! {
            #[pymethods]
            impl [<Py $rust_struct>] {
                #[pyo3(signature = (parents_a, parents_b, seed=None))]
                pub fn crossover<'py>(
                    &self,
                    py: Python<'py>,
                    parents_a: PyReadonlyArrayDyn<'py, f64>,
                    parents_b: PyReadonlyArrayDyn<'py, f64>,
                    seed: Option<u64>,
                ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                    let owned_parents_a = parents_a.to_owned_array();
                    let owned_parents_b = parents_b.to_owned_array();
                    let owned_parents_a = owned_parents_a
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| PyValueError::new_err("parent_a numpy array must be 2D to use crossover."))?;
                    let owned_parents_b = owned_parents_b
                        .into_dimensionality::<Ix2>()
                        .map_err(|_| PyValueError::new_err("parent_b numpy array must be 2D to use crossover."))?;
                    let mut rng = MOORandomGenerator::new_from_seed(seed);
                    let offspring = self.inner.operate(&owned_parents_a, &owned_parents_b, 1.0, &mut rng);
                    Ok(offspring.to_pyarray(py))
                }
            }
        }
    }
}

#[macro_export]
macro_rules! impl_py_sampling {
    ($doc:expr, $rust_struct:ident, $py_name:expr $(, $field:ident : $type:ty )* $(,)?) => {
        use pyo3::prelude::*;
        use numpy::{
            PyArray2,
            ToPyArray,
        };
        use crate::random::MOORandomGenerator;

        // Define the struct using paste to concatenate "Py" with $rust_struct.
        paste::paste! {
            #[doc = $doc]
            #[pyclass(name = $py_name)]
            #[derive(Clone)]
            pub struct [<Py $rust_struct>] {
                pub inner: $rust_struct,
            }
        }

        // Inject common implementation (constructor, getters, and setters)
        impl_py_common!($rust_struct $(, $field : $type )*);

        // Additional sampling-specific methods.
        paste::paste! {
            #[pymethods]
            impl [<Py $rust_struct>] {
                #[pyo3(signature = (pop_size, n_vars, seed=None))]
                pub fn sample<'py>(
                    &self,
                    py: Python<'py>,
                    pop_size: usize,
                    n_vars: usize,
                    seed: Option<u64>,
                ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                    let mut rng = MOORandomGenerator::new_from_seed(seed);
                    let sampled_population = self.inner.operate(pop_size, n_vars, &mut rng);
                    Ok(sampled_population.to_pyarray(py))
                }
            }
        }
    }
}
