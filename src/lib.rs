// lib.rs

extern crate core;

pub mod evaluator;
pub mod genetic;
pub mod prelude;

mod algorithms;
mod helpers;
mod non_dominated_sorting;
mod operators;

use pyo3::prelude::*;

pub use algorithms::nsga2::PyNsga2;
pub use operators::py_operators::{
    PyBitFlipMutation, PyRandomSamplingBinary, PyRandomSamplingFloat, PyRandomSamplingInt,
    PySinglePointBinaryCrossover, PyUniformBinaryCrossover,
};

/// Root module `pymoors` that includes all classes.
#[pymodule]
fn _pymoors(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add classes from algorithms
    m.add_class::<PyNsga2>()?;

    // Add classes from operators
    m.add_class::<PyBitFlipMutation>()?;
    m.add_class::<PyRandomSamplingBinary>()?;
    m.add_class::<PyRandomSamplingFloat>()?;
    m.add_class::<PyRandomSamplingInt>()?;
    m.add_class::<PySinglePointBinaryCrossover>()?;
    m.add_class::<PyUniformBinaryCrossover>()?;

    Ok(())
}
