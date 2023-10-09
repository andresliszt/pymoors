extern crate core;

mod non_dominated_sorting;
use pyo3::prelude::*;

pub use non_dominated_sorting::dominator::fast_non_dominated_sorting_py;

/// dominator module implemented in Rust
#[pymodule]
#[pyo3(name = "pymoors")]
fn dominance(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_non_dominated_sorting_py, m)?)?;
    Ok(())
}
