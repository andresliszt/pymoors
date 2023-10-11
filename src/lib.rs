extern crate core;

mod non_dominated_sorting;
use pyo3::{prelude::*, wrap_pymodule};

pub use non_dominated_sorting::dominator::fast_non_dominated_sorting_py;

/// dominator module implemented in Rust
#[pymodule]
fn dominance(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(fast_non_dominated_sorting_py))?;
    Ok(())
}

#[pymodule]
fn pymoors(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(dominance))?;
    Ok(())
}
