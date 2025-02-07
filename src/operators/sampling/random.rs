use crate::genetic::Genes;
use crate::operators::{GeneticOperator, SamplingOperator};
use pyo3::prelude::*;
use rand::{Rng, RngCore};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct RandomSamplingFloat {
    pub min: f64,
    pub max: f64,
}

impl GeneticOperator for RandomSamplingFloat {
    fn name(&self) -> String {
        "RandomSamplingFloat".to_string()
    }
}

impl SamplingOperator for RandomSamplingFloat {
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RngCore) -> Genes {
        (0..n_vars)
            .map(|_| rng.gen_range(self.min..self.max))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingInt {
    pub min: i32,
    pub max: i32,
}

impl GeneticOperator for RandomSamplingInt {
    fn name(&self) -> String {
        "RandomSamplingInt".to_string()
    }
}

impl SamplingOperator for RandomSamplingInt {
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RngCore) -> Genes {
        (0..n_vars)
            .map(|_| rng.gen_range(self.min..self.max) as f64)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingBinary;

impl GeneticOperator for RandomSamplingBinary {
    fn name(&self) -> String {
        "RandomSamplingBinary".to_string()
    }
}

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RngCore) -> Genes {
        (0..n_vars)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}

// Sampling operator for floating-point variables using uniform random distribution.
#[pyclass(name = "RandomSamplingFloat")]
#[derive(Clone)]
pub struct PyRandomSamplingFloat {
    pub inner: RandomSamplingFloat,
}

#[pymethods]
impl PyRandomSamplingFloat {
    /// Python constructor: `RandomSamplingFloat(min, max)`
    #[new]
    fn new(min: f64, max: f64) -> Self {
        Self {
            inner: RandomSamplingFloat { min, max },
        }
    }

    #[getter]
    fn get_min(&self) -> f64 {
        self.inner.min
    }

    #[setter]
    fn set_min(&mut self, value: f64) {
        self.inner.min = value;
    }

    #[getter]
    fn get_max(&self) -> f64 {
        self.inner.max
    }

    #[setter]
    fn set_max(&mut self, value: f64) {
        self.inner.max = value;
    }
}

/// Sampling operator for integer variables using uniform random distribution.
#[pyclass(name = "RandomSamplingInt")]
#[derive(Clone)]
pub struct PyRandomSamplingInt {
    pub inner: RandomSamplingInt,
}

#[pymethods]
impl PyRandomSamplingInt {
    /// Python constructor: `RandomSamplingInt(min, max)`
    #[new]
    fn new(min: i32, max: i32) -> Self {
        Self {
            inner: RandomSamplingInt { min, max },
        }
    }

    #[getter]
    fn get_min(&self) -> i32 {
        self.inner.min
    }

    #[setter]
    fn set_min(&mut self, value: i32) {
        self.inner.min = value;
    }

    #[getter]
    fn get_max(&self) -> i32 {
        self.inner.max
    }

    #[setter]
    fn set_max(&mut self, value: i32) {
        self.inner.max = value;
    }
}

/// Sampling operator for binary variables.
#[pyclass(name = "RandomSamplingBinary")]
#[derive(Clone)]
pub struct PyRandomSamplingBinary {
    pub inner: RandomSamplingBinary,
}

#[pymethods]
impl PyRandomSamplingBinary {
    #[new]
    fn new() -> Self {
        Self {
            inner: RandomSamplingBinary,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_sampling_float() {
        let sampler = RandomSamplingFloat {
            min: -1.0,
            max: 1.0,
        };
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, 5, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene >= -1.0 && gene < 1.0);
        }
    }

    #[test]
    fn test_random_sampling_int() {
        let sampler = RandomSamplingInt { min: 0, max: 10 };
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, 5, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene >= 0.0 && gene < 10.0);
        }
    }

    #[test]
    fn test_random_sampling_binary() {
        let sampler = RandomSamplingBinary;
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, 5, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene == 0.0 || gene == 1.0);
        }
    }
}
