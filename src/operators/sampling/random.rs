use crate::genetic::Genes;
use crate::operators::{GeneticOperator, SamplingOperator};
use crate::random::RandomGenerator;
use pyo3::prelude::*;
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
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> Genes {
        (0..n_vars)
            .map(|_| rng.gen_range_f64(self.min, self.max))
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
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> Genes {
        (0..n_vars)
            .map(|_| rng.gen_range_f64(self.min as f64, self.max as f64))
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
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> Genes {
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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use rand::RngCore;

    /// A controlled fake RandomGenerator for testing purposes.
    /// It returns predictable values:
    /// - `gen_range_f64(min, _max)` always returns `min`
    /// - `gen_bool(_p)` always returns `false`
    struct FakeRandomGenerator {
        dummy: TestDummyRng,
    }

    impl FakeRandomGenerator {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, min: usize, _max: usize) -> usize {
            min
        }
        fn gen_range_f64(&mut self, min: f64, _max: f64) -> f64 {
            min
        }
        fn gen_usize(&mut self) -> usize {
            0
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            false
        }
    }

    #[test]
    fn test_random_sampling_float_controlled() {
        let sampler = RandomSamplingFloat {
            min: -1.0,
            max: 1.0,
        };
        let mut rng = FakeRandomGenerator::new();

        // Generate a population of 10 individuals, each with 5 genes.
        let population = sampler.operate(10, 5, &mut rng);

        // Since our fake returns the minimum for every call to `gen_range_f64`,
        // every gene in the population should be -1.0.
        for gene in population.iter() {
            assert_eq!(*gene, -1.0);
        }
    }

    #[test]
    fn test_random_sampling_int_controlled() {
        let sampler = RandomSamplingInt { min: 0, max: 10 };
        let mut rng = FakeRandomGenerator::new();

        let population = sampler.operate(10, 5, &mut rng);

        // The operator uses `gen_range_f64` (with `min` as 0.0) for each gene,
        // so every gene should be 0.0.
        for gene in population.iter() {
            assert_eq!(*gene, 0.0);
        }
    }

    #[test]
    fn test_random_sampling_binary_controlled() {
        let sampler = RandomSamplingBinary;
        let mut rng = FakeRandomGenerator::new();

        let population = sampler.operate(10, 5, &mut rng);

        // Since our fake returns false for every call to `gen_bool(0.5)`,
        // each gene will be 0.0 (because the sampling operator maps false to 0.0).
        for gene in population.iter() {
            assert_eq!(*gene, 0.0);
        }
    }
}
