use crate::operators::{Genes, GeneticOperator, SamplingOperator};
use crate::random::RandomGenerator;
use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use std::fmt::Debug;

/// A sampling operator that returns a random permutation of [0..n_vars).
#[derive(Clone, Debug)]
pub struct PermutationSampling;

impl GeneticOperator for PermutationSampling {
    fn name(&self) -> String {
        "PermutationSampling".to_string()
    }
}

impl SamplingOperator for PermutationSampling {
    /// Generates a single individual of length `n_vars` where the genes
    /// are a shuffled permutation of the integers [0, 1, 2, ..., n_vars - 1].
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> Genes {
        // 1) Create a vector of indices [0, 1, 2, ..., n_vars - 1]
        let mut indices: Vec<f64> = (0..n_vars).map(|i| i as f64).collect();

        // 2) Shuffle the indices in-place using the `SliceRandom` trait
        indices.shuffle(rng.rng());

        Array1::from_vec(indices)
    }
}

/// Sampling operator for permutation-based variables.
#[pyclass(name = "PermutationSampling")]
#[derive(Clone)]
pub struct PyPermutationSampling {
    pub inner: PermutationSampling,
}

#[pymethods]
impl PyPermutationSampling {
    /// Python constructor: `PermutationSampling()`
    #[new]
    fn new() -> Self {
        Self {
            inner: PermutationSampling,
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*; // Import PermutationSampling, etc.
    use crate::random::RandomGenerator;
    use rand::RngCore;

    pub struct TestDummyRng;

    impl RngCore for TestDummyRng {
        fn next_u32(&mut self) -> u32 {
            0
        }
        fn next_u64(&mut self) -> u64 {
            unimplemented!("Not used in this test")
        }
        fn fill_bytes(&mut self, _dest: &mut [u8]) {
            unimplemented!("Not used in this test")
        }
        fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand::Error> {
            unimplemented!("Not used in this test")
        }
    }

    /// A fake RandomGenerator for testing. It contains a TestDummyRng to provide
    /// the required RngCore behavior.
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
            // Return a mutable reference to the internal dummy RNG.
            &mut self.dummy
        }

        fn gen_range_usize(&mut self, min: usize, _max: usize) -> usize {
            // Not used by the shuffle, so just return min.
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
    fn test_permutation_sampling_controlled() {
        // Create the sampling operator.
        let sampler = PermutationSampling;
        // Use our fake RNG.
        let mut rng = FakeRandomGenerator::new();

        let pop_size = 5;
        let n_vars = 4; // For example, 4 variables

        // Generate the population. It is assumed that `operate` (defined via
        // the SamplingOperator trait) generates pop_size individuals.
        let population = sampler.operate(pop_size, n_vars, &mut rng);

        // Check the population shape.
        assert_eq!(population.nrows(), pop_size);
        assert_eq!(population.ncols(), n_vars);

        // With our dummy RNG always returning 0, the shuffle will behave as follows:
        //
        // For the initial vector [0.0, 1.0, 2.0, 3.0]:
        //  - i = 3: j = 0, swap positions 3 and 0 => [3.0, 1.0, 2.0, 0.0]
        //  - i = 2: j = 0, swap positions 2 and 0 => [2.0, 1.0, 3.0, 0.0]
        //  - i = 1: j = 0, swap positions 1 and 0 => [1.0, 2.0, 3.0, 0.0]
        //
        // Thus, the expected permutation for each individual is:
        let expected = vec![1.0, 2.0, 3.0, 0.0];

        // Verify that each individual's genes match the expected permutation.
        for row in population.outer_iter() {
            let perm: Vec<f64> = row.to_vec();
            assert_eq!(
                perm, expected,
                "The permutation did not match the expected value."
            );
        }
    }
}
