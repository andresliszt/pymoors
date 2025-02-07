use crate::operators::{Genes, GeneticOperator, SamplingOperator};
use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rand::{seq::SliceRandom, RngCore};
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
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RngCore) -> Genes {
        // 1) Create a vector of indices [0, 1, 2, ..., n_vars - 1]
        let mut indices: Vec<f64> = (0..n_vars).map(|i| i as f64).collect();

        // 2) Shuffle the indices in-place using the `SliceRandom` trait
        indices.shuffle(rng);

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
mod tests {
    use super::*; // PermutationSampling
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_permutation_sampling() {
        let sampler = PermutationSampling;
        let mut rng = StdRng::seed_from_u64(42);

        let pop_size = 5;
        let n_vars = 4; // for example

        // Generate multiple individuals
        let population = sampler.operate(pop_size, n_vars, &mut rng);

        // population is an ndarray (PopulationGenes) of shape (pop_size, n_vars)
        assert_eq!(population.nrows(), pop_size);
        assert_eq!(population.ncols(), n_vars);

        for row in population.outer_iter() {
            // Convert row to a Vec<f64>
            let perm: Vec<f64> = row.to_vec();
            // Check it's length 4
            assert_eq!(perm.len(), 4);
            // Check that all elements are distinct and within [0..4)
            // a) Convert to i32 if you want integer check:
            let mut integers: Vec<i32> = perm.iter().map(|&x| x as i32).collect();
            integers.sort_unstable();
            assert_eq!(integers, vec![0, 1, 2, 3]);
        }
    }
}
