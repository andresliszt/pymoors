use crate::operators::{Genes, GeneticOperator, MutationOperator};
use pyo3::prelude::*;
use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct GaussianMutation {
    pub gene_mutation_rate: f64,
    pub sigma: f64,
}

impl GaussianMutation {
    pub fn new(gene_mutation_rate: f64, sigma: f64) -> Self {
        Self {
            gene_mutation_rate,
            sigma,
        }
    }
}

impl GeneticOperator for GaussianMutation {
    fn name(&self) -> String {
        "GaussianMutation".to_string()
    }
}

impl MutationOperator for GaussianMutation {
    fn mutate(&self, individual: &Genes, rng: &mut dyn RngCore) -> Genes {
        // Make a copy of the original individual's genes
        let mut mutated = individual.clone();

        // Prepare the normal distribution
        let normal_dist = Normal::new(0.0, self.sigma)
            .expect("Failed to create normal distribution. Sigma must be > 0.");

        // For each gene, decide if we mutate
        for val in mutated.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                // Sample from Normal(0, sigma) and add to the current gene
                let delta = normal_dist.sample(rng);
                *val += delta;
            }
        }

        mutated
    }
}

/// A Python class that encapsulates our Rust `GaussianMutation`.
#[pyclass(name = "GaussianMutation")]
#[derive(Clone)]
pub struct PyGaussianMutation {
    pub inner: GaussianMutation,
}

#[pymethods]
impl PyGaussianMutation {
    /// Python constructor: `GaussianMutation(gene_mutation_rate=0.1, sigma=0.01)`
    #[new]
    fn new(gene_mutation_rate: f64, sigma: f64) -> Self {
        let gmut = GaussianMutation::new(gene_mutation_rate, sigma);
        Self { inner: gmut }
    }

    #[getter]
    fn get_gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }

    #[setter]
    fn set_gene_mutation_rate(&mut self, value: f64) {
        self.inner.gene_mutation_rate = value;
    }

    #[getter]
    fn get_sigma(&self) -> f64 {
        self.inner.sigma
    }

    #[setter]
    fn set_sigma(&mut self, value: f64) {
        self.inner.sigma = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::PopulationGenes;
    use numpy::ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gaussian_mutation_all_genes() {
        // Create an individual [0.5, 0.5, 0.5]
        let pop: PopulationGenes = array![[0.5, 0.5, 0.5]];

        // Create operator with 100% chance each gene is mutated, sigma=0.1
        let mutation_operator = GaussianMutation::new(1.0, 0.1);

        // Use a fixed seed for deterministic test
        let mut rng = StdRng::seed_from_u64(42);

        // Mutate the population
        let mutated_pop = mutation_operator.operate(&pop, 1.0, &mut rng);

        // mutated_pop should differ from pop, but let's just ensure it's not identical
        assert_ne!(pop, mutated_pop);

        println!("Original: {:?}", pop);
        println!("Mutated: {:?}", mutated_pop);
    }

    #[test]
    fn test_gaussian_mutation_no_genes() {
        // If gene_mutation_rate=0.0, no genes are mutated
        let pop: PopulationGenes = array![[0.5, 0.5, 0.5]];
        let mutation_operator = GaussianMutation::new(0.0, 0.1);

        let mut rng = StdRng::seed_from_u64(42);
        let mutated_pop = mutation_operator.operate(&pop, 1.0, &mut rng);

        assert_eq!(pop, mutated_pop);
    }
}
