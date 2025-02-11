use crate::operators::{GenesMut, GeneticOperator, MutationOperator};
use pyo3::prelude::*;
use rand::{Rng, RngCore};

#[derive(Clone, Debug)]
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl GeneticOperator for BitFlipMutation {
    fn name(&self) -> String {
        "BitFlipMutation".to_string()
    }
}

impl MutationOperator for BitFlipMutation {
    fn mutate<'a>(&self, mut individual: GenesMut<'a>, rng: &mut dyn RngCore) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Mutation operator that flips bits in a binary individual with a specified mutation rate.
#[pyclass(name = "BitFlipMutation")]
#[derive(Clone)] // So we can clone when converting to Box<dyn MutationOperator>
pub struct PyBitFlipMutation {
    // The actual Rust struct
    pub inner: BitFlipMutation,
}

#[pymethods]
impl PyBitFlipMutation {
    /// Python constructor: `BitFlipMutation(gene_mutation_rate=0.05)`
    #[new]
    fn new(gene_mutation_rate: f64) -> Self {
        let bitflip = BitFlipMutation::new(gene_mutation_rate);
        Self { inner: bitflip }
    }

    #[getter]
    fn get_gene_mutation_rate(&self) -> f64 {
        self.inner.gene_mutation_rate
    }

    #[setter]
    fn set_gene_mutation_rate(&mut self, value: f64) {
        self.inner.gene_mutation_rate = value;
    }
}

// Tests module
#[cfg(test)]
mod tests {
    use super::*; // Bring all items from the parent module into scope
    use crate::genetic::PopulationGenes;
    use numpy::ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_bit_flip_mutation() {
        // Create an individual with known genes
        let mut pop: PopulationGenes = array![[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]];

        // Create a BitFlipMutation operator with a high gene mutation rate
        let mutation_operator = BitFlipMutation::new(1.0); // Ensure all bits are flipped

        // Use a fixed seed for RNG to make the test deterministic
        let mut rng = StdRng::seed_from_u64(42);

        println!("Original population: {:?}", pop);
        // Mutate the population
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // Check that all bits have been flipped
        let expected_pop: PopulationGenes =
            array![[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(expected_pop, pop);

        println!("Mutated population: {:?}", pop);
    }
}
