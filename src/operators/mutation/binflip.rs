use crate::operators::{GenesMut, GeneticOperator, MutationOperator, PopulationGenes};

use rand::prelude::*;

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

impl MutationOperator<u8> for BitFlipMutation {
    fn mutate<R>(&self, individual: &mut GenesMut<u8>, rng: &mut R)
    where
        R: Rng + Sized,
    {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene ^= 1; // Flip 0 to 1 and 1 to 0
            }
        }
    }
}

// Tests module
#[cfg(test)]
mod tests {
    use super::*; // Bring all items from the parent module into scope
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_bit_flip_mutation() {
        // Create an individual with known genes
        let mut pop: PopulationGenes<u8> = array![[0u8, 0, 0, 0, 0], [1u8, 1, 1, 1, 1]];

        // Create a BitFlipMutation operator with a high gene mutation rate
        let mutation_operator = BitFlipMutation::new(1.0); // Ensure all bits are flipped

        // Use a fixed seed for RNG to make the test deterministic
        let mut rng = StdRng::seed_from_u64(42);

        println!("Original population: {:?}", pop);
        // Mutate the individual
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // Check that all bits have been flipped
        let expected_pop: PopulationGenes<u8> = array![[1u8, 1, 1, 1, 1], [0u8, 0, 0, 0, 0]];
        assert_eq!(expected_pop, pop);

        println!("Mutated population: {:?}", pop);
    }
}
