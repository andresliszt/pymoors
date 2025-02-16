use crate::operators::{GenesMut, GeneticOperator, MutationOperator};
use crate::random::RandomGenerator;

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
    fn mutate<'a>(&self, mut individual: GenesMut<'a>, rng: &mut dyn RandomGenerator) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}

impl_py_mutation!(
    "Mutation operator that flips bits in a binary individual with a specified mutation rate",
    BitFlipMutation,
    "BitFlipMutation",
    gene_mutation_rate: f64
);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*; // Brings in BitFlipMutation and related types.
    use crate::genetic::PopulationGenes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use numpy::ndarray::array;
    use rand::RngCore;

    /// A fake RandomGenerator for testing that always returns `true` for `gen_bool`.
    struct FakeRandomGeneratorTrue {
        dummy: TestDummyRng,
    }

    impl FakeRandomGeneratorTrue {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGeneratorTrue {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            unimplemented!("Not used in test")
        }
        fn gen_range_f64(&mut self, _min: f64, _max: f64) -> f64 {
            unimplemented!("Not used in this test")
        }
        fn gen_usize(&mut self) -> usize {
            unimplemented!("Not used in this test")
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            // Always return true so that every gene is mutated.
            true
        }

        fn gen_proability(&mut self) -> f64 {
            1.0
        }
    }

    #[test]
    fn test_bit_flip_mutation_controlled() {
        // Create a population with two individuals:
        // - The first individual is all zeros.
        // - The second individual is all ones.
        let mut pop: PopulationGenes = array![[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]];

        // Create a BitFlipMutation operator with a gene mutation rate of 1.0,
        // so every gene should be considered for mutation.
        let mutation_operator = BitFlipMutation::new(1.0);
        assert_eq!(mutation_operator.name(), "BitFlipMutation");

        // Use our controlled fake RNG which always returns true for gen_bool.
        let mut rng = FakeRandomGeneratorTrue::new();

        // Mutate the population. The `operate` method (from MutationOperator) should
        // call `mutate` on each individual.
        mutation_operator.operate(&mut pop, 1.0, &mut rng);

        // After mutation, every bit should be flipped:
        // - The first individual (originally all 0.0) becomes all 1.0.
        // - The second individual (originally all 1.0) becomes all 0.0.
        let expected_pop: PopulationGenes =
            array![[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]];
        assert_eq!(expected_pop, pop);
    }
}
