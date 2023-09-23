use crate::operators::{Genes, GeneticOperator, MutationOperator};
use pyo3::prelude::*;
use rand::{Rng, RngCore};

#[derive(Clone, Debug)]
pub struct SwapMutation;

impl SwapMutation {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for SwapMutation {
    fn name(&self) -> String {
        "SwapMutation".to_string()
    }
}

/// In a typical permutation-based setup, each row is an array of distinct values.
/// The "swap" mutation picks two indices at random and swaps them.
impl MutationOperator for SwapMutation {
    fn mutate(&self, individual: &Genes, rng: &mut dyn RngCore) -> Genes {
        let mut mutated = individual.clone();
        let length = mutated.len();

        // If there's at most 1 element, there's nothing to swap
        if length > 1 {
            // Pick two distinct indices
            let idx1 = rng.gen_range(0..length);
            let mut idx2 = rng.gen_range(0..length);
            while idx2 == idx1 {
                idx2 = rng.gen_range(0..length);
            }

            // Swap the elements
            let tmp = mutated[idx1];
            mutated[idx1] = mutated[idx2];
            mutated[idx2] = tmp;
        }

        mutated
    }
}

/// A Python class that encapsulates our Rust `SwapMutation`.
#[pyclass(name = "SwapMutation")]
#[derive(Clone)]
pub struct PySwapMutation {
    // The actual Rust struct
    pub inner: SwapMutation,
}

#[pymethods]
impl PySwapMutation {
    /// Python constructor: `SwapMutation()`
    #[new]
    fn new() -> Self {
        let swap = SwapMutation::new();
        Self { inner: swap }
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
    fn test_swap_mutation_always_swaps_two_positions() {
        // Create an individual: [0,1,2,3,4]
        let original: PopulationGenes = array![[0.0, 1.0, 2.0, 3.0, 4.0]];

        // Create a SwapMutation operator (no probability inside)
        let mutation_operator = SwapMutation::new();

        // Fixed RNG seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        // Mutate the population (pop_size = 1)
        // Using your usual "operate" method:
        let mutated_pop = mutation_operator.operate(&original, 1.0, &mut rng);

        // There's only 1 row, so we grab it
        let mutated_individual = mutated_pop.row(0).to_owned();
        let original_row = original.row(0).to_owned();

        // Check how many positions changed
        let mut diff_positions = Vec::new();
        for i in 0..original_row.len() {
            if original_row[i] != mutated_individual[i] {
                diff_positions.push(i);
            }
        }

        // Exactly 2 positions should differ after one swap
        assert_eq!(diff_positions.len(), 2);

        // Check that the elements at those two positions are swapped
        let i = diff_positions[0];
        let j = diff_positions[1];
        assert_eq!(original_row[i], mutated_individual[j]);
        assert_eq!(original_row[j], mutated_individual[i]);
    }
}
