use rand::Rng;
use rand::distributions::Uniform;
use ndarray::Array1;

use crate::genetic::Genes;
use crate::operators::{GeneticOperator, CrossoverOperator};

#[derive(Clone, Debug)]
pub struct UniformBinaryCrossover;

impl UniformBinaryCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for UniformBinaryCrossover {
    fn name(&self) -> String {
        "UniformBinaryCrossover".to_string()
    }
}

impl CrossoverOperator<u8> for UniformBinaryCrossover {
    fn crossover<R>(
        &self,
        parent_a: &Genes<u8>,
        parent_b: &Genes<u8>,
        rng: &mut R,
    ) -> (Genes<u8>, Genes<u8>)
    where
        R: Rng + Sized,
    {
        assert_eq!(
            parent_a.len(),
            parent_b.len(),
            "Parents must have the same number of genes"
        );

        let num_genes = parent_a.len();
        let mut offspring_a = Array1::zeros(num_genes);
        let mut offspring_b = Array1::zeros(num_genes);

        let uniform = Uniform::new(0.0f64, 1.0);

        for i in 0..num_genes {
            if rng.sample(uniform) < 0.5 {
                // Swap genes
                offspring_a[i] = parent_b[i];
                offspring_b[i] = parent_a[i];
            } else {
                // Keep genes
                offspring_a[i] = parent_a[i];
                offspring_b[i] = parent_b[i];
            }
        }

        (offspring_a, offspring_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_uniform_binary_crossover() {
        let parent_a = array![0u8, 1, 1, 0, 1];
        let parent_b = array![1u8, 0, 0, 1, 0];

        let crossover_operator = UniformBinaryCrossover::new();
        let mut rng = StdRng::seed_from_u64(42);

        let (offspring_a, offspring_b) = crossover_operator.crossover(&parent_a, &parent_b, &mut rng);

        // Expected offspring based on the fixed seed and swapping decisions
        let expected_offspring_a = array![0u8, 1, 1, 1, 0];
        let expected_offspring_b = array![1u8, 0, 0, 0, 1];

        assert_eq!(offspring_a, expected_offspring_a);
        assert_eq!(offspring_b, expected_offspring_b);
    }
}
