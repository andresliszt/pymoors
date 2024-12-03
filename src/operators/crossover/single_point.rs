use rand::Rng;
use ndarray::{Axis, Array1, concatenate, s};

use crate::operators::{GeneticOperator, CrossoverOperator};
use crate::genetic::Genes;

#[derive(Clone, Debug)]
pub struct SinglePointBinaryCrossover;

impl SinglePointBinaryCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for SinglePointBinaryCrossover {
    fn name(&self) -> String {
        "SinglePointBinaryCrossover".to_string()
    }
}

impl CrossoverOperator<u8> for SinglePointBinaryCrossover {
    fn crossover<R>(
        &self,
        parent_a: &Genes<u8>,
        parent_b: &Genes<u8>,
        rng: &mut R,
    ) -> (Genes<u8>, Genes<u8>)
    where
        R: Rng + Sized,
    {
        let num_genes = parent_a.len();
        assert_eq!(
            num_genes,
            parent_b.len(),
            "Parents must have the same number of genes"
        );

        if num_genes == 0 {
            return (Array1::default(0), Array1::default(0));
        }

        // Choose a crossover point between 1 and num_genes - 1
        let crossover_point = rng.gen_range(1..num_genes);

        // Split parents at the crossover point and create offspring
        let offspring_a = concatenate![
            Axis(0),
            parent_a.slice(s![..crossover_point]),
            parent_b.slice(s![crossover_point..])
        ];

        let offspring_b = concatenate![
            Axis(0),
            parent_b.slice(s![..crossover_point]),
            parent_a.slice(s![crossover_point..])
        ];

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
    fn test_single_point_binary_crossover() {
        let parent_a = array![0u8, 1, 1, 0, 1];
        let parent_b = array![1u8, 0, 0, 1, 0];

        let crossover_operator = SinglePointBinaryCrossover::new();
        let mut rng = StdRng::seed_from_u64(42);

        let (offspring_a, offspring_b) = crossover_operator.crossover(&parent_a, &parent_b, &mut rng);

        // Since the seed is fixed, the crossover point is deterministic
        // For the seed 42 and num_genes = 5, crossover_point should be 3
        let expected_offspring_a = array![0u8, 1, 1, 1, 0];
        let expected_offspring_b = array![1u8, 0, 0, 0, 1];

        assert_eq!(offspring_a, expected_offspring_a);
        assert_eq!(offspring_b, expected_offspring_b);
    }
}
