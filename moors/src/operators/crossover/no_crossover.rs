use ndarray::Array1;

use crate::{operators::CrossoverOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// NOP crossover operator: returns parents unchanged as offspring.
pub struct NoCrossover;

impl NoCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl CrossoverOperator for NoCrossover {
    #[inline]
    fn crossover(
        &self,
        parent_a: &Array1<f64>,
        parent_b: &Array1<f64>,
        _rng: &mut impl RandomGenerator,
    ) -> (Array1<f64>, Array1<f64>) {
        (parent_a.clone(), parent_b.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::MOORandomGenerator;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_no_crossover_returns_parents_unchanged() {
        let parent_a = array![1.0, 2.0, 3.0];
        let parent_b = array![4.0, 5.0, 6.0];
        let mut rng = MOORandomGenerator::new(StdRng::seed_from_u64(42));
        let op = NoCrossover::new();

        let (child_a, child_b) = op.crossover(&parent_a, &parent_b, &mut rng);

        assert_eq!(child_a, parent_a);
        assert_eq!(child_b, parent_b);
    }
}
