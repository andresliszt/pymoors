use ndarray::ArrayViewMut1;

use crate::{operators::MutationOperator, random::RandomGenerator};

#[derive(Debug, Clone)]
/// NOP Mutation operator
pub struct NoMutation;

impl NoMutation {
    pub fn new() -> Self {
        Self {}
    }
}

impl MutationOperator for NoMutation {
    #[inline]
    fn mutate<'a>(&self, mut _individual: ArrayViewMut1<'a, f64>, _rng: &mut impl RandomGenerator) {
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
    fn test_no_mutation_leaves_individual_unchanged() {
        let mut population = array![[1.0, 2.0, 3.0]];
        let expected = population.clone();
        let mut rng = MOORandomGenerator::new(StdRng::seed_from_u64(42));
        let op = NoMutation::new();

        op.operate(&mut population, 1.0, &mut rng);

        assert_eq!(population, expected);
    }
}
