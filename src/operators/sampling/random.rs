use rand::distributions::Distribution;
use rand::Rng;
use crate::genetic::Genes;
use crate::operators::{GeneticOperator, SamplingOperator};
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct RandomSamplingFloat {
    pub min: f64,
    pub max: f64,
}

impl GeneticOperator for RandomSamplingFloat {
    fn name(&self) -> String {
        "RandomSamplingFloat".to_string()
    }
}

impl SamplingOperator for RandomSamplingFloat {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes
    where
        R: Rng + Sized,
    {
        let num_genes = 10; // Example number of genes
        (0..num_genes)
            .map(|_| rng.gen_range(self.min..self.max))
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingInt {
    pub min: i32,
    pub max: i32,
}

impl GeneticOperator for RandomSamplingInt {
    fn name(&self) -> String {
        "RandomSamplingInt".to_string()
    }
}

impl SamplingOperator for RandomSamplingInt {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes
    where
        R: Rng + Sized,
    {
        let num_genes = 10; // Example number of genes
        (0..num_genes)
            .map(|_| rng.gen_range(self.min..self.max) as f64)
            .collect()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingBinary;

impl GeneticOperator for RandomSamplingBinary {
    fn name(&self) -> String {
        "RandomSamplingBinary".to_string()
    }
}

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes
    where
        R: Rng + Sized,
    {
        let num_genes = 10; // Example number of genes
        (0..num_genes)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_sampling_float() {
        let sampler = RandomSamplingFloat { min: -1.0, max: 1.0 };
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 10);
        for &gene in population.iter() {
            assert!(gene >= -1.0 && gene < 1.0);
        }
    }

    #[test]
    fn test_random_sampling_int() {
        let sampler = RandomSamplingInt { min: 0, max: 10 };
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 10);
        for &gene in population.iter() {
            assert!(gene >= 0.0 && gene < 10.0);
        }
    }

    #[test]
    fn test_random_sampling_binary() {
        let sampler = RandomSamplingBinary;
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 10);
        for &gene in population.iter() {
            assert!(gene == 0.0 || gene == 1.0);
        }
    }
}
