use ndarray::Array1;
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use crate::genetic::Genes;
use crate::operators::{GeneticOperator, SamplingOperator};

#[derive(Clone, Debug)]
pub struct RandomSamplingFloat {
    dimensions: usize,
    lower_bound: f64,
    upper_bound: f64,
}

impl RandomSamplingFloat {
    pub fn new(dimensions: usize, lower_bound: f64, upper_bound: f64) -> Self {
        Self {
            dimensions,
            lower_bound,
            upper_bound,
        }
    }
}

impl GeneticOperator for RandomSamplingFloat {
    fn name(&self) -> String {
        "RandomSamplingFloat".to_string()
    }
}

impl SamplingOperator<f64> for RandomSamplingFloat {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes<f64>
    where
        R: Rng + Sized,
    {
        let uniform = Uniform::new(self.lower_bound, self.upper_bound);
        (0..self.dimensions)
            .map(|_| uniform.sample(rng))
            .collect::<Array1<f64>>()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingInt {
    dimensions: usize,
    lower_bound: i32,
    upper_bound: i32,
}

impl RandomSamplingInt {
    pub fn new(dimensions: usize, lower_bound: i32, upper_bound: i32) -> Self {
        Self {
            dimensions,
            lower_bound,
            upper_bound,
        }
    }
}

impl GeneticOperator for RandomSamplingInt {
    fn name(&self) -> String {
        "RandomSamplingInt".to_string()
    }
}

impl SamplingOperator<i32> for RandomSamplingInt {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes<i32>
    where
        R: Rng + Sized,
    {
        let uniform = Uniform::new(self.lower_bound, self.upper_bound);
        (0..self.dimensions)
            .map(|_| uniform.sample(rng))
            .collect::<Array1<i32>>()
    }
}

#[derive(Clone, Debug)]
pub struct RandomSamplingBinary {
    dimensions: usize,
}

impl RandomSamplingBinary {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl GeneticOperator for RandomSamplingBinary {
    fn name(&self) -> String {
        "RandomSamplingBinary".to_string()
    }
}

impl SamplingOperator<u8> for RandomSamplingBinary {
    fn sample_individual<R>(&self, rng: &mut R) -> Genes<u8>
    where
        R: Rng + Sized,
    {
        let uniform = Uniform::new(0, 2); // Binary values: 0 or 1
        (0..self.dimensions)
            .map(|_| uniform.sample(rng) as u8)
            .collect::<Array1<u8>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_sampling_float() {
        let sampler = RandomSamplingFloat::new(5, -1.0, 1.0);
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene >= -1.0 && gene < 1.0);
        }
    }

    #[test]
    fn test_random_sampling_int() {
        let sampler = RandomSamplingInt::new(5, 0, 10);
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene >= 0 && gene < 10);
        }
    }

    #[test]
    fn test_random_sampling_binary() {
        let sampler = RandomSamplingBinary::new(5);
        let mut rng = StdRng::from_seed([0; 32]);
        let population = sampler.operate(10, &mut rng);

        assert_eq!(population.nrows(), 10);
        assert_eq!(population.ncols(), 5);
        for &gene in population.iter() {
            assert!(gene == 0 || gene == 1);
        }
    }
}
