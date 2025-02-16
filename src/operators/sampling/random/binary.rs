use std::fmt::Debug;

use pymoors_macros::py_operator;

use crate::genetic::Genes;
use crate::operators::{GeneticOperator, SamplingOperator};
use crate::random::RandomGenerator;

#[py_operator("sampling", "Sampling operator for binary variables.")]
#[derive(Clone, Debug)]
pub struct RandomSamplingBinary;

impl RandomSamplingBinary {
    pub fn new() -> Self {
        Self
    }
}

impl GeneticOperator for RandomSamplingBinary {
    fn name(&self) -> String {
        "RandomSamplingBinary".to_string()
    }
}

impl SamplingOperator for RandomSamplingBinary {
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> Genes {
        (0..n_vars)
            .map(|_| if rng.gen_bool(0.5) { 1.0 } else { 0.0 })
            .collect()
    }
}
