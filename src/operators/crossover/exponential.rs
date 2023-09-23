use pyo3::prelude::*;
use rand::{Rng, RngCore};

use crate::genetic::Genes;
use crate::operators::{CrossoverOperator, GeneticOperator};

#[derive(Clone, Debug)]
pub struct ExponentialCrossover {
    pub exponential_crossover_rate: f64,
}

impl ExponentialCrossover {
    pub fn new(exponential_crossover_rate: f64) -> Self {
        Self {
            exponential_crossover_rate,
        }
    }
}

impl GeneticOperator for ExponentialCrossover {
    fn name(&self) -> String {
        format!(
            "ExponentialCrossover(exponential_crossover_rate={})",
            self.exponential_crossover_rate
        )
    }
}

impl CrossoverOperator for ExponentialCrossover {
    fn crossover(
        &self,
        parent_a: &Genes,
        parent_b: &Genes,
        rng: &mut dyn RngCore,
    ) -> (Genes, Genes) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len());

        // We'll do "child_a" from (A,B) and "child_b" from (B,A).
        let mut child_a = parent_a.clone();
        let mut child_b = parent_b.clone();

        // random start [0..len)
        let start = rng.gen_range(0..len);

        // We'll copy from parent_b into child_a as long as rng < CR
        // in a circular manner
        let mut i = start;
        loop {
            child_a[i] = parent_b[i];
            // Next index (wrap around)
            i = (i + 1) % len;
            if i == start {
                break; // completed a full circle, stop
            }
            let r: f64 = rng.gen();
            if r >= self.exponential_crossover_rate {
                break;
            }
        }

        // Similarly for child_b, but swap roles of A/B
        let start_b = rng.gen_range(0..len);
        let mut j = start_b;
        loop {
            child_b[j] = parent_a[j];
            j = (j + 1) % len;
            if j == start_b {
                break;
            }
            let r: f64 = rng.gen();
            if r >= self.exponential_crossover_rate {
                break;
            }
        }

        (child_a, child_b)
    }
}

#[pyclass(name = "ExponentialCrossover")]
#[derive(Clone)]
pub struct PyExponentialCrossover {
    pub inner: ExponentialCrossover,
}

#[pymethods]
impl PyExponentialCrossover {
    #[new]
    fn new(exponential_crossover_rate: f64) -> Self {
        Self {
            inner: ExponentialCrossover::new(exponential_crossover_rate),
        }
    }
}
