use pyo3::prelude::*;
use rand::{Rng, RngCore};

use crate::genetic::Genes;
use crate::operators::{CrossoverOperator, GeneticOperator};

/// Simulated Binary Crossover (SBX) operator for real-coded genetic algorithms.
///
/// # SBX Overview
///
/// - SBX mimics the behavior of single-point crossover in binary-coded GAs,
///   but for continuous variables.
/// - The parameter `distribution_index` (often called "eta") controls how far
///   offspring can deviate from the parents. Larger eta values produce offspring
///   closer to the parents (less exploration), while smaller values allow
///   offspring to be more spread out (more exploration).
///
/// **Reference**: Deb, Kalyanmoy, and R. B. Agrawal. "Simulated binary crossover
/// for continuous search space." Complex systems 9.2 (1995): 115-148.
#[derive(Clone, Debug)]
pub struct SimulatedBinaryCrossover {
    /// Distribution index (η) that controls offspring spread. Typical range: [2, 20].
    pub distribution_index: f64,
}

impl SimulatedBinaryCrossover {
    /// Create a new `SimulatedBinaryCrossover` operator with the given distribution index.
    pub fn new(distribution_index: f64) -> Self {
        Self { distribution_index }
    }

    /// Perform SBX on a pair of floating-point values (y1, y2).
    /// Returns two offspring (o1, o2).
    ///
    /// This function is private because it's only intended to be used
    /// internally within the operator's main crossover method.
    fn sbx_crossover(&self, y1: f64, y2: f64, rng: &mut dyn RngCore) -> (f64, f64) {
        let (p1, p2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
        let rand_u = rng.gen::<f64>();

        // Compute beta_q according to Deb & Agrawal (1995)
        let beta_q = if rand_u <= 0.5 {
            (2.0 * rand_u).powf(1.0 / (self.distribution_index + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - rand_u))).powf(1.0 / (self.distribution_index + 1.0))
        };

        // Generate offspring
        let c1 = 0.5 * ((p1 + p2) - beta_q * (p2 - p1));
        let c2 = 0.5 * ((p1 + p2) + beta_q * (p2 - p1));

        // Reassign offspring to preserve the original order (y1, y2)
        if y1 < y2 {
            (c1, c2)
        } else {
            (c2, c1)
        }
    }
}

impl GeneticOperator for SimulatedBinaryCrossover {
    fn name(&self) -> String {
        format!(
            "SimulatedBinaryCrossover(distribution_index={})",
            self.distribution_index
        )
    }
}

impl CrossoverOperator for SimulatedBinaryCrossover {
    fn crossover(
        &self,
        parent_a: &Genes,
        parent_b: &Genes,
        rng: &mut dyn RngCore,
    ) -> (Genes, Genes) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len());

        let mut child_a = parent_a.clone();
        let mut child_b = parent_b.clone();

        for i in 0..len {
            let p1 = parent_a[i];
            let p2 = parent_b[i];

            // If the two parent genes are nearly identical, just copy
            if (p1 - p2).abs() < 1e-14 {
                child_a[i] = p1;
                child_b[i] = p2;
            } else {
                // Always do SBX here (no internal crossover_rate)
                let (c1, c2) = self.sbx_crossover(p1, p2, rng);
                child_a[i] = c1;
                child_b[i] = c2;
            }
        }

        (child_a, child_b)
    }
}

#[pyclass(name = "SimulatedBinaryCrossover")]
#[derive(Clone)]
pub struct PySimulatedBinaryCrossover {
    pub inner: SimulatedBinaryCrossover,
}

#[pymethods]
impl PySimulatedBinaryCrossover {
    /// Create a new Python-exposed `SimulatedBinaryCrossover` with the given distribution index.
    #[new]
    fn new(distribution_index: f64) -> Self {
        Self {
            inner: SimulatedBinaryCrossover::new(distribution_index),
        }
    }
}
