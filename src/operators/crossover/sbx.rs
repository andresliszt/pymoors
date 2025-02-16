use crate::genetic::Genes;
use crate::operators::{CrossoverOperator, GeneticOperator};
use crate::random::RandomGenerator;
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
    fn sbx_crossover(&self, y1: f64, y2: f64, rng: &mut dyn RandomGenerator) -> (f64, f64) {
        let (p1, p2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
        let rand_u = rng.gen_proability();

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
        rng: &mut dyn RandomGenerator,
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

impl_py_crossover!(
    "Simulated Binary Crossover (SBX) operator for real-coded genetic algorithms.",
    SimulatedBinaryCrossover,
    "SimulatedBinaryCrossover",
    distribution_index: f64,
);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::genetic::Genes;
    use crate::random::{RandomGenerator, TestDummyRng};
    use numpy::ndarray::array;

    /// A fake random generator for controlled testing of SBX.
    /// It only provides predetermined probability values via `gen_proability`.
    struct FakeRandom {
        /// Predefined probability values to be returned sequentially.
        probability_values: Vec<f64>,
        /// Dummy RNG to satisfy the trait requirement.
        dummy: TestDummyRng,
    }

    impl FakeRandom {
        /// Creates a new instance with the given probability responses.
        fn new(probability_values: Vec<f64>) -> Self {
            Self {
                probability_values,
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandom {
        fn rng(&mut self) -> &mut dyn rand::RngCore {
            &mut self.dummy
        }
        fn gen_range_usize(&mut self, _min: usize, _max: usize) -> usize {
            unimplemented!("Not used in SBX test")
        }
        fn gen_range_f64(&mut self, _min: f64, _max: f64) -> f64 {
            unimplemented!("Not used in SBX test")
        }
        fn gen_usize(&mut self) -> usize {
            unimplemented!("Not used in SBX test")
        }
        fn gen_bool(&mut self, _p: f64) -> bool {
            unimplemented!("Not used in SBX test")
        }
        fn gen_proability(&mut self) -> f64 {
            // Return the next predetermined probability value.
            self.probability_values.remove(0)
        }
    }

    #[test]
    fn test_simulated_binary_crossover() {
        // Define two parent genes as Genes (Array1<f64>).
        // Gene 0: p1 = 1.0, p2 = 3.0 => will undergo SBX.
        // Gene 1: p1 = 5.0, p2 = 5.0 => nearly equal, so no crossover.
        let parent_a: Genes = array![1.0, 5.0];
        let parent_b: Genes = array![3.0, 5.0];

        // Create the SBX operator with distribution_index = 2.0.
        let operator = SimulatedBinaryCrossover::new(2.0);
        assert_eq!(
            operator.name(),
            "SimulatedBinaryCrossover(distribution_index=2)"
        );

        // Set up a fake random generator:
        // For gene 0, we force rand_u = 0.25.
        // (For gene 1, no random value is needed because the parents are identical.)
        let mut fake_rng = FakeRandom::new(vec![0.25]);

        // Perform the crossover.
        let (child_a, child_b) = operator.crossover(&parent_a, &parent_b, &mut fake_rng);

        // For gene 0:
        // p1 = 1.0 and p2 = 3.0, distribution_index = 2.0, and rand_u = 0.25.
        // Compute beta_q:
        //   beta_q = (2 * 0.25)^(1/(2.0+1)) = 0.5^(1/3) ≈ 0.7937005259.
        // Offspring:
        //   c1 = 0.5 * ((1.0 + 3.0) - beta_q * (3.0 - 1.0))
        //      = 0.5 * (4.0 - 0.7937005259 * 2)
        //      ≈ 0.5 * (4.0 - 1.5874010518)
        //      ≈ 0.5 * 2.4125989482 ≈ 1.2062994741.
        //   c2 = 0.5 * ((1.0 + 3.0) + beta_q * (3.0 - 1.0))
        //      = 0.5 * (4.0 + 1.5874010518)
        //      ≈ 0.5 * 5.5874010518 ≈ 2.7937005259.
        //
        // Since 1.0 < 3.0, sbx_crossover returns (c1, c2).
        //
        // For gene 1:
        // The parents are identical, so no crossover is performed.
        // Offspring simply copy the parent's value: 5.0.
        //
        // Therefore, expected children:
        // child_a: [1.2062994741, 5.0]
        // child_b: [2.7937005259, 5.0]
        let tol = 1e-8;
        assert!(
            (child_a[0] - 1.2062994741).abs() < tol,
            "Gene 0 of child_a not as expected: {}",
            child_a[0]
        );
        assert!(
            (child_b[0] - 2.7937005259).abs() < tol,
            "Gene 0 of child_b not as expected: {}",
            child_b[0]
        );
        assert!(
            (child_a[1] - 5.0).abs() < tol,
            "Gene 1 of child_a not as expected: {}",
            child_a[1]
        );
        assert!(
            (child_b[1] - 5.0).abs() < tol,
            "Gene 1 of child_b not as expected: {}",
            child_b[1]
        );
    }
}
