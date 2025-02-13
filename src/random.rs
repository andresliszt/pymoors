use rand::rngs::StdRng;
use rand::{Rng, RngCore};

/// A trait defining a unified interface for generating random values,
/// used across genetic operators and algorithms.
pub trait RandomGenerator {
    /// Generates a random `usize` in the range `[min, max)`.
    fn gen_range_usize(&mut self, min: usize, max: usize) -> usize;

    /// Generates a random `f64` in the range `[min, max)`.
    fn gen_range_f64(&mut self, min: f64, max: f64) -> f64;

    /// Generates a random `usize` with no specified range.
    fn gen_usize(&mut self) -> usize;

    /// Generates a random boolean value with `p` probability of being `true`.
    fn gen_bool(&mut self, p: f64) -> bool;

    /// Returns a mutable reference to the underlying RNG implementing `RngCore`.
    fn rng(&mut self) -> &mut dyn RngCore;

    /// Generates a random probability as an `f64` in the range `[0.0, 1.0)`.
    fn gen_proability(&mut self) -> f64 {
        self.rng().gen::<f64>()
    }
}

/// The production implementation of `RandomGenerator` using `StdRng`.
pub struct MOORandomGenerator {
    rng: StdRng,
}

impl MOORandomGenerator {
    /// Creates a new `MOORandomGenerator` with the provided `StdRng`.
    pub fn new(rng: StdRng) -> Self {
        Self { rng }
    }
}

impl RandomGenerator for MOORandomGenerator {
    /// Returns a mutable reference to the underlying `StdRng`.
    fn rng(&mut self) -> &mut dyn RngCore {
        &mut self.rng
    }

    /// Generates a random `usize` in the range `[min, max)` using the underlying RNG.
    fn gen_range_usize(&mut self, min: usize, max: usize) -> usize {
        self.rng.gen_range(min..max)
    }

    /// Generates a random `f64` in the range `[min, max)` using the underlying RNG.
    fn gen_range_f64(&mut self, min: f64, max: f64) -> f64 {
        self.rng.gen_range(min..max)
    }

    /// Generates a random `usize` using the underlying RNG.
    fn gen_usize(&mut self) -> usize {
        self.rng.gen()
    }

    /// Generates a random boolean value with probability `p` of being `true`
    /// using the underlying RNG.
    fn gen_bool(&mut self, p: f64) -> bool {
        self.rng.gen_bool(p)
    }
}

/// A dummy implementation of `RngCore` for testing purposes.
/// This struct is used when methods are called via the `RandomGenerator` trait
/// without directly interacting with self.rng. This is for testing only, see several
/// examples in the operators module
pub struct TestDummyRng;

impl RngCore for TestDummyRng {
    /// Not used in tests. This method is unimplemented.
    fn next_u32(&mut self) -> u32 {
        unimplemented!("Not used in this test")
    }

    /// Not used in tests. This method is unimplemented.
    fn next_u64(&mut self) -> u64 {
        unimplemented!("Not used in this test")
    }

    /// Not used in tests. This method is unimplemented.
    fn fill_bytes(&mut self, _dest: &mut [u8]) {
        unimplemented!("Not used in this test")
    }

    /// Not used in tests. This method is unimplemented.
    fn try_fill_bytes(&mut self, _dest: &mut [u8]) -> Result<(), rand::Error> {
        unimplemented!("Not used in this test")
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_gen_range_usize() {
        // Create a MOORandomGenerator with a fixed seed.
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        let min = 10;
        let max = 20;
        let value = rng.gen_range_usize(min, max);

        // Check that the generated value is within [min, max).
        assert!(
            value >= min && value < max,
            "gen_range_usize produced {} which is not in [{}, {})",
            value,
            min,
            max
        );
    }

    #[test]
    fn test_gen_range_f64() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        let min = 3.0;
        let max = 10.0;
        let value = rng.gen_range_f64(min, max);

        // Check that the generated value is within [min, max).
        assert!(
            value >= min && value < max,
            "gen_range_f64 produced {} which is not in [{}, {})",
            value,
            min,
            max
        );
    }

    #[test]
    fn test_gen_bool() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        // With a probability of 1.0, it should always return true.
        assert!(rng.gen_bool(1.0), "gen_bool(1.0) did not return true");

        // With a probability of 0.0, it should always return false.
        assert!(!rng.gen_bool(0.0), "gen_bool(0.0) did not return false");
    }

    #[test]
    fn test_gen_probability() {
        let seed = [42u8; 32];
        let mut rng = MOORandomGenerator::new(StdRng::from_seed(seed));

        // gen_proability (note the method name) returns a f64 in the range [0, 1).
        let prob = rng.gen_proability();
        assert!(
            prob >= 0.0 && prob < 1.0,
            "gen_proability produced {} which is not in [0, 1)",
            prob
        );
    }
}
