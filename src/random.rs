use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::thread_rng;
use rand::RngCore;

pub fn get_rng(seed: Option<u64>) -> Box<dyn RngCore> {
    match seed {
        Some(state) => Box::new(ChaCha8Rng::seed_from_u64(state)),
        None => Box::new(thread_rng())
    }
}
