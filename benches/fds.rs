extern crate pymoors;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;

use pymoors::non_dominated_sorting::fast_non_dominated_sorting;

/// Generates random fitness data for the population with a fixed seed.
///
/// # Parameters
/// - `pop_size`: The number of individuals in the population.
/// - `n_obj`: The number of objectives (columns) in the fitness array.
/// - `seed`: A fixed seed for reproducibility.
///
/// # Returns
/// An `Array2<f64>` of shape `(pop_size, n_obj)` with random values in the range [0, 100).
fn generate_population_fitness(pop_size: usize, n_obj: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let data: Vec<f64> = (0..pop_size * n_obj)
        .map(|_| rng.gen_range(0.0..100.0))
        .collect();
    Array2::from_shape_vec((pop_size, n_obj), data)
        .expect("Error creating population fitness array")
}

/// Benchmark for the `fast_non_dominated_sorting` function.
fn bench_fast_non_dominated_sorting(c: &mut Criterion) {
    // Set parameters for the population.
    let pop_size = 10000;
    let n_obj = 2;
    let seed = 42; // Fixed seed for reproducibility.
    let population_fitness = generate_population_fitness(pop_size, n_obj, seed);

    c.bench_function("fast_non_dominated_sorting", |b| {
        b.iter(|| {
            // Use black_box to prevent the compiler from optimizing away computations.
            let fronts = fast_non_dominated_sorting(black_box(&population_fitness));
            black_box(fronts);
        })
    });
}

criterion_group!(benches, bench_fast_non_dominated_sorting);
criterion_main!(benches);
