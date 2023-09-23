use crate::genetic::PopulationFitness;
use numpy::ndarray::{ArrayView1, Axis};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Inlines the check for "does f1 dominate f2?" to reduce call overhead.
#[inline]
fn dominates(f1: &ArrayView1<f64>, f2: &ArrayView1<f64>) -> bool {
    let mut better = false;
    // We assume f1.len() == f2.len()
    for (&a, &b) in f1.iter().zip(f2.iter()) {
        if a > b {
            return false;
        } else if a < b {
            better = true;
        }
    }
    better
}

/// Parallel Fast Non-Dominated Sorting.
/// Returns a vector of fronts, each front is a list of indices.
pub fn fast_non_dominated_sorting(population_fitness: &PopulationFitness) -> Vec<Vec<usize>> {
    let pop_size = population_fitness.shape()[0];

    // Thread-safe data structures
    let domination_count = (0..pop_size)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>();
    let dominated_sets = (0..pop_size)
        .map(|_| Mutex::new(Vec::new()))
        .collect::<Vec<_>>();

    // Precompute row views to avoid repeated indexing
    let fitness_rows: Vec<ArrayView1<f64>> = (0..pop_size)
        .map(|i| population_fitness.index_axis(Axis(0), i))
        .collect();

    // Parallel pairwise comparisons: p < q, each thread updates local data
    (0..pop_size).into_par_iter().for_each(|p| {
        // We'll accumulate changes locally to reduce locking overhead
        let mut local_updates = Vec::new();

        for q in (p + 1)..pop_size {
            let p_dominates_q = dominates(&fitness_rows[p], &fitness_rows[q]);
            let q_dominates_p = dominates(&fitness_rows[q], &fitness_rows[p]);

            if p_dominates_q {
                // p dominates q
                local_updates.push((p, q));
            } else if q_dominates_p {
                // q dominates p
                local_updates.push((q, p));
            }
            // else -> neither dominates
        }

        // Apply local updates to shared data
        // For each (dominator, dominated) pair:
        for (dominator, dominated) in local_updates {
            // push dominated into dominator's list
            {
                let mut lock = dominated_sets[dominator].lock().unwrap();
                lock.push(dominated);
            }
            // increment atomic domination_count of "dominated"
            domination_count[dominated].fetch_add(1, Ordering::Relaxed);
        }
    });

    // Convert to normal Vec<Vec<usize>>
    let mut dominated_sets_vec: Vec<Vec<usize>> = dominated_sets
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect();

    // Build first front
    let mut fronts = Vec::new();
    let mut first_front = Vec::new();
    for i in 0..pop_size {
        if domination_count[i].load(Ordering::Relaxed) == 0 {
            first_front.push(i);
        }
    }
    fronts.push(first_front.clone());

    // Construct subsequent fronts
    let mut current_front = first_front;
    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &p in &current_front {
            for &q in &dominated_sets_vec[p] {
                let old_count = domination_count[q].fetch_sub(1, Ordering::Relaxed);
                if old_count == 1 {
                    // now it's zero
                    next_front.push(q);
                }
            }
        }
        if next_front.is_empty() {
            break;
        }
        fronts.push(next_front.clone());
        current_front = next_front;
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Array2};

    #[test]
    fn test_dominates() {
        // Test case 1: First vector _dominates the second
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        assert_eq!(dominates(&a.view(), &b.view()), true);

        // Test case 2: Second vector _dominates the first
        let a = array![3.0, 3.0, 3.0];
        let b = array![2.0, 4.0, 5.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 3: Neither vector _dominates the other
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 1.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 4: Equal vectors
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);
    }

    // #[test]
    // fn test_get_current_front() {
    //     // Define the fitness values of the population
    //     let population_fitness = array![
    //         [1.0, 2.0], // Genes 0
    //         [2.0, 1.0], // Genes 1
    //         [1.5, 1.5], // Genes 2
    //         [3.0, 4.0], // Genes 3 (dominated by everyone)
    //     ];

    //     // All individuals are initially considered
    //     let remainder_indexes = vec![0, 1, 2, 3];

    //     // Compute the current Pareto front
    //     let current_front = _get_current_front(&population_fitness, &remainder_indexes);

    //     // Expected front: individuals 0, 1, and 2 (not dominated by anyone in this set)
    //     let expected_front = vec![0, 1, 2];

    //     assert_eq!(current_front, expected_front);
    // }

    // #[test]
    // fn test_get_current_front_partial_population() {
    //     // Define the fitness values of the population
    //     let population_fitness = array![
    //         [1.0, 2.0], // Genes 0
    //         [2.0, 1.0], // Genes 1
    //         [1.5, 1.5], // Genes 2
    //         [3.0, 4.0], // Genes 3 (dominated by everyone)
    //     ];

    //     // Consider only a subset of individuals (partial population)
    //     let remainder_indexes = vec![1, 2, 3];

    //     // Compute the current Pareto front
    //     let current_front = _get_current_front(&population_fitness, &remainder_indexes);

    //     // Expected front: individual   s 1 and 2 (within the subset)
    //     let expected_front = vec![1, 2];

    //     assert_eq!(current_front, expected_front);
    // }

    #[test]
    fn test_fast_non_dominated_sorting() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1.0, 2.0], // Genes 0
            [2.0, 1.0], // Genes 1
            [1.5, 1.5], // Genes 2
            [3.0, 4.0], // Genes 3 (dominated by everyone)
            [4.0, 3.0]  // Genes 4 (dominated by everyone)
        ];

        // Perform fast non-dominated sorting
        let fronts = fast_non_dominated_sorting(&population_fitness);

        // Expected Pareto fronts:
        // Front 1: Individuals 0, 1, 2
        // Front 2: Individuals 3, 4
        let expected_fronts = vec![
            vec![0, 1, 2], // Front 1
            vec![3, 4],    // Front 2
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_single_front() {
        // Define a population where no individual dominates another
        let population_fitness = array![
            [1.0, 2.0], // Genes 0
            [2.0, 1.0], // Genes 1
            [1.5, 1.5], // Genes 2
        ];

        // Perform fast non-dominated sorting
        let fronts = fast_non_dominated_sorting(&population_fitness);

        // Expected Pareto front: All individuals belong to the same front
        let expected_fronts = vec![
            vec![0, 1, 2], // All individuals in Front 1
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_empty_population() {
        // Define an empty population
        let population_fitness: Array2<f64> = Array2::zeros((0, 0));

        // Perform fast non-dominated sorting
        let fronts = fast_non_dominated_sorting(&population_fitness);

        // Expected: No fronts
        let expected_fronts: Vec<Vec<usize>> = vec![];

        assert_eq!(fronts, expected_fronts);
    }
}
