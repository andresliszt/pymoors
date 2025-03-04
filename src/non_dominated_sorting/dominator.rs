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
/// The individuals are grouped into fronts in order of non-dominance.
/// If during the construction of fronts the cumulative count of individuals reaches or exceeds
/// `min_survivors`, the entire last front is included (even if it causes the total to exceed `min_survivors`)
/// and no further fronts are added.
pub fn fast_non_dominated_sorting(
    population_fitness: &PopulationFitness,
    min_survivors: usize,
) -> Vec<Vec<usize>> {
    let population_size = population_fitness.shape()[0];

    // Thread-safe data structures
    let domination_count = (0..population_size)
        .map(|_| AtomicUsize::new(0))
        .collect::<Vec<_>>();
    let dominated_sets = (0..population_size)
        .map(|_| Mutex::new(Vec::new()))
        .collect::<Vec<_>>();

    // Precompute row views to avoid repeated indexing
    let fitness_rows: Vec<ArrayView1<f64>> = (0..population_size)
        .map(|i| population_fitness.index_axis(Axis(0), i))
        .collect();

    // Parallel pairwise comparisons: for each pair (p, q) with p < q, each thread updates local data
    (0..population_size).into_par_iter().for_each(|p| {
        // Accumulate changes locally to reduce locking overhead
        let mut local_updates = Vec::new();

        for q in (p + 1)..population_size {
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

        // Apply local updates to shared data:
        // For each (dominator, dominated) pair:
        for (dominator, dominated) in local_updates {
            {
                // Push dominated into the dominator's list
                let mut lock = dominated_sets[dominator].lock().unwrap();
                lock.push(dominated);
            }
            // Increment the atomic domination_count of the dominated individual
            domination_count[dominated].fetch_add(1, Ordering::Relaxed);
        }
    });

    // Convert to a normal Vec<Vec<usize>>
    let dominated_sets_vec: Vec<Vec<usize>> = dominated_sets
        .into_iter()
        .map(|m| m.into_inner().unwrap())
        .collect();

    // Build the first front
    let mut fronts = Vec::new();
    let mut first_front = Vec::new();
    for i in 0..population_size {
        if domination_count[i].load(Ordering::Relaxed) == 0 {
            first_front.push(i);
        }
    }
    fronts.push(first_front.clone());
    let mut count = first_front.len();
    // If the first front already reaches min_survivors, return immediately.
    if count >= min_survivors {
        return fronts;
    }

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
        // If adding the next front reaches or exceeds min_survivors,
        // include the entire front and stop the construction.
        if count + next_front.len() >= min_survivors {
            fronts.push(next_front);
            break;
        } else {
            count += next_front.len();
            fronts.push(next_front.clone());
            current_front = next_front;
        }
    }

    fronts
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Array2};

    #[test]
    fn test_dominates() {
        // Test case 1: The first vector dominates the second
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        assert_eq!(dominates(&a.view(), &b.view()), true);

        // Test case 2: The second vector dominates the first
        let a = array![3.0, 3.0, 3.0];
        let b = array![2.0, 4.0, 5.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 3: Neither vector dominates the other
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 1.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);

        // Test case 4: Equal vectors
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(dominates(&a.view(), &b.view()), false);
    }

    #[test]
    fn test_fast_non_dominated_sorting() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1.0, 2.0], // Individual 0
            [2.0, 1.0], // Individual 1
            [1.5, 1.5], // Individual 2
            [3.0, 4.0], // Individual 3 (dominated by everyone)
            [4.0, 3.0]  // Individual 4 (dominated by everyone)
        ];

        // Perform fast non-dominated sorting with min_survivors = 5
        let fronts = fast_non_dominated_sorting(&population_fitness, 5);

        // Expected Pareto fronts:
        // Front 1: Individuals 0, 1, 2
        // Front 2: Individuals 3, 4 (the entire front is included when min_survivors is reached)
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
            [1.0, 2.0], // Individual 0
            [2.0, 1.0], // Individual 1
            [1.5, 1.5], // Individual 2
        ];

        // Perform fast non-dominated sorting with min_survivors = 3
        let fronts = fast_non_dominated_sorting(&population_fitness, 3);

        // Expected Pareto front: All individuals belong to the same front.
        // The front is returned in its entirety when min_survivors is reached.
        let expected_fronts = vec![
            vec![0, 1, 2], // Front 1
        ];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_empty_population() {
        // Define an empty population
        let population_fitness: Array2<f64> = Array2::zeros((0, 0));

        // Perform fast non-dominated sorting with min_survivors = 0
        let fronts = fast_non_dominated_sorting(&population_fitness, 0);

        // Expected: No fronts
        let expected_fronts: Vec<Vec<usize>> = vec![vec![]];

        assert_eq!(fronts, expected_fronts);
    }

    #[test]
    fn test_fast_non_dominated_sorting_n_survive_cut() {
        // Define a population with clear dominance relationships and duplicate fitness values
        // to force multiple individuals in a front.
        let population_fitness = array![
            [1.0, 1.0], // Individual 0: best, not dominated by anyone
            [2.0, 2.0], // Individual 1: dominated by 0
            [2.0, 2.0], // Individual 2: duplicate of 1, same front as 1
            [3.0, 3.0], // Individual 3: dominated by 0,1,2
            [4.0, 4.0]  // Individual 4: dominated by 0,1,2,3
        ];

        // Set min_survivors = 2. The first front is [0] (1 individual).
        // The next front is [1, 2] (2 individuals). Adding this front reaches min_survivors (1 + 2 > 2)
        // so the algorithm should include the entire second front and stop.
        let fronts = fast_non_dominated_sorting(&population_fitness, 2);
        let expected_fronts = vec![
            vec![0],    // Front 1
            vec![1, 2], // Front 2 (included entirely when min_survivors is reached)
        ];

        assert_eq!(fronts, expected_fronts);
    }
}
