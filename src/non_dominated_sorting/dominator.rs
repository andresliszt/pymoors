use std::collections::HashSet;

use ndarray::{ArrayView1, ArrayView2, Axis};

use crate::genetic::{FitnessValue, PopulationFitness};

/// Determines if one fitness vector _dominates another in multi-objective optimization.
///
/// # Parameters
///
/// - `a`: Reference to the first fitness vector.
/// - `b`: Reference to the second fitness vector.
/// - `epsilon`: Optional epsilon value for epsilon-dominance. If `None`, zero is used.
///
/// # Returns
///
/// - `true` if `a` _dominates `b`.
/// - `false` otherwise.
///
/// # Type Constraints
///
/// - `F`: Must implement `PartialOrd`, `Copy`, `Add<Output = F>`, and `Zero`.
///
/// # Panics
///
/// - Panics if the lengths of the fitness vectors `a` and `b` are not equal.
///
/// # Example
///
/// ```rust
/// use ndarray::array;
/// // Assume the `_dominates` function is defined in the current scope.
///
/// let a = array![1.0, 2.0, 3.0];
/// let b = array![1.5, 1.5, 3.0];
///
/// assert_eq!(_dominates(&a, &b, None), false);
/// assert_eq!(_dominates(&b, &a, None), true);
/// ```
fn _dominates<F>(f1: &ArrayView1<F>, f2: &ArrayView1<F>) -> bool
where
    F: FitnessValue,
{
    let mut better_in_any = false;

    for (&f1_i, &f2_i) in f1.iter().zip(f2.iter()) {
        if f1_i > f2_i {
            // 'a' is worse than 'b' in this objective
            return false;
        } else if f1_i < f2_i {
            // 'a' is better than 'b' in this objective
            better_in_any = true;
        }
        // If neither condition holds, 'a_i' and 'b_i' are approximately equal
    }

    better_in_any
}

fn _get_current_front<F>(
    population_fitness: ArrayView2<'_, F>,
    remainder_indexes: &Vec<usize>,
) -> Vec<usize>
where
    F: FitnessValue,
{
    // Filter population fitness based on remainder_indexes
    let filtered_population = population_fitness.select(Axis(0), remainder_indexes);

    let population_size = filtered_population.shape()[0];

    // Create an empty vector for the current Pareto Front
    let mut current_front: Vec<usize> = Vec::new();

    // Create a vector to keep track of the domination count for each individual
    let mut domination_count: Vec<usize> = vec![0; population_size];

    // Create a list of individuals that each individual dominates
    let mut dominated_individuals: Vec<Vec<usize>> = vec![Vec::new(); population_size];

    for i in 0..population_size {
        for j in (i + 1)..population_size {
            let p = filtered_population.row(i);
            let q = filtered_population.row(j);
            if _dominates(&p, &q) {
                // Genes i dominates individual j
                dominated_individuals[i].push(j);
                domination_count[j] += 1;
            } else if _dominates(&q, &p) {
                // Genes j dominates individual i
                dominated_individuals[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // Individuals with domination count zero belong to the current front
    for i in 0..population_size {
        if domination_count[i] == 0 {
            current_front.push(remainder_indexes[i]);
        }
    }

    current_front
}

fn _update_remainder_individuals<'a, 'b>(
    remainder_individuals: &'a mut Vec<usize>,
    current_front: &'b Vec<usize>,
) -> &'a mut Vec<usize> {
    let current_front_set: HashSet<usize> = current_front.iter().cloned().collect();
    remainder_individuals.retain(|x| !current_front_set.contains(x));
    remainder_individuals
}

pub fn fast_non_dominated_sorting<F>(population_fitness: &PopulationFitness<F>) -> Vec<Vec<usize>>
where
    F: FitnessValue,
{
    // Get population size
    let population_size: usize = population_fitness.shape()[0];
    // Container for the fronts
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    // Initial list of individual indices
    let mut remainder_indexes: Vec<usize> = (0..population_size).collect();

    // Generate fronts iteratively
    while !remainder_indexes.is_empty() {
        // Get the current front using the helper function
        let current_front: Vec<usize> = _get_current_front(
            population_fitness
                .select(Axis(0), &remainder_indexes)
                .view(),
            &remainder_indexes,
        );

        // Add the current front to the list of fronts
        fronts.push(current_front.clone());

        // Update the remainder individuals by removing the current front
        _update_remainder_individuals(&mut remainder_indexes, &current_front);
    }

    fronts
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn test_dominates_f64() {
        // Test case 1: First vector _dominates the second
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        assert_eq!(_dominates(&a.view(), &b.view()), true);

        // Test case 2: Second vector _dominates the first
        let a = array![3.0, 3.0, 3.0];
        let b = array![2.0, 4.0, 5.0];
        assert_eq!(_dominates(&a.view(), &b.view()), false);

        // Test case 3: Neither vector _dominates the other
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 1.0, 3.0];
        assert_eq!(_dominates(&a.view(), &b.view()), false);

        // Test case 4: Equal vectors
        let a = array![1.0, 2.0, 3.0];
        let b = array![1.0, 2.0, 3.0];
        assert_eq!(_dominates(&a.view(), &b.view()), false);
    }

    #[test]
    fn test_dominates_i32() {
        // Test case 1: First vector _dominates the second
        let a = array![1, 2, 3];
        let b = array![2, 3, 4];
        assert_eq!(_dominates(&a.view(), &b.view()), true);

        // Test case 2: Second vector _dominates the first
        let a = array![3, 3, 3];
        let b = array![2, 4, 5];
        assert_eq!(_dominates(&a.view(), &b.view()), false);

        // Test case 3: Neither vector _dominates the other
        let a = array![1, 2, 3];
        let b = array![2, 1, 3];
        assert_eq!(_dominates(&a.view(), &b.view()), false);

        // Test case 4: Equal vectors
        let a = array![1, 2, 3];
        let b = array![1, 2, 3];
        assert_eq!(_dominates(&a.view(), &b.view()), false);
    }

    #[test]
    fn test_get_current_front_f64() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1.0, 2.0], // Genes 0
            [2.0, 1.0], // Genes 1
            [1.5, 1.5], // Genes 2
            [3.0, 4.0], // Genes 3 (dominated by everyone)
        ];

        // All individuals are initially considered
        let remainder_indexes = vec![0, 1, 2, 3];

        // Compute the current Pareto front
        let current_front = _get_current_front(population_fitness.view(), &remainder_indexes);

        // Expected front: individuals 0, 1, and 2 (not dominated by anyone in this set)
        let expected_front = vec![0, 1, 2];

        assert_eq!(current_front, expected_front);
    }

    #[test]
    fn test_get_current_front_i32() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1, 2], // Genes 0
            [2, 1], // Genes 1
            [1, 1], // Genes 2 (dominated by no one)
            [3, 4], // Genes 3 (dominated by everyone)
        ];

        // All individuals are initially considered
        let remainder_indexes = vec![0, 1, 2, 3];

        // Compute the current Pareto front
        let current_front = _get_current_front(population_fitness.view(), &remainder_indexes);

        // Expected front: individual 2 dominates everyone
        let expected_front = vec![2];

        assert_eq!(current_front, expected_front);
    }

    #[test]
    fn test_get_current_front_partial_population() {
        // Define the fitness values of the population
        let population_fitness = array![
            [1.0, 2.0], // Genes 0
            [2.0, 1.0], // Genes 1
            [1.5, 1.5], // Genes 2
            [3.0, 4.0], // Genes 3 (dominated by everyone)
        ];

        // Consider only a subset of individuals (partial population)
        let remainder_indexes = vec![1, 2, 3];

        // Compute the current Pareto front
        let current_front = _get_current_front(population_fitness.view(), &remainder_indexes);

        // Expected front: individuals 1 and 2 (within the subset)
        let expected_front = vec![1, 2];

        assert_eq!(current_front, expected_front);
    }

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
