use std::cmp::Ordering;
use std::fmt::Debug;

use ndarray::{Array1, Array2, Axis, Zip};

use crate::genetic::PopulationFitness;
use crate::helpers::extreme_points::{get_nadir, get_nideal};
use crate::operators::{
    FrontContext, GeneticOperator, SurvivalOperator, SurvivalScoringComparison,
};
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the R-NSGA2 algorithm presented in the paper
/// Reference Point Based Multi-Objective Optimization Using Evolutionary Algorithms

#[derive(Clone, Debug)]
pub struct Rnsga2ReferencePointsSurvival {
    reference_points: Array2<f64>,
    epsilon: f64,
}

impl GeneticOperator for Rnsga2ReferencePointsSurvival {
    fn name(&self) -> String {
        "Rnsga2ReferencePointsSurvival".to_string()
    }
}

impl Rnsga2ReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>, epsilon: f64) -> Self {
        Self {
            reference_points,
            epsilon,
        }
    }
}

/// This function computes the weighted, normalized Euclidean distance matrix between each solution
/// in the front (fitness matrix) and a set of reference points.
/// (It is already defined in your diversity_metrics module.)
/// Here we assume it is available as:
///    weighted_distance_matrix(fitness: &PopulationFitness, reference: &Array2<f64>) -> Array2<f64>
/// and that reference_points_rank_distance calls it to compute a ranking.
///
/// The splitting front procedure below uses weighted_distance_matrix(fitness, fitness)
/// to compute the internal distances among solutions.

impl SurvivalOperator for Rnsga2ReferencePointsSurvival {
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Minimize
    }

    fn survival_score(
        &self,
        front_fitness: &PopulationFitness,
        context: FrontContext,
        rng: &mut dyn RandomGenerator,
    ) -> Array1<f64> {
        let nadir = get_nadir(&front_fitness);
        let nideal = get_nideal(&front_fitness);
        let n_objectives = front_fitness.ncols();
        let weights = Array1::from_elem(n_objectives, 1.0 / (n_objectives as f64));
        if let FrontContext::Splitting = context {
            assign_crowding_distance_splitting_front(
                &front_fitness,
                &self.reference_points,
                &weights,
                self.epsilon,
                &nadir,
                &nideal,
                rng,
            )
        } else {
            assign_crowding_distance_to_inner_front(
                &front_fitness,
                &self.reference_points,
                &weights,
                &nadir,
                &nideal,
            )
        }
    }
}

/// Computes the weighted, normalized Euclidean distance between two objective vectors `f1` and `f2`.
/// Normalization is performed using the provided ideal (`nideal`) and nadir (`nadir`) points.
/// If for any objective the range (nadir - nideal) is zero, the normalized difference is set to 0.0.
/// This is the equation (3) in the presented paper
fn weighted_normalized_euclidean_distance(
    f1: &Array1<f64>,
    f2: &Array1<f64>,
    weights: &Array1<f64>,
    nideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> f64 {
    // Compute the element-wise difference between f1 and f2.
    let diff = f1 - f2;
    // Compute the range for normalization.
    let ranges = nadir - nideal;
    // Allocate an array to store the normalized differences.
    let mut normalized_diff = Array1::<f64>::zeros(diff.len());
    // Populate normalized_diff: if the range is zero, set the value to 0.0.
    Zip::from(&mut normalized_diff)
        .and(&diff)
        .and(&ranges)
        .for_each(|out, &d, &r| {
            *out = if r == 0.0 { 0.0 } else { d / r };
        });

    // Compute the weighted sum of squared normalized differences.
    let weighted_sum_sq: f64 = normalized_diff.mapv(|x| x * x).dot(weights);
    // Return the square root of the weighted sum.
    weighted_sum_sq.sqrt()
}

/// Computes the sum of normalized absolute differences between two solutions.
/// For each objective, it calculates:
///    |sol1[j] - sol2[j]| / (nadir[j] - nideal[j])
/// and returns the sum over all objectives.
fn sum_normalized_difference(
    sol1: &Array1<f64>,
    sol2: &Array1<f64>,
    nideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> f64 {
    let ranges = nadir - nideal;
    sol1.iter()
        .zip(sol2.iter())
        .zip(ranges.iter())
        .map(
            |((a, b), &r)| {
                if r == 0.0 {
                    0.0
                } else {
                    (a - b).abs() / r
                }
            },
        )
        .sum()
}

fn distance_to_reference(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    nideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> Array1<f64> {
    // --- Step 1: Compute initial crowding distances based on reference points ---
    // Initialize each solution's crowding distance with infinity.
    let num_front_individuals = front_fitness.nrows();
    let mut crowding = vec![f64::INFINITY; num_front_individuals];

    for rp in reference_points.axis_iter(Axis(0)) {
        let mut solution_distances: Vec<(usize, f64)> = front_fitness
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, sol)| {
                // Convert the row view into an Array1 for distance computation.
                let distance = weighted_normalized_euclidean_distance(
                    &sol.to_owned(),
                    &rp.to_owned(),
                    weights,
                    nideal,
                    nadir,
                );
                (i, distance)
            })
            .collect();
        // Sort solutions by distance (ascending order).
        solution_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        // Assign rank based on order: the closest solution gets rank 1, the next gets rank 2, etc.
        for (rank, (i, _)) in solution_distances.into_iter().enumerate() {
            let current_rank = (rank + 1) as f64;
            if current_rank < crowding[i] {
                crowding[i] = current_rank;
            }
        }
    }
    Array1::from_vec(crowding)
}

/// Assigns a crowding distance (ranking) to each solution.
///
/// The algorithm works in two stages:
///
///   Step 1: For each reference point, compute the normalized distance from each solution and sort the solutions
///           in ascending order (closest gets rank 1). For each solution, store the best (minimum) rank across all
///           reference points.
///
///   Step 2 (extended as Step 3): Group solutions that have a sum of normalized differences (across objectives) ≤ epsilon.
///           From each group, randomly retain one solution and assign a penalty (infinity) as the crowding distance
///           to the rest.
///
/// # Parameters
/// - `solutions`: An Array2<f64> where each row is a solution.
/// - `reference_points`: An Array2<f64> where each row is a reference point.
/// - `weights`, `nideal`, `nadir`: Parameters used to normalize the distance.
/// - `epsilon`: The threshold used to group similar solutions.
/// - `rng`: A mutable reference to an object implementing `RngCore` to be used for random shuffling.
///
/// # Returns
/// A vector of crowding distances (as f64) for each solution.
fn assign_crowding_distance_to_inner_front(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    nadir: &Array1<f64>,
    nideal: &Array1<f64>,
) -> Array1<f64> {
    distance_to_reference(&front_fitness, &reference_points, &weights, &nideal, &nadir)
}

fn assign_crowding_distance_splitting_front(
    front_fitness: &PopulationFitness,
    reference_points: &Array2<f64>,
    weights: &Array1<f64>,
    epsilon: f64,
    nadir: &Array1<f64>,
    nideal: &Array1<f64>,
    rng: &mut dyn RandomGenerator,
) -> Array1<f64> {
    let num_front_individuals = front_fitness.nrows();
    let mut crowding = assign_crowding_distance_to_inner_front(
        &front_fitness,
        &reference_points,
        &weights,
        nadir,
        nideal,
    );

    // --- Step 3: Group similar solutions using epsilon ---
    // Group solutions that have a sum of normalized differences ≤ epsilon.
    let mut visited = vec![false; num_front_individuals];
    let mut groups: Vec<Vec<usize>> = Vec::new();

    for i in 0..num_front_individuals {
        if visited[i] {
            continue;
        }
        let mut group = vec![i];
        visited[i] = true;
        let sol_i = front_fitness.row(i).to_owned();
        for j in (i + 1)..num_front_individuals {
            if !visited[j] {
                let sol_j = front_fitness.row(j).to_owned();
                let sum_diff = sum_normalized_difference(&sol_i, &sol_j, nideal, nadir);
                if sum_diff <= epsilon {
                    group.push(j);
                    visited[j] = true;
                }
            }
        }
        groups.push(group);
    }

    // For each group with more than one solution, randomly retain one solution and assign a penalty (infinity)
    // to all other solutions in that group.
    for group in groups {
        if group.len() > 1 {
            let mut group_copy = group.clone();
            rng.shuffle_vec_usize(&mut group_copy);
            // Retain the first solution in the shuffled group.
            // All other solutions in the group receive infinity.
            for &idx in group_copy.iter().skip(1) {
                crowding[idx] = f64::INFINITY;
            }
        }
    }
    crowding
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_distance_zero() {
        // When both objective vectors are identical, the distance should be 0.
        let f1 = array![1.0, 2.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let nideal = array![0.0, 0.0];
        let nadir = array![1.0, 1.0];
        let distance = weighted_normalized_euclidean_distance(&f1, &f2, &weights, &nideal, &nadir);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_distance_simple() {
        // Example:
        // f1 = [3, 4], f2 = [1, 2]
        // nideal = [1, 2], nadir = [5, 6] => range = [4, 4]
        // Normalized differences = [(3-1)/4, (4-2)/4] = [0.5, 0.5]
        // With weights = [1, 1], the weighted sum of squares is 0.5^2 + 0.5^2 = 0.25 + 0.25 = 0.5,
        // and the distance is sqrt(0.5).
        let f1 = array![3.0, 4.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let nideal = array![1.0, 2.0];
        let nadir = array![5.0, 6.0];
        let distance = weighted_normalized_euclidean_distance(&f1, &f2, &weights, &nideal, &nadir);
        let expected = (0.25_f64 + 0.25).sqrt();
        assert!((distance - expected).abs() < 1e-6);
    }

    #[test]
    fn test_distance_with_zero_range() {
        // Test scenario where one of the objectives has a zero range.
        // For the first objective: nideal = 1, nadir = 1, so range = 0 and normalized difference = 0.
        // For the second objective: nideal = 2, nadir = 6, so range = 4 and normalized difference = (4-2)/4 = 0.5.
        let f1 = array![3.0, 4.0];
        let f2 = array![1.0, 2.0];
        let weights = array![1.0, 1.0];
        let nideal = array![1.0, 2.0];
        let nadir = array![1.0, 6.0];
        let distance: f64 =
            weighted_normalized_euclidean_distance(&f1, &f2, &weights, &nideal, &nadir);
        let expected: f64 = (0.0_f64 * 0.0 + 0.5 * 0.5).sqrt();
        assert!((distance - expected).abs() < 1e-6);
    }
}
