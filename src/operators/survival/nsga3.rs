use std::borrow::Cow;

use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::Solve;
use ndarray_stats::QuantileExt;
use rand::prelude::SliceRandom;

use crate::genetic::{Fronts, Population, PopulationFitness};
use crate::helpers::extreme_points::{get_nadir, get_nideal};
use crate::operators::{FrontContext, GeneticOperator, SurvivalOperator};
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the NSGA3 algorithm presented in the paper
/// An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach

#[derive(Clone, Debug)]
pub struct ReferencePoints {
    points: Array2<f64>,
    are_aspirational: bool,
}

impl ReferencePoints {
    pub fn new(points: Array2<f64>, are_aspirational: bool) -> Self {
        Self {
            points,
            are_aspirational,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Nsga3ReferencePointsSurvival {
    reference_points: ReferencePoints, // Each row is a reference point
}

impl GeneticOperator for Nsga3ReferencePointsSurvival {
    fn name(&self) -> String {
        "Nsga3ReferencePointsSurvival".to_string()
    }
}

impl Nsga3ReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>) -> Self {
        let reference_points = ReferencePoints::new(reference_points, true);
        Self { reference_points }
    }
}

impl SurvivalOperator for Nsga3ReferencePointsSurvival {
    fn survival_score(
        &self,
        _front_fitness: &PopulationFitness,
        _context: FrontContext,
        _rng: &mut dyn RandomGenerator,
    ) -> Array1<f64> {
        unimplemented!("It doesn't use survival score")
    }

    fn operate(
        &self,
        fronts: &mut Fronts,
        n_survive: usize,
        rng: &mut dyn RandomGenerator,
    ) -> Population {
        // Drain fronts to consume them and get an iterator of owned Population values.
        let mut drained = fronts.drain(..);

        // Initialize survivors with the first front.
        let mut survivors = drained
            .next()
            .expect("No fronts available to form survivors");
        let mut n_survivors = survivors.len();

        // Iterate over the remaining fronts.
        for front in drained {
            let front_size = front.len();
            if n_survivors + front_size <= n_survive {
                survivors = Population::merge(&survivors, &front);
                n_survivors += front_size;
            } else {
                // Only part of this front is needed.
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // this is the S_t varaible defined in the Algorithm 1 in the presented paper
                    let st = Population::merge(&survivors, &front);
                    let z_min = get_nideal(&st.fitness);
                    // they are a = (a1, ..., a_m) the intercepts
                    let intercepts = compute_intercepts(&st.fitness, &z_min);
                    // This call is the normalize function (Algorithm 2)
                    let normalized_fitness = normalize(&st.fitness, &z_min, &intercepts);
                    // Now as the paper says, if the points are aspirational then normalize
                    // Use Cow so that when aspirational is false, we borrow the reference points.
                    let zr: Cow<Array2<f64>> = if self.reference_points.are_aspirational {
                        Cow::Owned(normalize(
                            &self.reference_points.points,
                            &z_min,
                            &intercepts,
                        ))
                    } else {
                        Cow::Borrowed(&self.reference_points.points)
                    };
                    let (assignments, distances) = associate(&normalized_fitness, &zr);

                    // Compute niching count for every individual except in the splitting front
                    let n_complete = survivors.len();
                    let survivors_assignments = &assignments[0..n_complete];
                    let mut niche_counts = compute_niche_counts(survivors_assignments, zr.nrows());

                    // Perform niching on the splitting front to select exactly `remaining`
                    // Prepare the indices of solutions that belong to the splitting front.
                    // These are the ones with index >= n_complete.
                    let mut splitting_indices: Vec<usize> = (n_complete..st.len()).collect();
                    let chosen_indices = niching(
                        remaining,
                        &mut niche_counts,
                        &assignments,
                        &distances,
                        &mut splitting_indices,
                        rng,
                    );
                    let selection_from_splitting_front = front.selected(&chosen_indices);
                    survivors = Population::merge(&survivors, &selection_from_splitting_front);
                }
                break;
            }
        }
        survivors
    }
}

/// Calculates the Achievement Scalarizing Function (ASF) for a given solution `x`
/// (which represents the translated objective values f'_i(x)) and a weight vector `w`.
/// Any weight equal to zero is replaced by a small epsilon (1e-6) to avoid division by zero.
/// This is the equation (4) in the presented paper
fn asf(x: &Array1<f64>, w: &Array1<f64>) -> f64 {
    // Compute the element-wise ratio: f'_i(x) / w_i.
    let ratios = x / w;
    // The ASF is the maximum of these ratios.
    ratios.fold(std::f64::MIN, |acc, &val| acc.max(val))
}

/// Computes the extreme points (z_max) from the translated population.
/// For each objective j, constructs a weight vector:
///   w^j = [eps, ..., 1.0 (at position j), ..., eps],
/// then selects the solution that minimizes ASF(s, w^j) using argmin from ndarray-stats.
fn compute_extreme_points(translated_pop: &PopulationFitness, epsilon: f64) -> Array2<f64> {
    let n_objectives = translated_pop.ncols();
    // Initialize an array to hold the extreme vectors; one per objective.
    let mut extreme_points = Array2::<f64>::zeros((n_objectives, n_objectives));

    // For each objective j, compute the corresponding extreme point.
    for j in 0..n_objectives {
        // Build the weight vector for objective j:
        // All elements are epsilon except for the j-th element which is 1.0.
        let mut weight = Array1::<f64>::from_elem(n_objectives, epsilon);
        weight[j] = 1.0;

        // Compute the ASF value for each solution in the translated population.
        let asf_values: Vec<f64> = translated_pop
            .outer_iter()
            .map(|solution| asf(&solution.to_owned(), &weight))
            .collect();
        let asf_array = Array1::from(asf_values);

        // Use argmin from ndarray-stats to get the index of the minimum ASF value.
        let best_idx = asf_array.argmin().unwrap();

        // The extreme point for objective j is the translated objective vector
        // of the solution that minimized ASF with weight vector w^j.
        let extreme = translated_pop.row(best_idx);
        // Place this extreme vector in the j-th row of extreme_points.
        extreme_points.slice_mut(s![j, ..]).assign(&extreme);
    }
    extreme_points
}

/// Computes the intercepts vector `a` by solving the linear system:
/// Z_max * b = 1, where 1 is a vector of ones.
/// then the intercepts in the objective axis are given by a = 1/b
fn compute_intercepts(population_fitness: &PopulationFitness, z_min: &Array1<f64>) -> Array1<f64> {
    // Create a vector of ones with length equal to the number of rows (or objectives)
    let translated = population_fitness - z_min;
    let z_max = compute_extreme_points(&translated, 1e-6);
    let m = z_max.nrows();
    let ones = Array1::ones(m);
    let solution = z_max.solve_into(ones);
    // Solve the system Z_max * b = ones using the ndarray-linalg's solve_into method.
    // The method returns Result<Array1<f64>, _>.
    match solution {
        Ok(a) => {
            // Check if any component of a is nearly zero.
            if a.iter().any(|&val| val.abs() < 1e-6) {
                // Fallback: use min-max normalization by returning the column-wise maximums.
                get_nadir(&translated)
            } else {
                // Calculate intercepts as 1 / a.
                let intercept = a.mapv(|val| 1.0 / val);
                // Additional check: if the computed intercept is less than the observed maximum,
                // use the observed maximum (fallback to min-max).
                let fallback = get_nadir(&translated);
                // Replace zip_map with an iterator-based elementwise combination.
                let combined: Vec<f64> = intercept
                    .iter()
                    .zip(fallback.iter())
                    .map(|(&calc, &fb)| if calc < fb { fb } else { calc })
                    .collect();
                Array1::from(combined)
            }
        }
        Err(_) => {
            // If solving the system fails, fallback to min-max normalization.
            get_nadir(&translated)
        }
    }
}

/// Normalizes given the ideal point (zmin) and the intercepts (a).
/// For each xi, the normalization is defined as:
///
///   xⁿ_i = (x_i - zmin_i) / (a_i - zmin_i)
///
/// # Arguments
///
/// * `x` - A 2D array where each row is going to be normalized to the hyperplane
/// * `zmin` - An Array1 representing the ideal point for each objective.
/// * `intercepts` - An Array1 with the intercepts a for each objective.
///
/// This and compute_intercepts conforms the Algorithm 2 in the presented paper
fn normalize(x: &Array2<f64>, z_min: &Array1<f64>, intercepts: &Array1<f64>) -> Array2<f64> {
    // Calculate the denominators (a_i - zmin_i) for each objective.
    let denom = intercepts - z_min;
    let translated = x - z_min;
    translated / denom
}

/// Associates each solution s (each row in st) with the reference w (each row in zr)
/// that minimizes the perpendicular distance d⊥(s, w).
/// This is the algorithm (3) in the presented paper
fn associate(st_fitness: &PopulationFitness, zr: &Array2<f64>) -> (Vec<usize>, Vec<f64>) {
    let n = st_fitness.nrows();

    // 1. Compute squared norms for each solution: shape (n,)
    let norm_s_sq: Array1<f64> = st_fitness.outer_iter().map(|s| s.dot(&s)).collect();

    // 2. Compute squared norms for each reference: shape (m,)
    let norm_w_sq: Array1<f64> = zr.outer_iter().map(|w| w.dot(&w)).collect();

    // 3. Compute dot products between each s and each w: matrix A of shape (n, m)
    let dot = st_fitness.dot(&zr.t());

    // 4. Reshape norms for broadcasting:
    let norm_s_sq = norm_s_sq.insert_axis(Axis(1)); // shape (n, 1)
    let norm_w_sq = norm_w_sq.insert_axis(Axis(0)); // shape (1, m)

    // 5. Compute the squared dot products
    let dot_sq = dot.mapv(|x| x * x);

    // 6. Compute the squared perpendicular distance:
    // d2[i, j] = ||s_i||^2 - (dot[i,j]^2 / ||w_j||^2)
    let d2 = &norm_s_sq - &dot_sq / &norm_w_sq;

    // 8. For each solution (each row in d), find the index of the reference that minimizes the distance.
    let mut assignments = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);

    for row in d2.outer_iter() {
        let (min_idx, &min_val) = row
            .indexed_iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assignments.push(min_idx);
        distances.push(min_val);
    }

    (assignments, distances)
}

/// Computes the niche counts for each reference point given only the assignments
/// from individuals that are not part of the splitting front.
///
/// # Arguments
/// * `assignments` - A slice of reference point indices assigned to individuals in complete fronts.
/// * `n_references` - The total number of reference points.
///
/// # Returns
/// A vector of niche counts (ρ_j) where each element corresponds to a reference point.
fn compute_niche_counts(assignments: &[usize], n_references: usize) -> Vec<usize> {
    let mut niche_counts = vec![0; n_references];
    for &assigned_ref in assignments.iter() {
        niche_counts[assigned_ref] += 1;
    }
    niche_counts
}

/// Implements the Niching procedure (algorithm 4 in the presented paper) for NSGA-III.
///
/// # Arguments
/// * `n_remaining` - The number of individuals left to assign.
/// * `niche_counts` - A mutable vector of niche counts (ρ_j) for each reference point.
/// * `assignments` - A vector where each element is the reference point index (π(s)) associated with a solution.
/// * `distances` - A vector where each element is the perpendicular distance d(s) for a solution.
/// * `available_refs` - A mutable vector of the available reference point indices (initially all indices in Zr).
/// * `splitting_front` - A mutable vector containing the indices of solutions in the splitting front (Fl).
///
/// # Returns
/// A vector of solution indices (Pt+1) that have been selected for the next population.
fn niching(
    mut n_remaining: usize,
    niche_counts: &mut Vec<usize>,
    assignments: &Vec<usize>,
    distances: &Vec<f64>,
    splitting_front: &mut Vec<usize>,
    rng: &mut dyn RandomGenerator,
) -> Vec<usize> {
    // Create available_refs inside the function based on the number of reference points.
    let mut available_refs: Vec<usize> = (0..niche_counts.len()).collect();
    let mut pt_next = Vec::new();

    // While there are still individuals to assign...
    while n_remaining > 0 {
        // If no reference points remain, break out.
        if available_refs.is_empty() {
            break;
        }

        // Step 3: Compute Jmin = { j in available_refs such that ρ_j is minimal }
        let min_count = available_refs
            .iter()
            .map(|&j| niche_counts[j])
            .min()
            .unwrap(); // safe because available_refs is not empty
        let jmin: Vec<usize> = available_refs
            .iter()
            .copied()
            .filter(|&j| niche_counts[j] == min_count)
            .collect();

        // Step 4: Select a random reference point from Jmin
        let j_bar = *rng.choose_usize(&jmin).unwrap();

        // Step 5: I_j_bar = { s in splitting_front such that assignments[s] == j_bar }
        let i_j_bar: Vec<usize> = splitting_front
            .iter()
            .copied()
            .filter(|&s| assignments[s] == j_bar)
            .collect();

        if !i_j_bar.is_empty() {
            // If there are candidate solutions for j_bar in the splitting front:
            let s_chosen = if niche_counts[j_bar] == 0 {
                // If the niche count for j_bar is zero, select the solution with minimum d(s)
                *i_j_bar
                    .iter()
                    .min_by(|&&s1, &&s2| distances[s1].partial_cmp(&distances[s2]).unwrap())
                    .unwrap()
            } else {
                // Otherwise, select a random solution from I_j_bar
                *rng.choose_usize(&i_j_bar).unwrap()
            };

            // Add the chosen solution to Pt+1
            pt_next.push(s_chosen);

            // Remove the chosen solution from the splitting front
            if let Some(pos) = splitting_front.iter().position(|&s| s == s_chosen) {
                splitting_front.remove(pos);
            }

            // Update the niche count for j_bar and decrement n_remaining
            niche_counts[j_bar] += 1;
            n_remaining -= 1;
        } else {
            // If I_j_bar is empty, remove j_bar from available_refs
            if let Some(pos) = available_refs.iter().position(|&j| j == j_bar) {
                available_refs.remove(pos);
            }
        }
    }

    pt_next
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::{RandomGenerator, TestDummyRng};
    use ndarray::array;
    use rand::RngCore;

    #[test]
    fn test_asf_with_identity_weights() {
        // Example translated objective vector.
        let x = array![0.2, 0.5, 0.3];

        // When using the identity matrix, the weight vectors are unit vectors.
        // For a 3-objective problem, these are:
        //   w1 = [1, 0, 0]
        //   w2 = [0, 1, 0]
        //   w3 = [0, 0, 1]
        //
        // Note: Since zeros in the weight vector are replaced with epsilon (1e-6),
        // the division will produce very large values in those components.

        // Test for w1 = [1, 0, 0]
        let w1 = array![1.0, 1e-6, 1e-6];
        let asf1 = asf(&x, &w1);
        // For w1, adjusted weights are [1.0, 1e-6, 1e-6] and the ratios are:
        //   0.2/1.0 = 0.2, 0.5/1e-6 = 500000, 0.3/1e-6 = 300000.
        // Thus, ASF should be 500000.
        assert_eq!(asf1, 500000.0);

        // Test for w2 = [0, 1, 0]
        let w2 = array![1e-6, 1.0, 1e-6];
        let asf2 = asf(&x, &w2);
        // For w2, adjusted weights are [1e-6, 1.0, 1e-6] and the ratios are:
        //   0.2/1e-6 = 200000, 0.5/1.0 = 0.5, 0.3/1e-6 = 300000.
        // Thus, ASF should be 300000.
        assert_eq!(asf2, 300000.0);

        // Test for w3 = [0, 0, 1]
        let w3 = array![1e-6, 1e-6, 1.0];
        let asf3 = asf(&x, &w3);
        // For w3, adjusted weights are [1e-6, 1e-6, 1.0] and the ratios are:
        //   0.2/1e-6 = 200000, 0.5/1e-6 = 500000, 0.3/1.0 = 0.3.
        // Thus, ASF should be 500000.
        assert_eq!(asf3, 500000.0);
    }

    // Test compute_extreme_points using a simple two-solution, two-objective case.
    #[test]
    fn test_compute_extreme_points() {
        // Two solutions:
        //   Solution A: [1.0, 10.0]
        //   Solution B: [10.0, 1.0]
        let pop = array![[1.0, 10.0], [10.0, 1.0]];
        let epsilon = 1e-6;
        let extreme = compute_extreme_points(&pop, epsilon);

        // For objective 0, we expect the extreme point to be B: [10.0, 1.0]
        // For objective 1, we expect the extreme point to be A: [1.0, 10.0]
        let expected = array![[10.0, 1.0], [1.0, 10.0]];

        assert_eq!(
            extreme, expected,
            "Computed extreme points do not match expected values"
        );
    }

    // Test compute_intercepts using a simple two-solution case.
    #[test]
    fn test_compute_intercepts() {
        // Using the same pop: A = [1, 10], B = [10, 1]
        let pop = array![[1.0, 10.0], [10.0, 1.0]];
        // Ideal point: column-wise minimum is [1, 1]
        let z_min = array![1.0, 1.0];
        let intercepts = compute_intercepts(&pop, &z_min);
        // For our simple case, expected intercepts are roughly 9 for both objectives.
        assert!((intercepts[0] - 9.0).abs() < 1e-2);
        assert!((intercepts[1] - 9.0).abs() < 1e-2);
    }

    // Test normalize with a simple 2x2 matrix.
    #[test]
    fn test_normalize() {
        // Let x be a 2x2 matrix.
        let x = array![[2.0, 3.0], [4.0, 5.0]];
        let z_min = array![1.0, 2.0];
        let intercepts = array![9.0, 9.0];
        // denom = intercepts - z_min = [8, 7]
        // Row0 normalized: [(2-1)/8, (3-2)/7] = [0.125, 0.142857...]
        // Row1 normalized: [(4-1)/8, (5-2)/7] = [0.375, 0.428571...]
        let normalized = normalize(&x, &z_min, &intercepts);
        let expected = array![[0.125, 0.142857], [0.375, 0.428571]];
        for (a, b) in normalized.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "Expected {}, got {}", b, a);
        }
    }

    // Test associate: simple case with two solutions and two reference points.
    #[test]
    fn test_associate() {
        // Two solutions:
        // A = [1, 10] and B = [10, 1]
        let st_fitness = array![[1.0, 10.0], [10.0, 1.0]];
        // Reference set: identity-like
        let zr = array![[1.0, 0.0], [0.0, 1.0]];
        let (assignments, distances) = associate(&st_fitness, &zr);
        // For solution A, expected assignment is 1; for B, expected is 0.
        assert_eq!(assignments, vec![1, 0]);
        // Expected distances are approximately 1.0 (allowing for floating-point error)
        for (i, d) in distances.iter().enumerate() {
            assert!(
                ((*d) - 1.0).abs() < 1e-5,
                "Solution {}: expected distance 1, got {}",
                i,
                d
            );
        }
    }

    // Test compute_niche_counts.
    #[test]
    fn test_compute_niche_counts() {
        // Given assignments, e.g., [0, 1, 0, 1, 1]
        let assignments = vec![0, 1, 0, 1, 1];
        let n_references = 2;
        let niche_counts = compute_niche_counts(&assignments, n_references);
        assert_eq!(niche_counts, vec![2, 3]);
    }

    struct FakeRandomGenerator {
        dummy: TestDummyRng,
    }

    impl FakeRandomGenerator {
        fn new() -> Self {
            Self {
                dummy: TestDummyRng,
            }
        }
    }

    impl RandomGenerator for FakeRandomGenerator {
        fn rng(&mut self) -> &mut dyn RngCore {
            &mut self.dummy
        }
        fn choose_usize<'a>(&mut self, vector: &'a [usize]) -> Option<&'a usize> {
            // Always choose the first element for deterministic behavior.
            vector.first()
        }
    }

    #[test]
    fn test_niching_with_dummy_rng() {
        // Inputs for the niching function.
        let assignments = vec![0, 1, 0, 1]; // Each solution's assigned reference point.
        let distances = vec![10.0, 20.0, 30.0, 40.0]; // Perpendicular distances.
        let mut niche_counts = vec![0, 0]; // For two reference points.
        let mut splitting_front = vec![0, 1, 2, 3]; // Indices of solutions in the splitting front.
        let n_remaining = 2; // We want to select 2 individuals.

        let mut dummy_rng = FakeRandomGenerator::new();

        let chosen = niching(
            n_remaining,
            &mut niche_counts,
            &assignments,
            &distances,
            &mut splitting_front,
            &mut dummy_rng,
        );
        // Expected: first iteration picks index 0, second iteration picks index 1.
        assert_eq!(chosen, vec![0, 1]);
    }
}
