use std::collections::HashSet;
use std::fmt::Debug;

use ndarray_stats::QuantileExt;
use numpy::ndarray::{Array1, Array2, ArrayView1, Axis};

use crate::genetic::PopulationFitness;
use crate::helpers::extreme_points::get_nideal;
use crate::helpers::linalg::{cross_p_distances, lp_norm_arrayview};
use crate::operators::{
    survival::helpers::HyperPlaneNormalization, FrontContext, GeneticOperator, SurvivalOperator,
};
use crate::random::RandomGenerator;

struct AgeMoeaHyperPlaneNormalization;

impl AgeMoeaHyperPlaneNormalization {
    pub fn new() -> Self {
        Self
    }
}
impl HyperPlaneNormalization for AgeMoeaHyperPlaneNormalization {
    fn compute_extreme_points(&self, population_fitness: &PopulationFitness) -> Array2<f64> {
        // Number of objectives (columns)
        let n_objectives = population_fitness.shape()[1];
        // Initialize the Z_max matrix with dimensions (num_objectives x num_objectives)
        let mut z_max = Array2::<f64>::zeros((n_objectives, n_objectives));

        // For each objective i, identify the extreme vector.
        for i in 0..n_objectives {
            // Get the i-th column as a 1D array view.
            let col = population_fitness.column(i);
            // Use ndarray-stats argmax to find the row index of the maximum element in the column.
            let max_row_index = col.argmax().expect("Column must have at least one element");

            // The extreme vector is the row at max_row_index.
            let extreme_vector = population_fitness.row(max_row_index);
            // Place the extreme vector as the i-th row of Z_max.
            for j in 0..n_objectives {
                z_max[[i, j]] = extreme_vector[j];
            }
        }
        z_max
    }
}

#[derive(Clone, Debug)]
pub struct AgeMoeaSurvival;

impl GeneticOperator for AgeMoeaSurvival {
    fn name(&self) -> String {
        "RankCrowdingSurvival".to_string()
    }
}

impl AgeMoeaSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl SurvivalOperator for AgeMoeaSurvival {
    fn survival_score(
        &self,
        front_fitness: &PopulationFitness,
        context: FrontContext,
        _rng: &mut dyn RandomGenerator,
    ) -> Array1<f64> {
        let normalized_front_fitness = normalize_front_with_intercepts(front_fitness);
        let central_point = get_central_point_normalized(&normalized_front_fitness);
        // Check if the central point is a zero vector.
        // This is the rare case where there is a single optimum,
        // meaning each individual's fitness is indeed z_ideal.
        // In this case, return a survival score vector with the same number of elements as front_fitness,
        // filled entirely with infinity.
        if central_point.iter().all(|&x| x == 0.0) {
            return Array1::from_elem(front_fitness.nrows(), std::f64::INFINITY);
        }
        let p = compute_exponent_p(&central_point);
        if let FrontContext::First = context {
            assign_survival_scores_first_front(&normalized_front_fitness, p)
        } else {
            assign_survival_scores_higher_front(&normalized_front_fitness, p)
        }
    }
}

/// Normalizes the first non-dominated front (F1) using intercepts obtained by solving
/// the system Z_max * a = 1.
/// The normalization is performed by translating the front by subtracting the ideal point
/// and then dividing each objective by its corresponding intercept.
pub fn normalize_front_with_intercepts(front_fitness: &PopulationFitness) -> PopulationFitness {
    // Compute the ideal point and translate the front.
    let z_min = get_nideal(front_fitness);
    let translated = front_fitness - &z_min;

    if translated.iter().all(|&value| value == 0.0) {
        return translated;
    }
    let normalizer = AgeMoeaHyperPlaneNormalization::new();
    // Obtain the intercepts by solving the linear system.
    let intercepts = normalizer.compute_hyperplane_intercepts(&translated);

    // Normalize each solution element-wise by the intercepts (using broadcasting).
    translated / &intercepts
}

/// Computes the central point C of the non-dominated front given normalized objectives.
/// This is the equation (6) in the referenced paper
///
/// C is defined as the solution that minimizes its perpendicular distance to the unit vector
/// in the direction of (1, 1, ..., 1). That is, we define:
///
///   beta = (1, 1, ..., 1)
///   beta_hat = beta / ||beta||   (with ||beta|| = sqrt(M), where M is the number of objectives)
///
/// For each solution x, the perpendicular distance is computed as:
///
///   distance = || x - (x · beta_hat) * beta_hat ||₂
///
/// The method computes all distances in a vectorized way and returns the solution with the minimal distance.
///
/// # Arguments
///
/// * `normalized_objectives` - A 2D array (n x m) of normalized objectives (each row is a solution).
///
/// # Returns
///
/// An Array1<f64> corresponding to the central point in the normalized objective space.
fn get_central_point_normalized(normalized_fitness: &PopulationFitness) -> Array1<f64> {
    // Determine the number of objectives (columns)
    let num_objectives = normalized_fitness.shape()[1];

    // Create the reference vector beta = (1,1,...,1) and compute its unit vector beta_hat.
    let beta = Array1::<f64>::ones(num_objectives);
    let beta_norm = beta.dot(&beta).sqrt(); // sqrt(M)
    let beta_hat = beta.mapv(|x| x / beta_norm);

    // Compute the dot product of each solution (row) with beta_hat.
    // This yields an Array1 of length n.
    let dot_products = normalized_fitness.dot(&beta_hat);

    // Compute the projection for each solution: (dot_product)*beta_hat.
    // We expand dot_products to shape (n, 1) for broadcasting.
    let projections = dot_products.insert_axis(Axis(1)) * &beta_hat;

    // Compute the difference: the component of each solution perpendicular to beta_hat.
    let diff = normalized_fitness - &projections;

    // Compute the squared Euclidean norm for each row.
    let squared_norms = diff.map_axis(Axis(1), |row| row.dot(&row));

    // Find the index of the minimum squared norm.
    let min_index = squared_norms
        .argmin()
        .expect("There should be at least one solution in the front");

    // Return the row corresponding to the minimum perpendicular distance.
    normalized_fitness.row(min_index).to_owned()
}

/// Computes the exponent p from the central point C
/// This is the equation (8) in the presented paper
fn compute_exponent_p(central: &Array1<f64>) -> f64 {
    let m = central.len() as f64;
    // Ensure all components are > 0.
    for &value in central.iter() {
        assert!(
            value > 0.0,
            "All components of the central point must be > 0"
        );
    }
    // Compute the product of all coordinates.
    let product: f64 = central.iter().product();
    let ln_m = m.ln();
    let ln_product = product.ln();
    ln_m / (ln_m - ln_product)
}

/// Proximity(S) = || fₙ(S) ||ₚ
/// (Assuming the front is normalized so that the ideal point is at the origin.)
/// This is the equation (9) in the presented paper
fn proximity(normalized_individual_fitness: &ArrayView1<f64>, p: f64) -> f64 {
    lp_norm_arrayview(normalized_individual_fitness, p)
}

/// Assigns survival scores to the first front (F₁) based on the AGE-MOEA algorithm.
///
/// Extreme solutions (those with the maximum value in at least one objective)
/// are immediately assigned a score of +∞. Then, using two sets:
///   - The considered set (initially the extreme solutions)
///   - The remaining set (all other solutions)
/// the diversity for each candidate in the remaining set is computed as the sum of
/// the smallest and the second-smallest distances (using the L_p norm) from that candidate
/// to the solutions in the considered set. The candidate's value is defined as:
///
///     value[S] = diversity[S] / proximity[S]
///
/// The candidate with the maximum value is selected, its score assigned, and it is moved
/// from the remaining set to the considered set. This process continues until all candidates
/// have been processed.
///
/// # Arguments
///
/// * `front` - A (n x m) matrix where each row represents a normalized solution.
/// * `p` - The exponent for the L_p norm.
///
/// # Returns
///
/// An Array1<f64> containing the survival scores for each solution in the same order as the rows of `front`.
fn assign_survival_scores_first_front(
    normalized_front_fitness: &PopulationFitness,
    p: f64,
) -> Array1<f64> {
    let num_solutions = normalized_front_fitness.nrows();
    let num_objectives = normalized_front_fitness.ncols();

    // Initialize the scores vector.
    let mut scores = vec![0.0; num_solutions];

    // Identify extreme solutions: for each objective, select the solution with the maximum value.
    let mut extreme_indices = HashSet::new();
    for j in 0..num_objectives {
        let col = normalized_front_fitness.column(j);
        let idx = col
            .argmax()
            .expect("Each column must have at least one element");
        extreme_indices.insert(idx);
    }
    // Assign an infinite score to extreme solutions.
    for &idx in &extreme_indices {
        scores[idx] = f64::INFINITY;
    }

    // Remaining solutions: those that are not extreme.
    let mut remaining: Vec<usize> = (0..num_solutions)
        .filter(|&i| !extreme_indices.contains(&i))
        .collect();
    // Considered solutions: initially, the extreme solutions.
    let mut considered: Vec<usize> = extreme_indices.iter().copied().collect();

    // Precompute proximities for each solution using the proximity function.
    // Here, proximity(S) is defined as the L_p norm of solution S.
    let mut proximities = vec![0.0; num_solutions];
    for i in 0..num_solutions {
        proximities[i] = proximity(&normalized_front_fitness.row(i), p);
    }

    // Compute the pairwise distance matrix using the L_p norm.
    let distance_matrix = cross_p_distances(normalized_front_fitness, normalized_front_fitness, p);

    // Iterative process: while there are remaining solutions.
    while !remaining.is_empty() {
        // For each candidate S in the remaining set, compute diversity as:
        // diversity[S] = (min_{T in considered} dist[S, T] + second_min_{T in considered} dist[S, T])
        // and then compute value[S] = diversity[S] / proximity[S].
        let mut candidate_values: Vec<(usize, f64)> = Vec::new();
        for &s in &remaining {
            // Compute distances from candidate s to all solutions in the considered set.
            let mut dists: Vec<f64> = considered
                .iter()
                .map(|&t| distance_matrix[[s, t]])
                .collect();
            // Sort distances to get the smallest and second smallest.
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let diversity = if dists.len() >= 2 {
                dists[0] + dists[1]
            } else {
                // If there is only one element in the considered set.
                dists[0]
            };
            let value = if proximities[s] != 0.0 {
                diversity / proximities[s]
            } else {
                0.0
            };
            candidate_values.push((s, value));
        }

        // Select the candidate s* with the maximum value.
        if let Some(&(s_star, max_value)) = candidate_values
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            scores[s_star] = max_value;
            // Add s_star to the considered set.
            considered.push(s_star);
            // Remove s_star from the remaining set.
            remaining.retain(|&x| x != s_star);
        } else {
            break;
        }
    }

    Array1::from(scores)
}

/// An Array1<f64> with the survival scores for each solution in the same order as the rows of `normalized_front_fitness`.
pub fn assign_survival_scores_higher_front(
    normalized_front_fitness: &PopulationFitness,
    p: f64,
) -> Array1<f64> {
    // Iterate over each row (solution) and compute the score.
    let scores: Vec<f64> = normalized_front_fitness
        .axis_iter(Axis(0))
        .map(|row| proximity(&row, p))
        .collect();
    Array1::from(scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};

    use crate::genetic::{Fronts, Population};
    use crate::random::NoopRandomGenerator;

    /// Helper function for comparing Array1<f64> element-wise.
    fn assert_array1_abs_diff_eq(a: &Array1<f64>, b: &Array1<f64>, epsilon: f64) {
        assert_eq!(a.len(), b.len(), "Arrays have different lengths");
        for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (val_a - val_b).abs() < epsilon,
                "Difference at index {}: {} vs {}",
                i,
                val_a,
                val_b
            );
        }
    }

    /// Helper function for comparing Array2<f64> element-wise.
    fn assert_array2_abs_diff_eq(a: &Array2<f64>, b: &Array2<f64>, epsilon: f64) {
        assert_eq!(a.shape(), b.shape(), "Arrays have different shapes");
        for ((i, j), val_a) in a.indexed_iter() {
            let val_b = b[[i, j]];
            assert!(
                (val_a - val_b).abs() < epsilon,
                "Difference at position ({}, {}): {} vs {}",
                i,
                j,
                val_a,
                val_b
            );
        }
    }

    #[test]
    fn test_solve_intercepts() {
        // Use a front with 2 objectives that yields a unique solution.
        // Front:
        // [ [1.0, 2.0],
        //   [3.0, 1.0] ]
        // Ideal point = [1.0, 1.0]
        // Translated front = [ [0.0, 1.0],
        //                       [2.0, 0.0] ]
        // Extreme vector for objective 0: row 1 -> [2.0, 0.0]
        // Extreme vector for objective 1: row 0 -> [0.0, 1.0]
        // Z_max = [ [2.0, 0.0],
        //           [0.0, 1.0] ]
        // System: 2*a0 = 1, 1*a1 = 1 => a = [0.5, 1.0]
        // Intercepts = 1 / a = [2.0, 1.0]
        let front: PopulationFitness = array![[1.0, 2.0], [3.0, 1.0]];
        let normalizer = AgeMoeaHyperPlaneNormalization::new();
        let intercepts = normalizer.compute_hyperplane_intercepts(&front);
        let expected = array![2.0, 1.0];
        assert_eq!(&intercepts, &expected);
    }

    #[test]
    fn test_solve_intercepts_no_solution() {
        // Construct a front with 2 objectives that yields a singular Z_max.
        // For example, front:
        // [ [1.0, 2.0],
        //   [1.0, 3.0] ]
        // Ideal point = [1.0, 2.0]
        // Translated front = [ [0.0, 0.0],
        //                       [0.0, 1.0] ]
        // For objective 0:
        //   Column 0: [0.0, 0.0] -> argmax returns row 0 -> extreme vector = [0.0, 0.0]
        // For objective 1:
        //   Column 1: [0.0, 1.0] -> argmax returns row 1 -> extreme vector = [0.0, 1.0]
        // Z_max = [ [0.0, 0.0],
        //           [0.0, 1.0] ]
        // The system Z_max * a = [1, 1] has no solution.
        // Fallback returns get_nadir(translated):
        //   For column 0: max(0.0, 0.0) = 0.0, for column 1: max(0.0, 1.0) = 1.0
        // Expected intercepts = [0.0, 1.0]
        let front: PopulationFitness = array![[1.0, 2.0], [1.0, 3.0]];
        let normalizer = AgeMoeaHyperPlaneNormalization::new();
        let intercepts = normalizer.compute_hyperplane_intercepts(&front);
        let expected = array![0.0, 1.0];
        assert_eq!(&intercepts, &expected);
    }

    #[test]
    fn test_normalize_front_with_intercepts() {
        // Using the same front as in test_solve_intercepts.
        // Front:
        // [ [1.0, 2.0],
        //   [3.0, 1.0] ]
        // Ideal point = [1.0, 1.0]
        // Translated front = [ [0.0, 1.0],
        //                       [2.0, 0.0] ]
        // Intercepts = [2.0, 1.0]
        // Normalized front = [ [0/2, 1/1] = [0.0, 1.0],
        //                      [2/2, 0/1] = [1.0, 0.0] ]
        let front: PopulationFitness = array![[1.0, 2.0], [3.0, 1.0]];
        let normalized = normalize_front_with_intercepts(&front);
        let expected = array![[0.0, 1.0], [1.0, 0.0]];
        assert_array2_abs_diff_eq(&normalized, &expected, 1e-6);
    }

    #[test]
    fn test_get_central_point_normalized_2d() {
        // Create a 2D front with values between 0 and 1.
        // Example front:
        // Row 0: [0.1, 0.9]
        // Row 1: [0.9, 0.1]
        // Row 2: [0.5, 0.5]
        // With beta_hat = (1,1)/sqrt(2), the first two points have a nonzero perpendicular distance,
        // while [0.5, 0.5] has zero distance.
        let normalized = array![[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]];
        let central = get_central_point_normalized(&normalized);
        let expected = array![0.5, 0.5];
        assert_eq!(&central, &expected);
    }

    #[test]
    fn test_c_paper_example() {
        // Paper example: normalized front includes the two extreme points (1,0) and (0,1)
        // along with the central point (0.5, 0.5).
        // Here, values are in [0,1]. The perpendicular distances are:
        // For [1, 0]:
        //   dot = 1*0.7071 + 0*0.7071 ≈ 0.7071, projection = (0.5,0.5),
        //   diff = (1,0) - (0.5,0.5) = (0.5,-0.5), norm ≈ 0.7071.
        // For [0, 1]:
        //   diff = (-0.5,0.5), norm ≈ 0.7071.
        // For [0.5, 0.5]:
        //   diff = (0,0), norm = 0.
        // The expected central point is [0.5,0.5].
        let normalized = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]];
        let central = get_central_point_normalized(&normalized);
        let expected = array![0.5, 0.5];
        assert_array1_abs_diff_eq(&central, &expected, 1e-6);

        let p = 2.0;
        let prox = proximity(&central.view(), p);
        let expected_prox = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((prox - expected_prox).abs() < 1e-6);
    }

    #[test]
    fn test_proximity_2d() {
        // For a normalized 2D solution [0.5, 0.5] and p = 2,
        // its L₂ norm is sqrt(0.5² + 0.5²) = sqrt(0.5) ≈ 0.70710678.
        let solution = array![0.5, 0.5];
        let p = 2.0;
        let prox = proximity(&solution.view(), p);
        let expected = (0.5_f64.powi(2) + 0.5_f64.powi(2)).sqrt();
        assert!((prox - expected).abs() < 1e-6);
    }

    /// Test assign_survival_scores for p = 2 (squared distances).
    ///
    /// With the front:
    /// [1.0, 0.0]
    /// [0.5, 0.5]
    /// [0.0, 1.0]
    ///
    /// Extreme solutions (by maximum in at least one objective) are indices 0 and 2,
    /// and they should be assigned +∞.
    /// The candidate (index 1) has:
    ///   - Squared distance to index 0: 0.5
    ///   - Squared distance to index 2: 0.5
    ///   → Diversity = 1.0
    ///   - Proximity (L₂ norm of [0.5, 0.5]) = √0.5 ≈ 0.70710678
    ///   → Expected score ≈ 1.0 / 0.70710678 = 1.41421356
    #[test]
    fn test_assign_survival_scores_p2() {
        let front: Array2<f64> = array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];
        let p = 2.0;
        let scores = assign_survival_scores_first_front(&front, p);

        // Extreme solutions: indices 0 and 2.
        assert!(
            scores[0].is_infinite(),
            "Index 0 should be extreme (score = +∞)"
        );
        assert!(
            scores[2].is_infinite(),
            "Index 2 should be extreme (score = +∞)"
        );

        // Candidate at index 1 should have a score of about 1.41421356.
        let candidate_score = scores[1];
        let expected_score = 1.0 / 0.70710678; // ≈ 1.41421356
        assert!(
            (candidate_score - expected_score).abs() < 1e-6,
            "Index 1 should have a score of ~1.41421356, got {}",
            candidate_score
        );
    }

    /// Test assign_survival_scores for p = 1 (Manhattan distance).
    ///
    /// Using the same front:
    /// [1.0, 0.0]
    /// [0.5, 0.5]
    /// [0.0, 1.0]
    ///
    /// The Manhattan distance from [0.5, 0.5] to [1.0,0.0] is:
    ///   |0.5-1.0| + |0.5-0.0| = 0.5 + 0.5 = 1.0,
    /// and similarly to [0.0,1.0] is 1.0.
    /// Hence, diversity = 1.0 + 1.0 = 2.0, and proximity = |0.5|+|0.5| = 1.0.
    /// Expected score for the candidate = 2.0 / 1.0 = 2.0.
    #[test]
    fn test_assign_survival_scores_p1() {
        let front: Array2<f64> = array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];
        let p = 1.0;
        let scores = assign_survival_scores_first_front(&front, p);

        // Extreme solutions: indices 0 and 2.
        assert!(
            scores[0].is_infinite(),
            "Index 0 should be extreme (score = +∞)"
        );
        assert!(
            scores[2].is_infinite(),
            "Index 2 should be extreme (score = +∞)"
        );

        // Candidate at index 1 should have a score of about 2.0.
        let candidate_score = scores[1];
        assert!(
            (candidate_score - 2.0).abs() < 1e-6,
            "Index 1 should have a score of ~2.0, got {}",
            candidate_score
        );
    }

    /// Test assign_survival_scores with a 4-solution front.
    ///
    /// The front is:
    /// [1.0, 0.0]
    /// [0.8, 0.2]
    /// [0.2, 0.8]
    /// [0.0, 1.0]
    ///
    /// Extreme solutions (max per objective) should be indices 0 and 3.
    /// The remaining candidates (indices 1 and 2) should have finite, positive scores.
    #[test]
    fn test_assign_survival_scores_multiple() {
        let front: Array2<f64> = array![[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]];
        let p = 2.0;
        let scores = assign_survival_scores_first_front(&front, p);

        // Extreme solutions: indices 0 and 3.
        assert!(
            scores[0].is_infinite(),
            "Index 0 should be extreme (score = +∞)"
        );
        assert!(
            scores[3].is_infinite(),
            "Index 3 should be extreme (score = +∞)"
        );

        // Check that candidates (indices 1 and 2) have finite scores.
        assert!(scores[1].is_finite(), "Index 1 should have a finite score");
        assert!(scores[2].is_finite(), "Index 2 should have a finite score");

        // Ensure scores are positive.
        assert!(scores[1] > 0.0, "Score for index 1 should be > 0");
        assert!(scores[2] > 0.0, "Score for index 2 should be > 0");
    }

    #[test]
    fn test_operate_age_moea_survival() {
        // Create a first front (F₁) with 3 individuals and 2 objectives.
        let fitness_front1: Array2<f64> = array![[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]];
        // For simplicity, let genes be the same as fitness.
        let genes_front1 = fitness_front1.clone();
        // All individuals in the first front have rank 0.
        let rank_front1: Array1<usize> = Array1::from(vec![0, 0, 0]);

        let front1 = Population::new(genes_front1, fitness_front1, None, rank_front1);

        // Create a second front with 2 individuals (e.g., lower-ranked solutions).
        let fitness_front2: Array2<f64> = array![[0.8, 0.2], [0.2, 0.8]];
        let genes_front2 = fitness_front2.clone();
        // Rank 1 for these individuals.
        let rank_front2: Array1<usize> = Array1::from(vec![1, 1]);

        let front2 = Population::new(genes_front2, fitness_front2, None, rank_front2);

        // Assemble the fronts into a vector.
        let mut fronts: Fronts = vec![front1, front2];

        // Set the desired number of survivors.
        // In this example, the first front (3 individuals) fits entirely,
        // and only 1 individual will be selected from the second (splitting) front.
        let n_survive = 4;

        // Create the AgeMoeaSurvival operator.
        let operator = AgeMoeaSurvival::new();
        let mut _rng = NoopRandomGenerator::new();
        // Call operate to select survivors.
        let survivors = operator.operate(&mut fronts, n_survive, &mut _rng);

        // Check that the final merged population contains the desired number of survivors.
        assert_eq!(
            survivors.len(),
            n_survive,
            "Expected {} survivors",
            n_survive
        );

        // Verify that the survival scores have been set.
        assert!(
            survivors.survival_score.is_some(),
            "Survival scores should be assigned in the survivors population"
        );
    }
}
