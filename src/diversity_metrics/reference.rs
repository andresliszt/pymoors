use crate::genetic::PopulationFitness;
use numpy::ndarray::{Array1, Array2, Axis};
use std::cmp::Ordering; // Assume PopulationFitness is defined as Array2<f64> elsewhere.

// ---------------------------------------------------------------------------
// Auxiliary Functions for Distance and Ranking Computations
// ---------------------------------------------------------------------------

/// Computes the ideal point from a fitness matrix.
/// Each element of the returned array is the minimum value along the corresponding column.
pub fn get_nideal(population_fitness: &PopulationFitness) -> Array1<f64> {
    population_fitness.fold_axis(Axis(0), f64::INFINITY, |a, &b| a.min(b))
}

/// Computes the nadir point from a fitness matrix.
/// Each element of the returned array is the maximum value along the corresponding column.
pub fn get_nadir(population_fitness: &PopulationFitness) -> Array1<f64> {
    population_fitness.fold_axis(Axis(0), f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Computes the weighted, normalized Euclidean distance between two objective vectors `f1` and `f2`.
/// Normalization is performed using the provided ideal (`nideal`) and nadir (`nadir`) points.
/// If for any objective the range (nadir - nideal) is zero, the normalized difference is set to 0.0.
pub fn weighted_normalized_euclidean_distance(
    f1: &Array1<f64>,
    f2: &Array1<f64>,
    weights: &Array1<f64>,
    nideal: &Array1<f64>,
    nadir: &Array1<f64>,
) -> f64 {
    let diff = f1 - f2;
    let mut sum_sq = 0.0;
    for j in 0..diff.len() {
        let range = nadir[j] - nideal[j];
        let normalized_diff = if range == 0.0 { 0.0 } else { diff[j] / range };
        sum_sq += weights[j] * normalized_diff * normalized_diff;
    }
    sum_sq.sqrt()
}

/// Computes the ranking for a single column of distances.
/// Given a slice of distances, it returns a vector of the same length where each element
/// represents the order (i.e., rank) of the corresponding solution in the sorted order (ascending).
///
/// For example, if the distances in a column are [0.3, 0.1, 0.5],
/// the sorted order is [0.1, 0.3, 0.5] and the corresponding ranks are:
/// - The solution with 0.1 gets rank 0,
/// - The solution with 0.3 gets rank 1,
/// - The solution with 0.5 gets rank 2.
fn compute_rank_for_column(distances: &[f64]) -> Vec<usize> {
    let n = distances.len();
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| {
        distances[i]
            .partial_cmp(&distances[j])
            .unwrap_or(Ordering::Equal)
    });
    let mut ranks = vec![0; n];
    for (order, &i) in indices.iter().enumerate() {
        ranks[i] = order;
    }
    ranks
}

/// Given a matrix of distances (with shape (n, P), where n is the number of solutions
/// in the front and P is the number of reference points), computes the ranking for each solution
fn ranking_by_distances(distances: &Array2<f64>) -> Array1<usize> {
    let (n, p) = distances.dim();
    let mut rank_matrix: Vec<Vec<usize>> = Vec::with_capacity(p);
    for k in 0..p {
        let col = distances.index_axis(Axis(1), k);
        let col_vec = col.to_vec();
        let ranks = compute_rank_for_column(&col_vec);
        rank_matrix.push(ranks);
    }
    let mut final_ranks = vec![usize::MAX; n];
    for i in 0..n {
        for k in 0..p {
            final_ranks[i] = final_ranks[i].min(rank_matrix[k][i]);
        }
    }
    Array1::from(final_ranks)
}

/// Computes the weighted distance matrix for a front.
///
/// # Parameters
/// - `fitness`: A PopulationFitness (Array2<f64>) representing the objectives of each solution in the front.
/// - `reference`: A matrix of reference points (Array2<f64>). In the RNSGA2 survival process, this function is used
///   with the reference points to compute the distance between each solution and each reference point.
///
/// # Process
/// 1. Determines the number of solutions (n) and the number of objectives (m).
/// 2. Computes the ideal and nadir points using `get_nideal` and `get_nadir`.
/// 3. Sets the weights as 1/m for each objective.
/// 4. Computes the weighted, normalized Euclidean distance between each solution in the fitness matrix
///    and each reference point.
///
/// # Returns
/// Returns a distance matrix (Array2<f64>) with dimensions (n, r), where r is the number of rows of the reference matrix.
pub fn weighted_distance_matrix(
    fitness: &PopulationFitness,
    reference: &Array2<f64>,
) -> Array2<f64> {
    let (n, m) = fitness.dim();
    let r = reference.nrows();
    let ideal = get_nideal(fitness);
    let nadir = get_nadir(fitness);
    // Set weights as 1/m for each objective.
    let weights = Array1::from_elem(m, 1.0 / (m as f64));
    let mut dist_matrix = Array2::<f64>::zeros((n, r));

    for i in 0..n {
        let f_i = fitness.row(i).to_owned();
        for j in 0..r {
            let r_j = reference.row(j).to_owned();
            let d = weighted_normalized_euclidean_distance(&f_i, &r_j, &weights, &ideal, &nadir);
            dist_matrix[[i, j]] = d;
        }
    }
    dist_matrix
}

/// Computes the ranking based on reference points.
/// It uses `weighted_distance_matrix` to obtain the distance matrix between each solution
/// in the front and the reference points, then calculates the final ranking with `ranking_by_distances`.
///
/// # Parameters
/// - `fitness`: A PopulationFitness matrix (Array2<f64>) containing the solutions of the front.
/// - `reference`: A matrix of reference points (Array2<f64>).
///
/// # Returns
/// Returns an Array1<f64> where each element represents the ranking for each solution.
/// This ranking is used as the diversity measure in the survival process of the RNSGA2 algorithm.
pub fn reference_points_rank_distance(
    fitness: &PopulationFitness,
    reference: &Array2<f64>,
) -> Array1<f64> {
    let distances = weighted_distance_matrix(fitness, reference);
    let ranks = ranking_by_distances(&distances);
    ranks.mapv(|r| r as f64)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_get_nideal() {
        // Example fitness matrix (3 solutions, 2 objectives)
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]];
        let ideal = get_nideal(&fitness);
        // For each objective, the expected minimum value:
        // First objective: min(1.0, 2.0, 0.5) = 0.5
        // Second objective: min(4.0, 3.0, 5.0) = 3.0
        assert_eq!(ideal, array![0.5, 3.0]);
    }

    #[test]
    fn test_get_nadir() {
        // Example fitness matrix (3 solutions, 2 objectives)
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]];
        let nadir = get_nadir(&fitness);
        // For each objective, the expected maximum value:
        // First objective: max(1.0, 2.0, 0.5) = 2.0
        // Second objective: max(4.0, 3.0, 5.0) = 5.0
        assert_eq!(nadir, array![2.0, 5.0]);
    }

    #[test]
    fn test_weighted_normalized_euclidean_distance() {
        let f1 = array![1.0, 2.0, 3.0];
        let f2 = array![2.0, 3.0, 4.0];
        // Assume 3 objectives; therefore, weights are 1/3 for each.
        let weights = Array1::from_elem(3, 1.0 / 3.0);
        // Define arbitrary ideal and nadir points for normalization:
        let ideal = array![0.0, 0.0, 0.0];
        let nadir = array![10.0, 10.0, 10.0];
        let distance = weighted_normalized_euclidean_distance(&f1, &f2, &weights, &ideal, &nadir);
        // The normalized differences will be (1/10, 1/10, 1/10)
        // The weights are (1/3, 1/3, 1/3)
        // Sum of squares: 3*(1/3)*(0.01) = 0.01, sqrt(0.01) ≈ 0.1.
        assert!((distance - 0.1).abs() < 1e-4);
    }

    #[test]
    fn test_compute_rank_for_column() {
        let distances = vec![0.3, 0.1, 0.5, 0.2];
        let ranks = compute_rank_for_column(&distances);
        // Sorted order: 0.1 (index 1), 0.2 (index 3), 0.3 (index 0), 0.5 (index 2)
        // Expected ranks: [2, 0, 3, 1]
        assert_eq!(ranks, vec![2, 0, 3, 1]);
    }

    #[test]
    fn test_ranking_by_distances() {
        // Example distance matrix (3 solutions, 2 reference points)
        let distances = array![[0.3, 0.4], [0.1, 0.2], [0.5, 0.6]];
        // For column 0: sorted order is [0.1 (index 1), 0.3 (index 0), 0.5 (index 2)] → ranks: [1, 0, 2]
        // For column 1: sorted order is [0.2 (index 1), 0.4 (index 0), 0.6 (index 2)] → ranks: [1, 0, 2]
        // Final ranking for each solution is the minimum across columns:
        // Solution 0: min(1,1)=1, Solution 1: min(0,0)=0, Solution 2: min(2,2)=2.
        let final_ranks = ranking_by_distances(&distances);
        assert_eq!(final_ranks, array![1, 0, 2]);
    }

    #[test]
    fn test_weighted_distance_matrix() {
        // Example fitness matrix (3 solutions, 2 objectives)
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]];
        // Reference matrix: for this test we define it with 2 reference points.
        let reference = array![[1.0, 4.0], [2.0, 3.0]];
        // Calculate the distance matrix: dimensions (n, r) = (3, 2)
        let dist_matrix = weighted_distance_matrix(&fitness, &reference);
        // For example, the distance between solution 0 and reference point 0:
        // Since fitness[0] == reference[0], the distance should be 0.
        assert!((dist_matrix[[0, 0]] - 0.0).abs() < 1e-6);
        // Verify that the matrix has the expected dimensions.
        assert_eq!(dist_matrix.dim(), (3, 2));
    }

    #[test]
    fn test_reference_points_rank_distance() {
        // Example fitness matrix (3 solutions, 2 objectives)
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]];
        // Reference matrix: for this test we use the same matrix as fitness,
        // so each solution will have at least one identical reference point (distance 0).
        let reference = fitness.clone();
        let rank_vector = reference_points_rank_distance(&fitness, &reference);
        // Since each solution is compared with itself (distance 0) in at least one reference point,
        // the resulting ranking should be 0 for each solution.
        let expected = array![0.0, 0.0, 0.0];
        assert_eq!(rank_vector, expected);
    }
}
