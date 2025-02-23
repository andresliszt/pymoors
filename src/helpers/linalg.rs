use ndarray::linalg::general_mat_mul;
use ndarray::{Array2, ArrayView1, Axis};

use crate::genetic::{PopulationFitness, PopulationGenes};

/// Computes the Lₚ norm of a array view vector.
pub fn lp_norm_arrayview(x: &ArrayView1<f64>, p: f64) -> f64 {
    x.iter()
        .map(|&val| val.abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p)
}

/// Computes the cross squared Euclidean distance matrix between `data` and `reference`
/// using matrix algebra.
///
/// For data of shape (n, d) and reference of shape (m, d), returns an (n x m) matrix
/// where the (i,j) element is the squared Euclidean distance between the i-th row of data
/// and the j-th row of reference.
pub fn cross_euclidean_distances(
    data: &PopulationGenes,
    reference: &PopulationGenes,
) -> PopulationGenes {
    // Compute the squared norms for data and reference.
    let data_norms = data.map_axis(Axis(1), |row| row.dot(&row));
    let ref_norms = reference.map_axis(Axis(1), |row| row.dot(&row));

    let data_norms_col = data_norms.insert_axis(Axis(1)); // shape (n, 1)
    let ref_norms_row = ref_norms.insert_axis(Axis(0)); // shape (1, m)
    println!("DATA_NORMS_COLS {} ", data_norms_col);
    println!("DATA_REFS_COLS {} ", ref_norms_row);
    println!("DATA {} ", data);
    println!("REF {} ", ref_norms_row);
    let n = data.nrows();
    let m = reference.nrows();
    let mut dot: PopulationGenes = PopulationGenes::zeros((n, m));
    general_mat_mul(1.0, data, &reference.t(), 0.0, &mut dot);
    println!("DOT {} ", dot);
    // Use the formula: d² = ||x||² + ||y||² - 2 * (x dot y)
    let dists_sq = &data_norms_col + &ref_norms_row - 2.0 * dot;
    dists_sq.mapv(|x| if x < 0.0 { 0.0 } else { x.sqrt() })
}

pub fn cross_p_distances(
    data: &PopulationFitness,
    reference: &PopulationFitness,
    p: f64,
) -> Array2<f64> {
    // Expand dimensions so that `data` has shape (n, 1, d) and `reference` has shape (1, m, d)
    // Expand dimensions and convert to owned arrays so that subtraction is allowed.
    let data_expanded = data.view().insert_axis(Axis(1)).to_owned(); // shape (n, 1, d)
    let reference_expanded = reference.view().insert_axis(Axis(0)).to_owned(); // shape (1, m, d)

    // Compute the element-wise differences.
    let diff = data_expanded - reference_expanded;

    // Compute the sum of |x - y|^p along the feature dimension (axis 2)
    let dists_p = diff.mapv(|x| x.abs().powf(p)).sum_axis(Axis(2));
    dists_p
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_cross_euclidean_distances() {
        // Create sample data and reference arrays (each row is a point in 2D space).
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let reference = array![[0.0, 0.0], [2.0, 2.0]];

        // Expected squared Euclidean distances:
        // - Distance between [0,0] and [0,0]: 0²+0² = 0
        // - Distance between [0,0] and [2,2]: 2²+2² = 8
        // - Distance between [1,1] and [0,0]: 1²+1² = 2
        // - Distance between [1,1] and [2,2]: 1²+1² = 2
        let expected = array![[0.0, 8.0], [2.0, 2.0]];

        let result = cross_euclidean_distances(&data, &reference);
        assert!(result.abs_diff_eq(&expected, 1e-6));
    }

    #[test]
    fn test_cross_p_distances() {
        // Create sample data and reference arrays.
        let data = array![[0.0, 0.0], [1.0, 1.0]];
        let reference = array![[0.0, 0.0], [2.0, 2.0]];

        // For p = 2, the function should return the sum of squared differences,
        // which is equivalent to the squared Euclidean distances.
        let expected_p2 = array![[0.0, 8.0], [2.0, 2.0]];
        let result_p2 = cross_p_distances(&data, &reference, 2.0);
        assert!(result_p2.abs_diff_eq(&expected_p2, 1e-6));

        // For p = 1, the function should return the Manhattan distances (without taking any root).
        // Manhattan distances:
        // - [0,0] vs [0,0]: |0-0| + |0-0| = 0
        // - [0,0] vs [2,2]: |0-2| + |0-2| = 4
        // - [1,1] vs [0,0]: |1-0| + |1-0| = 2
        // - [1,1] vs [2,2]: |1-2| + |1-2| = 2
        let expected_p1 = array![[0.0, 4.0], [2.0, 2.0]];
        let result_p1 = cross_p_distances(&data, &reference, 1.0);
        assert!(result_p1.abs_diff_eq(&expected_p1, 1e-6));
    }
}
