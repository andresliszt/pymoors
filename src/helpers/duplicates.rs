use std::collections::HashSet;
use std::fmt::Debug;

use numpy::ndarray::Axis;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;

use crate::genetic::PopulationGenes;

/// A trait for removing duplicates (exact or close) from a population.
pub trait PopulationCleaner: Debug {
    fn remove(&self, population: &PopulationGenes) -> PopulationGenes;
}

// -----------------------------------------------------------------------------
// EXACT DUPLICATES CLEANER (PARALLEL)
// -----------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct ExactDuplicatesCleaner;

impl ExactDuplicatesCleaner {
    pub fn new() -> Self {
        Self
    }
}

impl PopulationCleaner for ExactDuplicatesCleaner {
    fn remove(&self, population: &PopulationGenes) -> PopulationGenes {
        if population.is_empty() {
            return population.clone();
        }

        let ncols = population.ncols();
        let mut global_set: HashSet<Vec<OrderedFloat<f64>>> = HashSet::new();
        let mut deduped_rows = Vec::new();

        // Iterate through each row of the matrix
        for row in population.outer_iter() {
            // Convert the row into a vector of OrderedFloat to allow for proper hashing/comparison
            let row_hash: Vec<OrderedFloat<f64>> = row.iter().map(|&x| OrderedFloat(x)).collect();
            if global_set.insert(row_hash) {
                deduped_rows.push(row.to_vec());
            }
        }

        // Flatten the deduplicated rows into a single vector and rebuild the matrix
        let data_flat: Vec<f64> = deduped_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data_flat.len() / ncols, ncols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}

// -----------------------------------------------------------------------------
//  CLOSE DUPLICATES CLEANER (PARALLEL DISTANCE)
// -----------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct CloseDuplicatesCleaner {
    pub epsilon: f64,
}

impl CloseDuplicatesCleaner {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }

    /// Computes the full pairwise Euclidean distance matrix among the rows of `data`.
    ///
    /// For a dataset of shape (n, d) (n points in d dimensions), this function returns
    /// an (n x n) matrix where the (i,j) element is the Euclidean distance between row i and row j.
    fn pairwise_distances(&self, data: &PopulationGenes) -> PopulationGenes {
        // Compute the squared L2 norm for each row.
        let norms = data.map_axis(Axis(1), |row| row.dot(&row));

        // Reshape norms into a column vector (n x 1) and a row vector (1 x n)
        let norms_col = norms.clone().insert_axis(Axis(1)); // shape (n, 1)
        let norms_row = norms.insert_axis(Axis(0)); // shape (1, n)

        // Compute the dot product matrix: data.dot(data.T) yields an (n x n) matrix.
        let dot = data.dot(&data.t());

        // Using broadcasting, compute the squared distances:
        // d^2(i,j) = norms_col[i] + norms_row[j] - 2 * dot(i,j)
        let mut dists_sq = &norms_col + &norms_row - 2.0 * dot;

        // Due to numerical precision, some entries might be slightly negative; clamp them to zero.
        dists_sq.mapv_inplace(|x| if x < 0.0 { 0.0 } else { x });

        // Take the square root of every element to obtain the Euclidean distances.
        dists_sq.mapv(|x| x.sqrt())
    }
}

impl PopulationCleaner for CloseDuplicatesCleaner {
    fn remove(&self, population: &PopulationGenes) -> PopulationGenes {
        let n = population.nrows();
        // Compute the full pairwise distance matrix.
        let dists = self.pairwise_distances(population);
        // Create a boolean vector to mark rows to keep (true means keep)
        let mut keep = vec![true; n];

        // For each row, if it is marked to be kept, mark all later rows
        // that are within the epsilon distance as duplicates (i.e. not kept).
        for i in 0..n {
            if !keep[i] {
                continue;
            }
            // Only check subsequent rows to avoid duplicate comparisons.
            for j in (i + 1)..n {
                if dists[(i, j)] < self.epsilon {
                    keep[j] = false;
                }
            }
        }
        // Collect rows that are marked to be kept.
        let kept_rows: Vec<_> = population
            .outer_iter()
            .enumerate()
            .filter_map(|(i, row)| if keep[i] { Some(row.to_owned()) } else { None })
            .collect();

        // Build a new Array2 from the kept rows.
        let num_kept = kept_rows.len();
        let num_cols = population.ncols();
        let mut result = PopulationGenes::zeros((num_kept, num_cols));
        for (i, row) in kept_rows.into_iter().enumerate() {
            result.row_mut(i).assign(&row);
        }
        result
    }
}
// -----------------------------------------------------------------------------
// PYTHON-EXPOSED CLASSES, KEEPING NAMES
// -----------------------------------------------------------------------------

/// A Python class that encapsulates our parallel `ExactDuplicatesCleaner`.
#[pyclass(name = "ExactDuplicatesCleaner")]
#[derive(Clone, Debug)]
pub struct PyExactDuplicatesCleaner {
    pub inner: ExactDuplicatesCleaner,
}

#[pymethods]
impl PyExactDuplicatesCleaner {
    #[new]
    fn new() -> Self {
        let dup_cleaner = ExactDuplicatesCleaner::new();
        Self { inner: dup_cleaner }
    }
}

/// A Python class that encapsulates our parallel `CloseDuplicatesCleaner`.
#[pyclass(name = "CloseDuplicatesCleaner")]
#[derive(Clone, Debug)]
pub struct PyCloseDuplicatesCleaner {
    pub inner: CloseDuplicatesCleaner,
}

#[pymethods]
impl PyCloseDuplicatesCleaner {
    #[new]
    fn new(epsilon: f64) -> Self {
        let dup_cleaner = CloseDuplicatesCleaner::new(epsilon);
        Self { inner: dup_cleaner }
    }

    #[getter]
    fn epsilon(&self) -> f64 {
        self.inner.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::PopulationGenes;

    #[test]
    fn test_exact_duplicates_cleaner_removes_duplicates() {
        // Create a PopulationGenes with some repeated rows:
        //
        //  row 0: [1.0, 2.0, 3.0]
        //  row 1: [4.0, 5.0, 6.0]
        //  row 2: [1.0, 2.0, 3.0]   (duplicate of row 0)
        //  row 3: [7.0, 8.0, 9.0]
        //  row 4: [4.0, 5.0, 6.0]   (duplicate of row 1)

        let raw_data = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            1.0, 2.0, 3.0, // row 2
            7.0, 8.0, 9.0, // row 3
            4.0, 5.0, 6.0, // row 4
        ];

        let population =
            PopulationGenes::from_shape_vec((5, 3), raw_data).expect("Failed to create test array");

        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population);

        // We should end up with rows 0, 1, 3 as unique ones.
        // => 3 rows total.
        assert_eq!(cleaned.nrows(), 3);
        assert_eq!(cleaned.ncols(), 3);

        // Check that each row matches the expected values (and in the order of appearance)
        assert_eq!(cleaned.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cleaned.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(cleaned.row(2).to_vec(), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_exact_duplicates_cleaner_no_duplicates() {
        // Case where there are no duplicates at all
        let raw_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let population =
            PopulationGenes::from_shape_vec((3, 3), raw_data).expect("Failed to create test array");

        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population);

        // Everything should remain the same
        assert_eq!(cleaned.nrows(), 3);
        assert_eq!(cleaned.ncols(), 3);
        assert_eq!(cleaned, population);
    }

    #[test]
    fn test_close_duplicates_cleaner_small_epsilon() {
        // Create rows that differ by a small amount
        //
        // With epsilon = 0.01, these rows may be close but not close enough
        // so likely won't be merged.

        // Distance between row 1 and row 0: 0.009999999999988346
        // Distance between row 1 and row 2: 0.02236067977497184
        // Distance between row 2 and row 1: 0.02236067977497184
        // Distance between row 0 and row 1: 0.009999999999988346
        // Distance between row 0 and row 2: 0.02000000000006551
        // Distance between row 2 and row 0: 0.02000000000006551

        let raw_data = vec![1.0, 2.0, 3.0, 1.01, 2.0, 3.0, 1.0, 2.02, 3.0];

        let population =
            PopulationGenes::from_shape_vec((3, 3), raw_data).expect("Failed to create test array");

        // Very small epsilon
        let cleaner = CloseDuplicatesCleaner::new(0.0001);
        let cleaned = cleaner.remove(&population);
        // No removal at all
        assert_eq!(cleaned.nrows(), 3);
    }

    #[test]
    fn test_close_duplicates_cleaner_larger_epsilon() {
        // Similar rows but with epsilon = 0.05
        // and differences of about 0.02 => they should be considered "close duplicates."

        let raw_data = vec![1.0, 2.0, 3.0, 1.01, 2.02, 3.0, 10.0, 10.0, 10.0];

        let population =
            PopulationGenes::from_shape_vec((3, 3), raw_data).expect("Failed to create test array");

        let cleaner = CloseDuplicatesCleaner::new(0.05);
        let cleaned = cleaner.remove(&population);

        // Rows 0 and 1 should be within ~0.022... of each other, which is < 0.05,
        // => they will merge, leaving only one representative among them
        // => total 2 rows remain
        assert_eq!(cleaned.nrows(), 2);

        // The third row (10,10,10) is far from (1,2,3), so it remains as well.
    }
}
