use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;

use numpy::ndarray::Array2;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    /// Parallel approach to remove exact duplicate rows:
    /// 1) Convert each row to `Vec<f64>`.
    /// 2) Split rows into chunks, remove duplicates locally in each chunk (thread-local).
    /// 3) Merge results to ensure global uniqueness.
    fn remove(&self, population: &PopulationGenes) -> PopulationGenes {
        let nrows = population.nrows();
        if nrows == 0 {
            // empty population
            return population.clone();
        }
        let ncols = population.ncols();

        // Extract all rows as Vec<Vec<f64>>
        let all_rows: Vec<Vec<f64>> = population.outer_iter().map(|row| row.to_vec()).collect();

        let chunk_size = 100;

        // 1) In parallel, remove duplicates locally in each chunk
        //    Return a Vec of "locally deduplicated" rows per chunk
        let local_deduped: Vec<Vec<Vec<f64>>> = all_rows
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut local_set = HashSet::new();
                let mut local_vec = Vec::new();
                for row in chunk {
                    let row_hash: Vec<OrderedFloat<f64>> =
                        row.iter().map(|&val| OrderedFloat(val)).collect();
                    if local_set.insert(row_hash) {
                        local_vec.push(row.clone());
                    }
                }
                local_vec
            })
            .collect();

        // 2) Merge results into a global set
        let mut global_set = HashSet::new();
        let mut final_rows = Vec::new();
        for chunk_rows in local_deduped {
            for row in chunk_rows {
                let row_hash: Vec<OrderedFloat<f64>> =
                    row.iter().map(|&val| OrderedFloat(val)).collect();
                if global_set.insert(row_hash) {
                    final_rows.push(row);
                }
            }
        }

        // 3) Build final deduplicated matrix
        let data_flat: Vec<f64> = final_rows.into_iter().flatten().collect();
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

    /// Builds the NxN distance matrix in parallel using a 1D buffer,
    /// computing only the upper-triangular part (i ≤ j) and then mirroring it.
    /// This avoids redundant work because the distance matrix is symmetric.
    /// Complexity: O(N^2) on the number of rows, plus cost for dot products.
    fn pairwise_distance_parallel_1d(&self, population: &PopulationGenes) -> Array2<f64> {
        let n = population.nrows();
        let d = population.ncols();
        if n == 0 {
            return Array2::<f64>::zeros((0, 0));
        }

        // 1) Compute row_sums[i] = sum of squares of row i in parallel.
        let row_sums: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row_i = population.row(i);
                row_i.iter().map(|&x| x * x).sum::<f64>()
            })
            .collect();

        // Wrap row_sums in an Arc so it can be shared among threads.
        let row_sums = Arc::new(row_sums);

        // Get the flat slice of population data.
        let flat_pop = population
            .as_slice()
            .expect("Population must be contiguous data");

        // Wrap flat_pop in an Arc as well.
        let flat_pop = Arc::new(flat_pop);

        // 2) Compute the upper-triangular distances (i ≤ j) in parallel.
        let upper: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                // Clone the Arcs into the closure.
                let row_sums = row_sums.clone();
                let flat_pop = flat_pop.clone();
                (i..n)
                    .map(move |j| {
                        if i == j {
                            (i, j, 0.0)
                        } else {
                            let sum_i = row_sums[i];
                            let sum_j = row_sums[j];
                            // Get slices for rows i and j.
                            let row_i = &flat_pop[i * d..i * d + d];
                            let row_j = &flat_pop[j * d..j * d + d];
                            let dot_ij = row_i
                                .iter()
                                .zip(row_j.iter())
                                .map(|(&a, &b)| a * b)
                                .sum::<f64>();
                            let dist2 = sum_i + sum_j - 2.0 * dot_ij;
                            let distance = dist2.max(0.0).sqrt();
                            (i, j, distance)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        // 3) Create a full vector for the symmetric matrix and fill it.
        let mut full = vec![0.0; n * n];
        for (i, j, distance) in upper {
            full[i * n + j] = distance;
            full[j * n + i] = distance; // Mirror the upper triangle.
        }

        Array2::from_shape_vec((n, n), full)
            .expect("Failed to create distance matrix from 1D buffer")
    }
}

impl PopulationCleaner for CloseDuplicatesCleaner {
    /// Removes rows within `epsilon` distance of each other, keeping one representative.
    /// 1) Compute NxN distance matrix in parallel with the 1D approach.
    /// 2) Mark all points near the chosen row as visited, BFS-like in single thread.
    fn remove(&self, population: &PopulationGenes) -> PopulationGenes {
        let n = population.nrows();
        if n == 0 {
            return population.clone();
        }
        let ncols = population.ncols();

        // 1) Build distance matrix in parallel
        let dist_matrix = self.pairwise_distance_parallel_1d(population);

        // 2) BFS-like pass: for each row i not visited, keep it and mark neighbors within epsilon
        let mut visited = vec![false; n];
        let mut retained_indices = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }
            retained_indices.push(i);

            // Mark neighbors
            for j in 0..n {
                if dist_matrix[[i, j]] < self.epsilon {
                    visited[j] = true;
                }
            }
        }

        // 3) Build final array from retained rows
        let retained_rows: Vec<_> = retained_indices
            .iter()
            .map(|&idx| population.row(idx).to_vec())
            .collect();

        let data: Vec<f64> = retained_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data.len() / ncols, ncols), data)
            .expect("Failed to create final close-duplicates Array2")
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
