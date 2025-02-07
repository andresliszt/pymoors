use ndarray::linalg::general_mat_mul;
use ndarray::Axis;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use std::collections::HashSet;
use std::fmt::Debug;

use crate::genetic::PopulationGenes;

/// A trait for removing duplicates (exact or close) from a population.
///
/// The `remove` method accepts an optional reference population.
/// If `None`, duplicates are computed within the population;
/// if provided, duplicates are determined by comparing each row in the population to all rows in the reference.
pub trait PopulationCleaner: Debug {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes;
}

/// EXACT DUPLICATES CLEANER
#[derive(Clone, Debug)]
pub struct ExactDuplicatesCleaner;

impl ExactDuplicatesCleaner {
    pub fn new() -> Self {
        Self
    }
}

impl PopulationCleaner for ExactDuplicatesCleaner {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        let ncols = population.ncols();
        let mut unique_rows: Vec<Vec<f64>> = Vec::new();
        // A HashSet to hold the hashable representation of rows.
        let mut seen: HashSet<Vec<OrderedFloat<f64>>> = HashSet::new();

        // If a reference is provided, first add its rows into the set.
        if let Some(ref_pop) = reference {
            for row in ref_pop.outer_iter() {
                let hash_row: Vec<OrderedFloat<f64>> =
                    row.iter().map(|&x| OrderedFloat(x)).collect();
                seen.insert(hash_row);
            }
        }

        // Iterate over the population rows.
        for row in population.outer_iter() {
            let hash_row: Vec<OrderedFloat<f64>> = row.iter().map(|&x| OrderedFloat(x)).collect();
            // Insert returns true if the row was not in the set.
            if seen.insert(hash_row) {
                unique_rows.push(row.to_vec());
            }
        }

        // Flatten the unique rows into a single vector.
        let data_flat: Vec<f64> = unique_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data_flat.len() / ncols, ncols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}

// -----------------------------------------------------------------------------
// CLOSE DUPLICATES CLEANER (PARALLEL, DISTANCE-BASED)
// -----------------------------------------------------------------------------
#[derive(Clone, Debug)]
pub struct CloseDuplicatesCleaner {
    pub epsilon: f64,
}

impl CloseDuplicatesCleaner {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
    /// Computes the cross squared Euclidean distance matrix between `data` and `reference`
    /// using matrix algebra.
    ///
    /// For data of shape (n, d) and reference of shape (m, d), returns an (n x m) matrix
    /// where the (i,j) element is the squared Euclidean distance between the i-th row of data
    /// and the j-th row of reference.
    fn cross_squared_distances(
        &self,
        data: &PopulationGenes,
        reference: &PopulationGenes,
    ) -> PopulationGenes {
        let n = data.nrows();
        let m = reference.nrows();
        // Compute the squared norms for data and reference.
        let data_norms = data.map_axis(Axis(1), |row| row.dot(&row));
        let ref_norms = reference.map_axis(Axis(1), |row| row.dot(&row));

        let data_norms_col = data_norms.insert_axis(Axis(1)); // shape (n, 1)
        let ref_norms_row = ref_norms.insert_axis(Axis(0)); // shape (1, m)

        let mut dot: PopulationGenes = PopulationGenes::zeros((n, m));
        general_mat_mul(1.0, data, &reference.t(), 0.0, &mut dot);

        // Use the formula: d² = ||x||² + ||y||² - 2 * (x dot y)
        let dists_sq = &data_norms_col + &ref_norms_row - 2.0 * dot;
        dists_sq
    }
}

impl PopulationCleaner for CloseDuplicatesCleaner {
    fn remove(
        &self,
        population: &PopulationGenes,
        reference: Option<&PopulationGenes>,
    ) -> PopulationGenes {
        let ref_array = reference.unwrap_or(population);
        let n = population.nrows();
        let num_cols = population.ncols();
        let dists_sq = self.cross_squared_distances(population, ref_array);
        let eps_sq = self.epsilon * self.epsilon;
        let mut keep = vec![true; n];
        // Note: when reference_array = population there is no need to loop through the full
        // array, just use the upper triangle matrix logic
        if let Some(ref_pop) = reference {
            // Mark each row in the population as duplicate if its distance to any row in reference is below eps_sq.
            for i in 0..n {
                for j in 0..ref_pop.nrows() {
                    if dists_sq[(i, j)] < eps_sq {
                        keep[i] = false;
                        break;
                    }
                }
            }
        } else {
            for i in 0..n {
                if !keep[i] {
                    continue;
                }
                for j in (i + 1)..n {
                    if dists_sq[(i, j)] < eps_sq {
                        keep[j] = false;
                    }
                }
            }
        }
        let kept_rows: Vec<_> = population
            .outer_iter()
            .enumerate()
            .filter_map(|(i, row)| if keep[i] { Some(row.to_owned()) } else { None })
            .collect();
        let data_flat: Vec<f64> = kept_rows.into_iter().flatten().collect();
        PopulationGenes::from_shape_vec((data_flat.len() / num_cols, num_cols), data_flat)
            .expect("Failed to create deduplicated Array2")
    }
}

// -----------------------------------------------------------------------------
// PYTHON-EXPOSED CLASSES, KEEPING NAMES
// -----------------------------------------------------------------------------

/// Cleaner that removes exact duplicate individuals from the population.
#[pyclass(name = "ExactDuplicatesCleaner")]
#[derive(Clone, Debug)]
pub struct PyExactDuplicatesCleaner {
    pub inner: ExactDuplicatesCleaner,
}

#[pymethods]
impl PyExactDuplicatesCleaner {
    #[new]
    fn new() -> Self {
        Self {
            inner: ExactDuplicatesCleaner::new(),
        }
    }
}

/// Cleaner that removes individuals that are close to each other based on a specified epsilon.
#[pyclass(name = "CloseDuplicatesCleaner")]
#[derive(Clone, Debug)]
pub struct PyCloseDuplicatesCleaner {
    pub inner: CloseDuplicatesCleaner,
}

#[pymethods]
impl PyCloseDuplicatesCleaner {
    #[new]
    fn new(epsilon: f64) -> Self {
        Self {
            inner: CloseDuplicatesCleaner::new(epsilon),
        }
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
    use ndarray::array;

    #[test]
    fn test_exact_duplicates_cleaner_removes_duplicates_without_reference() {
        let raw_data = vec![
            1.0, 2.0, 3.0, // row 0
            4.0, 5.0, 6.0, // row 1
            1.0, 2.0, 3.0, // row 2 (duplicate of row 0)
            7.0, 8.0, 9.0, // row 3
            4.0, 5.0, 6.0, // row 4 (duplicate of row 1)
        ];
        let population =
            PopulationGenes::from_shape_vec((5, 3), raw_data).expect("Failed to create test array");
        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population, None);
        assert_eq!(cleaned.nrows(), 3);
        assert_eq!(cleaned.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(cleaned.row(1).to_vec(), vec![4.0, 5.0, 6.0]);
        assert_eq!(cleaned.row(2).to_vec(), vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_exact_duplicates_cleaner_with_reference() {
        let population = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let reference = array![[1.0, 2.0, 3.0]]; // row 0 is in reference
        let cleaner = ExactDuplicatesCleaner::new();
        let cleaned = cleaner.remove(&population, Some(&reference));
        assert_eq!(cleaned.nrows(), 1);
        assert_eq!(cleaned.row(0).to_vec(), vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_close_duplicates_cleaner_without_reference() {
        let population = array![
            [1.0, 2.0, 3.0],
            [1.05, 2.05, 3.05], // very similar to row 0
            [4.0, 5.0, 6.0]
        ];
        let epsilon = 0.1;
        let cleaner = CloseDuplicatesCleaner::new(epsilon);
        let cleaned = cleaner.remove(&population, None);
        // Expect rows 0 and 2 remain.
        assert_eq!(cleaned.nrows(), 2);
    }

    #[test]
    fn test_close_duplicates_cleaner_with_reference() {
        let population = array![[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]];
        let reference = array![
            [1.01, 2.01, 3.01] // close to row 0 of population
        ];
        let epsilon = 0.05;
        let cleaner = CloseDuplicatesCleaner::new(epsilon);
        let cleaned = cleaner.remove(&population, Some(&reference));
        // Row 0 should be removed.
        assert_eq!(cleaned.nrows(), 1);
        assert_eq!(cleaned.row(0).to_vec(), vec![10.0, 10.0, 10.0]);
    }
}
