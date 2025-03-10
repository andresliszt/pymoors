use numpy::ndarray::{Array1, Axis};

use crate::genetic::{PopulationFitness, Fronts};

// ---------------------------------------------------------------------------
// Auxiliary Functions for Distance and Ranking Computations
// ---------------------------------------------------------------------------

/// Computes the ideal point from a fitness matrix.
/// Each element of the returned array is the minimum value along the corresponding column.
pub fn get_ideal(population_fitness: &PopulationFitness) -> Array1<f64> {
    population_fitness.fold_axis(Axis(0), f64::INFINITY, |a, &b| a.min(b))
}

/// Computes the nadir point from a fitness matrix.
/// Each element of the returned array is the maximum value along the corresponding column.
pub fn get_nadir(population_fitness: &PopulationFitness) -> Array1<f64> {
    population_fitness.fold_axis(Axis(0), f64::NEG_INFINITY, |a, &b| a.max(b))
}

/// Computes the global ideal point from a slice of populations.
/// For each population, it computes its ideal point and then combines them by taking
/// the element-wise minimum across all populations.
pub fn get_ideal_from_fronts(fronts: &Fronts) -> Array1<f64> {
    // Initialize with the ideal of the first population.
    let mut global_ideal = get_ideal(&fronts[0].fitness);

    // Update the global ideal with each subsequent population's ideal.
    for pop in fronts.iter().skip(1) {
        let pop_ideal = get_ideal(&pop.fitness);
        global_ideal.zip_mut_with(&pop_ideal, |a, b| {
            *a = a.min(*b);
        });
    }

    global_ideal
}









// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_get_ideal() {
        // Example fitness matrix (3 solutions, 2 objectives)
        let fitness = array![[1.0, 4.0], [2.0, 3.0], [0.5, 5.0]];
        let ideal = get_ideal(&fitness);
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
}
