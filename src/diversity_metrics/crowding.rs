use std::f64::INFINITY;

use numpy::ndarray::Array1;

use crate::genetic::PopulationFitness;

/// Computes the crowding distance for a given Pareto population_fitness.
///
/// # Parameters:
/// - `population_fitness`: A 2D array where each row represents an individual's fitness values.
///
/// # Returns:
/// - A 1D array of crowding distances for each individual in the population_fitness.
pub fn crowding_distance(population_fitness: &PopulationFitness) -> Array1<f64> {
    let num_individuals = population_fitness.shape()[0];
    let num_objectives = population_fitness.shape()[1];

    // Handle edge cases
    if num_individuals <= 2 {
        let mut distances = Array1::zeros(num_individuals);
        if num_individuals > 0 {
            distances[0] = INFINITY; // Boundary individuals
        }
        if num_individuals > 1 {
            distances[num_individuals - 1] = INFINITY;
        }
        return distances;
    }

    // Initialize distances to zero
    let mut distances = Array1::zeros(num_individuals);

    // Iterate over each objective
    for obj_idx in 0..num_objectives {
        // Extract the column for the current objective
        let objective_values = population_fitness.column(obj_idx);

        // Sort indices based on the objective values
        let mut sorted_indices: Vec<usize> = (0..num_individuals).collect();
        sorted_indices.sort_by(|&i, &j| {
            objective_values[i]
                .partial_cmp(&objective_values[j])
                .unwrap()
        });

        // Assign INFINITY to border. TODO: Not sure if worst should have infinity
        distances[sorted_indices[0]] = INFINITY;
        distances[sorted_indices[num_individuals - 1]] = INFINITY;

        // Get min and max values for normalization
        let min_value = objective_values[sorted_indices[0]];
        let max_value = objective_values[sorted_indices[num_individuals - 1]];
        let range = max_value - min_value;

        if range != 0.0 {
            // Calculate crowding distances for intermediate individuals
            for k in 1..(num_individuals - 1) {
                let next = objective_values[sorted_indices[k + 1]];
                let prev = objective_values[sorted_indices[k - 1]];
                distances[sorted_indices[k]] += (next - prev) / range;
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::array;

    #[test]
    /// Tests the calculation of crowding distances for a given population fitness matrix.
    ///
    /// The test defines a `population_fitness` matrix for four individuals:
    ///     [1.0, 2.0]
    ///     [2.0, 1.0]
    ///     [1.5, 1.5]
    ///     [3.0, 3.0]
    ///
    /// For each objective, the ideal (minimum) and nadir (maximum) values are computed.
    /// Then, for interior solutions, the crowding distance is calculated based on the normalized difference
    /// between the neighboring solutions. According to the classical NSGA-II method (which sums the contributions),
    /// the expected crowding distances are as follows:
    ///   - Corner individuals (first, second, and fourth) are assigned INFINITY.
    ///   - The middle individual [1.5, 1.5] has a crowding distance of 1.0 (since its contribution from each objective sums to 1.0).
    ///
    /// The test asserts that the computed crowding distances match the expected values:
    ///     expected = [INFINITY, INFINITY, 1.0, INFINITY]
    fn test_crowding_distance() {
        // Define a population_fitness with multiple individuals.
        let population_fitness = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]];

        // Compute crowding distances.
        let distances = crowding_distance(&population_fitness);

        // Expected distances: the corner individuals are assigned INFINITY and the middle individual sums to 1.0.
        let expected = array![
            std::f64::INFINITY,
            std::f64::INFINITY,
            1.0,
            std::f64::INFINITY
        ];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_single_individual() {
        // Define a population_fitness with a single individual
        let population_fitness = array![[1.0, 2.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: single individual has INFINITY
        let expected = array![INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_two_individuals() {
        // Define a population_fitness with two individuals
        let population_fitness = array![[1.0, 2.0], [2.0, 1.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: both are corner individuals with INFINITY
        let expected = array![INFINITY, INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }

    #[test]
    fn test_crowding_distance_same_fitness_values() {
        // Define a population_fitness where all individuals have the same fitness values
        let population_fitness = array![[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected: all distances should remain zero except for the first
        let expected = array![INFINITY, 0.0, 0.0, 0.0, INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }
}
