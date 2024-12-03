use std::f64::INFINITY;

use ndarray::Array1;

use crate::genetic::{FitnessValue, PopulationFitness};

/// Computes the crowding distance for a given Pareto population_fitness.
///
/// # Parameters:
/// - `population_fitness`: A 2D array where each row represents an individual's fitness values.
///
/// # Returns:
/// - A 1D array of crowding distances for each individual in the population_fitness.
pub fn crowding_distance<F>(population_fitness: &PopulationFitness<F>) -> Array1<f64>
where
    F: FitnessValue + Into<f64>,
{
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
        let min_value: f64 = objective_values[sorted_indices[0]].into();
        let max_value: f64 = objective_values[sorted_indices[num_individuals - 1]].into();
        let range = max_value - min_value;

        if range != 0.0 {
            // Calculate crowding distances for intermediate individuals
            for k in 1..(num_individuals - 1) {
                let next = objective_values[sorted_indices[k + 1]].into();
                let prev = objective_values[sorted_indices[k - 1]].into();
                distances[sorted_indices[k]] += (next - prev) / range;
            }
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_crowding_distance_f64() {
        // Define a population_fitness with multiple individuals
        let population_fitness = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5], [3.0, 3.0]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);
        // Expected distances: corner individuals have INFINITY
        let expected = array![INFINITY, INFINITY, 0.5, INFINITY];

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

    #[test]
    fn test_crowding_distance_i32() {
        // Define a population_fitness with integer values
        let population_fitness = array![[1, 2], [2, 1], [1, 1], [3, 3]];

        // Compute crowding distances
        let distances = crowding_distance(&population_fitness);

        // Expected distances: corner individuals have INFINITY
        let expected = array![INFINITY, 0.5, 0.5, INFINITY];

        assert_eq!(distances.as_slice().unwrap(), expected.as_slice().unwrap());
    }
}
