use std::f64::INFINITY;
use std::fmt::Debug;

use ndarray::Array1;

use crate::genetic::PopulationFitness;
use crate::operators::{FrontContext, GeneticOperator, SurvivalOperator};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
pub struct RankCrowdingSurvival;

impl GeneticOperator for RankCrowdingSurvival {
    fn name(&self) -> String {
        "RankCrowdingSurvival".to_string()
    }
}

impl RankCrowdingSurvival {
    pub fn new() -> Self {
        Self {}
    }
}

impl SurvivalOperator for RankCrowdingSurvival {
    fn survival_score(
        &self,
        front_fitness: &PopulationFitness,
        _context: FrontContext,
        _rng: &mut dyn RandomGenerator,
    ) -> Array1<f64> {
        crowding_distance(&front_fitness)
    }
}

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
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use numpy::ndarray::{array, Array2};

    use crate::genetic::{Fronts, Population};
    use crate::random::NoopRandomGenerator;

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

    #[test]
    fn test_survival_selection_all_survive_single_front() {
        // All individuals can survive without partial selection.
        let genes = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let fitness = array![[0.1], [0.2], [0.3]];
        let constraints: Option<Array2<f64>> = None;
        let rank = array![0, 0, 0];

        let population = Population::new(
            genes.clone(),
            fitness.clone(),
            constraints.clone(),
            rank.clone(),
        );
        let mut fronts: Fronts = vec![population];

        let n_survive = 3;
        let selector = RankCrowdingSurvival;
        assert_eq!(selector.name(), "RankCrowdingSurvival");
        let mut _rng = NoopRandomGenerator::new();
        let new_population = selector.operate(&mut fronts, n_survive, &mut _rng);

        // All three should survive unchanged
        assert_eq!(new_population.len(), 3);
        assert_eq!(new_population.genes, genes);
        assert_eq!(new_population.fitness, fitness);
    }

    #[test]
    fn test_survival_selection_multiple_fronts() {
        /*
        Test for survival selection with multiple fronts in NSGA-II (classic approach).

        Scenario:
          - Front 1 contains 2 individuals (first front, rank = 0). Since n_survive = 4,
            all individuals from Front 1 are selected.
          - Front 2 contains 4 individuals (second front, rank = 1), but only 2 more individuals
            are needed to reach a total of 4 survivors.

        Classical NSGA-II crowding distance calculation (for a single objective) assigns
        an infinite crowding distance to the extreme individuals (those with minimum and maximum fitness values).
        For Front 2 with fitness values:
             [0.3], [0.4], [0.5], [0.6]
        the extreme individuals (with fitness 0.3 and 0.6) get a crowding distance of infinity,
        while the interior ones get finite values.
        Hence, when selecting 2 individuals from Front 2, the algorithm should select the two extremes:
             - The individual with fitness [0.3] (index 0)
             - The individual with fitness [0.6] (index 3)

        Expected final population:
          - From Front 1 (all individuals): genes [[0.0, 1.0], [2.0, 3.0]] with fitness [[0.1], [0.2]]
          - From Front 2 (selected extremes): genes [[4.0, 5.0], [10.0, 11.0]] with fitness [[0.3], [0.6]]
        */

        // Front 1: 2 individuals (first front, rank 0)
        let front1_genes = array![[0.0, 1.0], [2.0, 3.0]];
        let front1_fitness = array![[0.1], [0.2]];
        let front1_constraints: Option<Array2<f64>> = None;
        let front1_rank = array![0, 0];

        // Front 2: 4 individuals (second front, rank 1)
        // With fitness values arranged in increasing order: 0.3, 0.4, 0.5, 0.6.
        // In classical crowding distance, individuals with fitness 0.3 (first) and 0.6 (last) get INFINITY.
        let front2_genes = array![[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]];
        let front2_fitness = array![[0.3], [0.4], [0.5], [0.6]];
        let front2_constraints: Option<Array2<f64>> = None;
        let front2_rank = array![1, 1, 1, 1];

        let population1 = Population::new(
            front1_genes,
            front1_fitness,
            front1_constraints,
            front1_rank,
        );

        let population2 = Population::new(
            front2_genes,
            front2_fitness,
            front2_constraints,
            front2_rank,
        );

        let mut fronts: Vec<Population> = vec![population1, population2];

        let n_survive = 4; // We want 4 individuals total.

        // Use the survival operator (assumed to be RankCrowdingSurvival in NSGA-II classic mode).
        let selector = RankCrowdingSurvival;
        let mut _rng = NoopRandomGenerator::new();
        let new_population = selector.operate(&mut fronts, n_survive, &mut _rng);

        // The final population must have 4 individuals.
        assert_eq!(new_population.len(), n_survive);

        // Expected outcome:
        // - From Front 1, all individuals are selected: indices [0, 1] with genes [[0.0,1.0], [2.0,3.0]].
        // - From Front 2, only 2 individuals are selected based on crowding distance.
        //   In classical NSGA-II, the extreme individuals (with lowest and highest fitness) are selected.
        //   Therefore, from Front 2, the individuals at index 0 (fitness 0.3) and index 3 (fitness 0.6) are selected.
        //
        // Thus, the final population should have:
        //   IndividualGenes: [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]]
        //   Fitness: [[0.1], [0.2], [0.3], [0.6]]
        let expected_genes = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]];
        let expected_fitness = array![[0.1], [0.2], [0.3], [0.6]];
        assert_eq!(new_population.genes, expected_genes);
        assert_eq!(new_population.fitness, expected_fitness);
    }
}
