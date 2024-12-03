use ndarray::{Array1, Axis};

use crate::genetic::{FitnessValue, Population};
use crate::non_dominated_sorting::{crowding_distance, fast_non_dominated_sorting};
use crate::operators::GeneticOperator;

impl GeneticOperator for RankCrowdSurvival {
    fn name(&self) -> String {
        "BitFlipMutation".to_string()
    }
}

#[derive(Clone, Debug)]
pub struct RankCrowdSurvival;

impl RankCrowdSurvival {
    /// Performs survival selection based on rank and crowding distance.
    pub fn operate<Dna, F>(&self, population: &mut Population<Dna, F>, n_survive: usize)
    where
        Dna: Clone,
        F: FitnessValue + Into<f64>,
    {
        // Initialize rank and crowding_distance if they are None
        let pop_size = population.genes.nrows();
        if population.rank.is_none() {
            population.rank = Some(Array1::zeros(pop_size));
        }
        if population.crowding_distance.is_none() {
            population.crowding_distance = Some(Array1::zeros(pop_size));
        }

        // Perform non-dominated sorting
        let fronts = fast_non_dominated_sorting(&population.fitness);

        let mut survivors = Vec::new();

        // Keep track of the current front number (rank)
        let mut current_rank = 0;

        for front in fronts {
            if survivors.len() + front.len() <= n_survive {
                // Assign rank to individuals in the front
                {
                    let rank_view = population.rank.as_mut().unwrap();
                    rank_view.select(Axis(0), &front).fill(current_rank);
                }

                // Extract fitness of individuals in the front
                let front_fitness = population.fitness.select(Axis(0), &front).to_owned();

                // Calculate crowding distance for this front
                let distances = crowding_distance(&front_fitness);

                // Assign crowding distances to the individuals in the front
                {
                    let cd_view = population.crowding_distance.as_mut().unwrap();
                    cd_view.select(Axis(0), &front).assign(&distances);
                }

                // Add all individuals in this front to survivors
                survivors.extend(&front);
            } else {
                // Need to select individuals within this front
                let remaining = n_survive - survivors.len();

                // Assign rank to the individuals in this front
                {
                    let rank_view = population.rank.as_mut().unwrap();
                    rank_view.select(Axis(0), &front).fill(current_rank);
                }

                // Extract fitness of individuals in the front
                let front_fitness = population.fitness.select(Axis(0), &front).to_owned();

                // Calculate crowding distance for this front
                let distances = crowding_distance(&front_fitness);

                // Assign crowding distances to the individuals in the front
                {
                    let cd_view = population.crowding_distance.as_mut().unwrap();
                    cd_view.select(Axis(0), &front).assign(&distances);
                }

                // Create a vector of indices and their crowding distances
                let mut front_with_distances: Vec<usize> = front.clone();

                // Sort the front by crowding distance in descending order
                front_with_distances.sort_by(|&i, &j| {
                    population.crowding_distance.as_ref().unwrap()[j]
                        .partial_cmp(&population.crowding_distance.as_ref().unwrap()[i])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Select the required number of individuals
                survivors.extend(front_with_distances.iter().take(remaining).cloned());
                break;
            }

            // Increment the rank for the next front
            current_rank += 1;
        }

        // Retain only the survivors in the population
        let survivor_indices = Array1::from_vec(survivors);
        population.select(&survivor_indices);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Population;
    use ndarray::Array2;

    #[test]
    fn test_rank_crowd_survival_full_fronts() {
        // Create a mock population
        let genes = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, // Individual 0
                2.0, 1.5, // Individual 1
                1.5, 1.5, // Individual 2
                2.0, 2.0, // Individual 3
                1.0, 1.0, // Individual 4
                2.5, 2.5, // Individual 5
            ],
        )
        .unwrap();

        let fitness = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 4.0, // Individual 0
                2.0, 3.0, // Individual 1
                3.0, 2.0, // Individual 2
                4.0, 1.0, // Individual 3
                1.0, 1.0, // Individual 4
                5.0, 5.0, // Individual 5
            ],
        )
        .unwrap();

        let mut population = Population::new(genes, fitness);

        let n_survive = 4;

        let survival_operator = RankCrowdSurvival;

        survival_operator.operate(&mut population, n_survive);

        // Check that the population size is correct
        assert_eq!(population.genes.nrows(), n_survive);

        // Optional: Check that ranks and crowding distances are assigned
        assert!(population.rank.is_some());
        assert!(population.crowding_distance.is_some());

        // Check that ranks are assigned correctly
        for rank in population.rank.as_ref().unwrap().iter() {
            assert!(*rank >= 1);
        }

        // Check that crowding distances are non-negative
        for cd in population.crowding_distance.as_ref().unwrap().iter() {
            assert!(*cd >= 0.0);
        }
    }

    #[test]
    fn test_rank_crowd_survival_partial_front() {
        // Create a mock population
        let genes = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 2.0, // Individual 0
                2.0, 1.5, // Individual 1
                1.5, 1.5, // Individual 2
                2.0, 2.0, // Individual 3
                1.0, 1.0, // Individual 4
            ],
        )
        .unwrap();

        let fitness = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 4.0, // Individual 0
                2.0, 3.0, // Individual 1
                3.0, 2.0, // Individual 2
                4.0, 1.0, // Individual 3
                1.0, 1.0, // Individual 4
            ],
        )
        .unwrap();

        let mut population = Population::new(genes, fitness);

        let n_survive = 3;

        let survival_operator = RankCrowdSurvival;

        survival_operator.operate(&mut population, n_survive);

        // Check that the population size is correct
        assert_eq!(population.genes.nrows(), n_survive);

        // Check that ranks and crowding distances are assigned
        assert!(population.rank.is_some());
        assert!(population.crowding_distance.is_some());

        // Check that ranks are assigned correctly
        for rank in population.rank.as_ref().unwrap().iter() {
            assert!(*rank >= 0);
        }

        // Check that crowding distances are non-negative
        for cd in population.crowding_distance.as_ref().unwrap().iter() {
            assert!(*cd >= 0.0);
        }
    }
}
