use std::fmt::Debug;

use crate::genetic::Individual;
use crate::operators::{DuelResult, GeneticOperator, SelectionOperator};
use rand::RngCore;

// TODO: Enable pressure. Currently is fixed in 2

#[derive(Clone, Debug)]
pub struct RankAndCrowdingSelection {}

impl RankAndCrowdingSelection {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for RankAndCrowdingSelection {
    fn name(&self) -> String {
        "RankAndCrowdingSelection".to_string()
    }
}

impl SelectionOperator for RankAndCrowdingSelection {
    /// Runs the tournament selection on the given population and returns a vector of winners.
    /// This example assumes binary tournaments (pressure = 2)
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut dyn RngCore,
    ) -> DuelResult {
        // get feasibility
        let p1_feasible = p1.is_feasible();
        let p2_feasible = p2.is_feasible();
        // get rank
        let p1_rank = p1.rank;
        let p2_rank = p2.rank;
        // get cd
        let p1_cd = p1.crowding_distance;
        let p2_cd = p2.crowding_distance;

        let winner = if p1_feasible && !p2_feasible {
            DuelResult::LeftWins
        } else if p2_feasible && !p1_feasible {
            DuelResult::RightWins
        } else {
            // Both feasible or both infeasible
            if p1_rank < p2_rank {
                DuelResult::LeftWins
            } else if p2_rank < p1_rank {
                DuelResult::RightWins
            } else {
                // Same rank, compare crowding distance
                if p1_cd > p2_cd {
                    DuelResult::LeftWins
                } else if p1_cd < p2_cd {
                    DuelResult::RightWins
                } else {
                    DuelResult::Tie
                }
            }
        };

        winner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Population;
    use numpy::ndarray::{arr1, arr2, Array2};
    use rand::prelude::*;

    #[test]
    fn test_tournament_selection_no_constraints_basic() {
        // For a population of 4:
        // Rank: [0, 1, 0, 1]
        // CD: [10.0, 5.0, 9.0, 1.0]
        let genes = arr2(&[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
        let fitness = arr2(&[[0.5], [0.6], [0.7], [0.8]]);
        let constraints = None;
        let rank = arr1(&[0, 1, 0, 1]);
        let cd = arr1(&[10.0, 5.0, 9.0, 1.0]);

        let population = Population::new(genes, fitness, constraints, rank, cd);

        // n_crossovers = 2
        // total_needed = 2 * 2 * 2 = 8 participants → 4 tournaments → 4 winners.
        // After splitting: pop_a = 2 winners, pop_b = 2 winners.
        let n_crossovers = 2;
        let selector = RankAndCrowdingSelection {};
        let mut rng = thread_rng();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 2);
        assert_eq!(pop_b.len(), 2);
    }

    #[test]
    fn test_tournament_selection_with_constraints() {
        // Two individuals:
        // Individual 0: feasible
        // Individual 1: infeasible
        let genes = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let fitness = arr2(&[[0.5], [0.6]]);
        let constraints = Some(arr2(&[[-1.0, 0.0], [1.0, 1.0]]));
        let rank = arr1(&[0, 0]);
        let cd = arr1(&[5.0, 10.0]);

        let population = Population::new(genes, fitness, constraints, rank, cd);

        // n_crossovers = 1
        // total_needed = 1 * 2 * 2 = 4 participants → 2 tournaments → 2 winners total.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndCrowdingSelection {};
        let mut rng = thread_rng();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The feasible individual should be one of the winners.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_tournament_selection_same_rank_and_cd() {
        // If two individuals have the same rank and the same CD,
        // the second one wins in the event of a tie.
        let genes = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let fitness = arr2(&[[0.5], [0.6]]);
        let constraints = None;
        let rank = arr1(&[0, 0]);
        let cd = arr1(&[10.0, 10.0]);

        let population = Population::new(genes, fitness, constraints, rank, cd);

        // n_crossovers = 1
        // total_needed = 4 participants → 2 tournaments → 2 winners
        // After splitting: pop_a = 1 winner, pop_b = 1 winner
        let n_crossovers = 1;
        let selector = RankAndCrowdingSelection {};
        let mut rng = thread_rng();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // In a tie, the second individual chosen in the tournament wins.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_tournament_selection_large_population() {
        // Large population test to ensure stability.
        let pop_size = 100;
        let n_genes = 5;

        let genes = Array2::from_shape_fn((pop_size, n_genes), |(i, _)| i as f64);
        let fitness = Array2::from_shape_fn((pop_size, 1), |(i, _)| i as f64 / 100.0);
        let constraints = None;

        let mut rng = thread_rng();
        let rank_vec: Vec<usize> = (0..pop_size).map(|_| rng.gen_range(0..5)).collect();
        let rank = arr1(&rank_vec);

        let cd_vec: Vec<f64> = (0..pop_size).map(|_| rng.gen_range(0.0..10.0)).collect();
        let cd = arr1(&cd_vec);

        let population = Population::new(genes, fitness, constraints, rank, cd);

        // n_crossovers = 50
        // total_needed = 50 * 2 * 2 = 200 participants → 100 tournaments → 100 winners.
        // After splitting: pop_a = 50 winners, pop_b = 50 winners.
        let n_crossovers = 50;
        let selector = RankAndCrowdingSelection {};
        let mut rng = thread_rng();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 50);
        assert_eq!(pop_b.len(), 50);
    }

    #[test]
    fn test_tournament_selection_single_tournament() {
        // One crossover:
        // total_needed = 1 * 2 * 2 = 4 participants → 2 tournaments → 2 winners
        // After splitting: pop_a = 1, pop_b = 1
        let genes = arr2(&[[10.0, 20.0], [30.0, 40.0]]);
        let fitness = arr2(&[[1.0], [2.0]]);
        let constraints = None;
        let rank = arr1(&[0, 1]);
        let cd = arr1(&[5.0, 1.0]);

        let population = Population::new(genes, fitness, constraints, rank, cd);

        let n_crossovers = 1;
        let selector = RankAndCrowdingSelection {};
        let mut rng = thread_rng();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The first individual has a better rank, so it should be one of the winners.
        // The second winner would be from the second tournament.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }
}
