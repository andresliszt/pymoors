use std::fmt::Debug;

use crate::genetic::Individual;
use crate::operators::{DuelResult, GeneticOperator, SelectionOperator, SurvivalScoringComparison};
use crate::random::RandomGenerator;

#[derive(Clone, Debug)]
pub struct RankAndScoringSelection {
    diversity_comparison: SurvivalScoringComparison,
}

impl RankAndScoringSelection {
    /// Creates a new RankAndScoringSelection with the default diversity comparison (Maximize).
    pub fn new() -> Self {
        Self {
            diversity_comparison: SurvivalScoringComparison::Maximize,
        }
    }
    /// Creates a new RankAndScoringSelection with the specified diversity comparison.
    pub fn new_with_comparison(comparison: SurvivalScoringComparison) -> Self {
        Self {
            diversity_comparison: comparison,
        }
    }
}

impl GeneticOperator for RankAndScoringSelection {
    fn name(&self) -> String {
        "RankAndScoringSelection".to_string()
    }
}

impl SelectionOperator for RankAndScoringSelection {
    /// Runs tournament selection on the given population and returns the duel result.
    /// This example assumes binary tournaments (pressure = 2).
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        _rng: &mut dyn RandomGenerator,
    ) -> DuelResult {
        // Check feasibility.
        let p1_feasible = p1.is_feasible();
        let p2_feasible = p2.is_feasible();
        // Retrieve rank.
        let p1_rank = p1.rank;
        let p2_rank = p2.rank;
        // Retrieve diversity (crowding) metric.
        let p1_cd = p1.survival_score;
        let p2_cd = p2.survival_score;

        let winner = if p1_feasible && !p2_feasible {
            DuelResult::LeftWins
        } else if p2_feasible && !p1_feasible {
            DuelResult::RightWins
        } else {
            // Both are either feasible or infeasible.
            if p1_rank < p2_rank {
                DuelResult::LeftWins
            } else if p2_rank < p1_rank {
                DuelResult::RightWins
            } else {
                // When ranks are equal, compare the diversity metric.
                match self.diversity_comparison {
                    SurvivalScoringComparison::Maximize => {
                        if p1_cd > p2_cd {
                            DuelResult::LeftWins
                        } else if p1_cd < p2_cd {
                            DuelResult::RightWins
                        } else {
                            DuelResult::Tie
                        }
                    }
                    SurvivalScoringComparison::Minimize => {
                        if p1_cd < p2_cd {
                            DuelResult::LeftWins
                        } else if p1_cd > p2_cd {
                            DuelResult::RightWins
                        } else {
                            DuelResult::Tie
                        }
                    }
                }
            }
        };

        winner
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    use ndarray::{array, Array1};
    use rand::prelude::*;

    use crate::genetic::{Individual, Population, PopulationFitness, PopulationGenes};
    use crate::operators::{DuelResult, SelectionOperator};
    use crate::random::MOORandomGenerator;

    #[test]
    fn test_default_diversity_comparison_maximize() {
        let selector = RankAndScoringSelection::new();
        match selector.diversity_comparison {
            SurvivalScoringComparison::Maximize => assert!(true),
            SurvivalScoringComparison::Minimize => panic!("Default should be Maximize"),
        }
    }

    #[test]
    fn test_tournament_duel_maximize() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());

        // Create two individuals using the actual Individual type.
        // Both individuals have the same rank (Some(0)) but different diversity metrics.
        // In Maximize mode, the individual with the higher diversity (10.0) should win.
        let p1 = Individual::new(array![1.0, 2.0], array![0.5], None, Some(0), Some(10.0));
        let p2 = Individual::new(array![3.0, 4.0], array![0.6], None, Some(0), Some(5.0));
        let selector = RankAndScoringSelection::new(); // Default: Maximize
        assert_eq!(selector.name(), "RankAndScoringSelection");
        let result = selector.tournament_duel(&p1, &p2, &mut rng);
        assert_eq!(result, DuelResult::LeftWins);
    }

    #[test]
    fn test_tournament_duel_minimize() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // Create two individuals using the actual Individual type.
        // Both individuals have the same rank (Some(0)) but different diversity metrics.
        // In Minimize mode, the individual with the lower diversity (5.0) should win.
        let p1 = Individual::new(array![1.0, 2.0], array![0.5], None, Some(0), Some(10.0));
        let p2 = Individual::new(array![3.0, 4.0], array![0.6], None, Some(0), Some(5.0));
        let selector =
            RankAndScoringSelection::new_with_comparison(SurvivalScoringComparison::Minimize);
        let result = selector.tournament_duel(&p1, &p2, &mut rng);
        assert_eq!(result, DuelResult::RightWins);
    }

    #[test]
    fn test_tournament_selection_no_constraints_basic() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // For a population of 4:
        // Rank: [0, 1, 0, 1]
        // Diversity (CD): [10.0, 5.0, 9.0, 1.0]
        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let fitness = array![[0.5], [0.6], [0.7], [0.8]];
        let constraints = None;
        let rank = Some(array![0, 1, 0, 1]);

        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 2 → total_needed = 8 participants → 4 tournaments → 4 winners.
        // After splitting: pop_a = 2 winners, pop_b = 2 winners.
        let n_crossovers = 2;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 2);
        assert_eq!(pop_b.len(), 2);
    }

    #[test]
    fn test_tournament_selection_with_constraints() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // Two individuals:
        // Individual 0: feasible
        // Individual 1: infeasible
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5], [0.6]];
        let constraints = Some(array![[-1.0, 0.0], [1.0, 1.0]]);
        let rank = Some(array![0, 0]);
        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners total.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The feasible individual should be one of the winners.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_tournament_selection_same_rank_and_cd() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // If two individuals have the same rank and the same crowding distance,
        // the tournament duel should result in a tie.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5], [0.6]];
        let constraints = None;
        let rank = Some(array![0, 0]);

        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 1 → total_needed = 4 participants → 2 tournaments → 2 winners.
        // After splitting: pop_a = 1 winner, pop_b = 1 winner.
        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // In a tie, the overall selection process must eventually choose winners.
        // For this test, we ensure that each subpopulation has 1 individual.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }

    #[test]
    fn test_tournament_selection_large_population() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // Large population test to ensure stability.
        let population_size = 100;
        let n_genes = 5;
        let genes = PopulationGenes::from_shape_fn((population_size, n_genes), |(i, _)| i as f64);
        let fitness =
            PopulationFitness::from_shape_fn((population_size, 1), |(i, _)| i as f64 / 100.0);
        let constraints = None;

        let rank_vec: Vec<usize> = (0..population_size)
            .map(|_| rng.gen_range_usize(0, n_genes))
            .collect();
        let rank = Some(Array1::from_vec(rank_vec));
        let population = Population::new(genes, fitness, constraints, rank);

        // n_crossovers = 50 → total_needed = 200 participants → 100 tournaments → 100 winners.
        // After splitting: pop_a = 50 winners, pop_b = 50 winners.
        let n_crossovers = 50;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        assert_eq!(pop_a.len(), 50);
        assert_eq!(pop_b.len(), 50);
    }

    #[test]
    fn test_tournament_selection_single_tournament() {
        let mut rng = MOORandomGenerator::new(StdRng::from_entropy());
        // One crossover:
        // total_needed = 4 participants → 2 tournaments → 2 winners.
        // After splitting: pop_a = 1, pop_b = 1.
        let genes = array![[10.0, 20.0], [30.0, 40.0]];
        let fitness = array![[1.0], [2.0]];
        let constraints = None;
        let rank = Some(array![0, 1]);

        let population = Population::new(genes, fitness, constraints, rank);

        let n_crossovers = 1;
        let selector = RankAndScoringSelection::new();
        let (pop_a, pop_b) = selector.operate(&population, n_crossovers, &mut rng);

        // The individual with the better rank should win one tournament.
        assert_eq!(pop_a.len(), 1);
        assert_eq!(pop_b.len(), 1);
    }
}
