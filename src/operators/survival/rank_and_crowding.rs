use std::fmt::Debug;

use crate::diversity_metrics::crowding_distance;
use crate::genetic::{Fronts, FrontsExt, Population};
use crate::operators::{GeneticOperator, SurvivalOperator};

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
    fn operate(&self, fronts: &mut Fronts, n_survive: usize) -> Population {
        // We will collect sub-populations (fronts) that survive here
        let mut chosen_fronts: Vec<Population> = Vec::new();
        let mut n_survivors = 0;

        for front in fronts.iter_mut() {
            let front_size = front.len();
            // Compute crowding distance
            let cd = crowding_distance(&front.fitness);
            // Set the crowding distance to the front population
            front
                .set_diversity(cd)
                .expect("Failed to set diversity metric");
            // If this entire front fits into the survivor count
            if n_survivors + front_size <= n_survive {
                chosen_fronts.push(front.clone());
                n_survivors += front_size;
            } else {
                // Only part of this front fits
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // Sort by crowding distance (descending)
                    let cd = front.diversity_metric.clone().unwrap();
                    let mut indices: Vec<usize> = (0..front_size).collect();
                    indices.sort_by(|&i, &j| {
                        cd[j]
                            .partial_cmp(&cd[i])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });

                    // Take only 'remaining' individuals
                    let selected_indices = indices.into_iter().take(remaining).collect::<Vec<_>>();
                    let partial = front.selected(&selected_indices);
                    chosen_fronts.push(partial);
                }
                // No more slots left after this
                break;
            }
        }

        // Finally, combine the chosen fronts into a single population
        chosen_fronts.to_population()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use numpy::ndarray::{arr1, arr2, Array2};

    #[test]
    fn test_survival_selection_all_survive_single_front() {
        // All individuals can survive without partial selection.
        let genes = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
        let fitness = arr2(&[[0.1], [0.2], [0.3]]);
        let constraints: Option<Array2<f64>> = None;
        let rank = arr1(&[0, 0, 0]);

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
        let new_population = selector.operate(&mut fronts, n_survive);

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
        let front1_genes = arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        let front1_fitness = arr2(&[[0.1], [0.2]]);
        let front1_constraints: Option<Array2<f64>> = None;
        let front1_rank = arr1(&[0, 0]);

        // Front 2: 4 individuals (second front, rank 1)
        // With fitness values arranged in increasing order: 0.3, 0.4, 0.5, 0.6.
        // In classical crowding distance, individuals with fitness 0.3 (first) and 0.6 (last) get INFINITY.
        let front2_genes = arr2(&[[4.0, 5.0], [6.0, 7.0], [8.0, 9.0], [10.0, 11.0]]);
        let front2_fitness = arr2(&[[0.3], [0.4], [0.5], [0.6]]);
        let front2_constraints: Option<Array2<f64>> = None;
        let front2_rank = arr1(&[1, 1, 1, 1]);

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
        let new_population = selector.operate(&mut fronts, n_survive);

        // The final population must have 4 individuals.
        assert_eq!(new_population.len(), n_survive);

        // Expected outcome:
        // - From Front 1, all individuals are selected: indices [0, 1] with genes [[0.0,1.0], [2.0,3.0]].
        // - From Front 2, only 2 individuals are selected based on crowding distance.
        //   In classical NSGA-II, the extreme individuals (with lowest and highest fitness) are selected.
        //   Therefore, from Front 2, the individuals at index 0 (fitness 0.3) and index 3 (fitness 0.6) are selected.
        //
        // Thus, the final population should have:
        //   Genes: [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]]
        //   Fitness: [[0.1], [0.2], [0.3], [0.6]]
        let expected_genes = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [10.0, 11.0]]);
        let expected_fitness = arr2(&[[0.1], [0.2], [0.3], [0.6]]);
        assert_eq!(new_population.genes, expected_genes);
        assert_eq!(new_population.fitness, expected_fitness);
    }
}
