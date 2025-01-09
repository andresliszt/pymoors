use crate::genetic::{Fronts, FrontsExt, Population};
use crate::operators::{GeneticOperator, SurvivalOperator};
use std::fmt::Debug;

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
    fn operate(&self, fronts: &Fronts, n_survive: usize) -> Population {
        // We will collect sub-populations (fronts) that survive here
        let mut chosen_fronts: Vec<Population> = Vec::new();
        let mut n_survivors = 0;

        for front in fronts.iter() {
            let front_size = front.len();

            // If this entire front fits into the survivor count
            if n_survivors + front_size <= n_survive {
                chosen_fronts.push(front.clone());
                n_survivors += front_size;
            } else {
                // Only part of this front fits
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // Sort by crowding distance (descending)
                    let cd = front.crowding_distance.clone();
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
        chosen_fronts.flatten_fronts()
    }
}

#[cfg(test)]
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
        let crowding_distance = arr1(&[10.0, 5.0, 7.0]);

        let population = Population::new(
            genes.clone(),
            fitness.clone(),
            constraints.clone(),
            rank.clone(),
            crowding_distance.clone(),
        );
        let mut fronts: Fronts = vec![population];

        let n_survive = 3;
        let selector = RankCrowdingSurvival;
        let new_population = selector.operate(&mut fronts, n_survive);

        // All three should survive unchanged
        assert_eq!(new_population.len(), 3);
        assert_eq!(new_population.genes, genes);
        assert_eq!(new_population.fitness, fitness);
    }

    #[test]
    fn test_survival_selection_partial_survival_single_front() {
        // Only a subset of individuals survive, chosen by descending crowding distance.
        let genes = arr2(&[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
        let fitness = arr2(&[[0.1], [0.2], [0.3]]);
        let constraints: Option<Array2<f64>> = None;
        let rank = arr1(&[0, 0, 0]);
        let crowding_distance = arr1(&[10.0, 5.0, 7.0]);

        let population = Population::new(
            genes.clone(),
            fitness.clone(),
            constraints.clone(),
            rank.clone(),
            crowding_distance.clone(),
        );
        let mut fronts: Fronts = vec![population];

        let n_survive = 2;
        let selector = RankCrowdingSurvival;
        let new_population = selector.operate(&mut fronts, n_survive);

        // Sort by CD descending: indices by CD would be [0 (10.0), 2 (7.0), 1 (5.0)]
        // Top two: indices [0,2]
        assert_eq!(new_population.len(), 2);
        assert_eq!(new_population.genes, arr2(&[[0.0, 1.0], [4.0, 5.0]]));
        assert_eq!(new_population.fitness, arr2(&[[0.1], [0.3]]));
    }

    #[test]
    fn test_survival_selection_multiple_fronts() {
        // Multiple fronts scenario:
        // Front 1: 2 individuals, all must survive
        // Front 2: 3 individuals, but we only need 2 more to reach n_survive=4 total
        // Selection from Front 2 should be by crowding distance.

        let front1_genes = arr2(&[[0.0, 1.0], [2.0, 3.0]]);
        let front1_fitness = arr2(&[[0.1], [0.2]]);
        let front1_constraints: Option<Array2<f64>> = None;
        let front1_rank = arr1(&[0, 0]);
        let front1_cd = arr1(&[8.0, 9.0]);

        let front2_genes = arr2(&[[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]);
        let front2_fitness = arr2(&[[0.3], [0.4], [0.5]]);
        let front2_constraints: Option<Array2<f64>> = None;
        let front2_rank = arr1(&[1, 1, 1]);
        let front2_cd = arr1(&[3.0, 10.0, 1.0]);

        let population1 = Population::new(
            front1_genes,
            front1_fitness,
            front1_constraints,
            front1_rank,
            front1_cd,
        );

        let population2 = Population::new(
            front2_genes,
            front2_fitness,
            front2_constraints,
            front2_rank,
            front2_cd,
        );

        let mut fronts: Fronts = vec![population1, population2];

        let n_survive = 4; // We want 4 individuals total
        let selector = RankCrowdingSurvival;
        let new_population = selector.operate(&mut fronts, n_survive);

        // After selecting the full first front (2 individuals),
        // from the second front we pick 2 out of 3 by highest CD.
        // Front2 CDs: [3.0, 10.0, 1.0], sorted desc: indices [1,0,2]
        // Take the top 2: indices [1 (CD=10.0), 0 (CD=3.0)]
        // That means from front2_genes we take rows [1, 0] in that order.
        // But the code sorts indices and then selects top. The order in final population
        // depends on flattening. The `select` keeps the chosen order, so we should see
        // individuals in their original order relative to each other if that matters.

        assert_eq!(new_population.len(), 4);

        // The final population should have the first 2 individuals from front 1:
        // [[0, 1], [2, 3]]
        // And the chosen 2 from front 2 with the highest CD (indices 1 and 0 from front 2):
        // front2 index 1 -> [6, 7]
        // front2 index 0 -> [4, 5]

        let expected_genes = arr2(&[[0.0, 1.0], [2.0, 3.0], [6.0, 7.0], [4.0, 5.0]]);
        let expected_fitness = arr2(&[[0.1], [0.2], [0.4], [0.3]]);
        assert_eq!(new_population.genes, expected_genes);
        assert_eq!(new_population.fitness, expected_fitness);
    }
}
