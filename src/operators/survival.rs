use crate::genetic::{Fitness, Fronts, FrontsExt, Population};
use crate::operators::GeneticOperator;

impl GeneticOperator for RankCrowdSurvival {
    fn name(&self) -> String {
        "RankCrowdSurvival".to_string()
    }
}

#[derive(Clone, Debug)]
pub struct RankCrowdSurvival;

impl RankCrowdSurvival {
    /// Performs survival selection based on rank and crowding distance.
    pub fn operate<Dna, F>(
        &self,
        fronts: &mut Fronts<Dna, F>,
        n_survive: usize,
    ) -> Population<Dna, F>
    where
        Dna: Clone,
        F: Fitness + Into<f64>,
    {
        let mut n_survivors = 0;
        for pop_front in fronts.iter_mut() {
            if n_survivors + pop_front.len() <= n_survive {
                // all individuals in this front are survivors
                n_survivors += pop_front.len();
                continue;
            } else {
                let remaining = n_survive - n_survivors;
                // Sort the individuals by crowding distance in descending order
                let cd = pop_front.crowding_distance.clone();
                let mut indices: Vec<usize> = (0..pop_front.len()).collect();
                indices.sort_by(|&i, &j| {
                    cd[j]
                        .partial_cmp(&cd[i])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Keep only the top `remaining` individuals
                let selected_indices = indices.into_iter().take(remaining).collect::<Vec<_>>();
                pop_front.select(&ndarray::Array1::from_vec(selected_indices));
                break; // Stop, we've reached `n_survive`
            }
        }
        fronts.flatten_fronts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};

    #[test]
    fn test_survival_selection_all_survive_single_front() {
        // All individuals can survive without partial selection.
        let genes = arr2(&[[0, 1], [2, 3], [4, 5]]);
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
        let mut fronts: Fronts<_, f64> = vec![population];

        let n_survive = 3;
        let selector = RankCrowdSurvival;
        let new_population = selector.operate(&mut fronts, n_survive);

        // All three should survive unchanged
        assert_eq!(new_population.len(), 3);
        assert_eq!(new_population.genes, genes);
        assert_eq!(new_population.fitness, fitness);
    }

    #[test]
    fn test_survival_selection_partial_survival_single_front() {
        // Only a subset of individuals survive, chosen by descending crowding distance.
        let genes = arr2(&[[0, 1], [2, 3], [4, 5]]);
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
        let mut fronts: Fronts<_, f64> = vec![population];

        let n_survive = 2;
        let selector = RankCrowdSurvival;
        let new_population = selector.operate(&mut fronts, n_survive);

        // Sort by CD descending: indices by CD would be [0 (10.0), 2 (7.0), 1 (5.0)]
        // Top two: indices [0,2]
        assert_eq!(new_population.len(), 2);
        assert_eq!(new_population.genes, arr2(&[[0, 1], [4, 5]]));
        assert_eq!(new_population.fitness, arr2(&[[0.1], [0.3]]));
    }

    #[test]
    fn test_survival_selection_multiple_fronts() {
        // Multiple fronts scenario:
        // Front 1: 2 individuals, all must survive
        // Front 2: 3 individuals, but we only need 2 more to reach n_survive=4 total
        // Selection from Front 2 should be by crowding distance.

        let front1_genes = arr2(&[[0, 1], [2, 3]]);
        let front1_fitness = arr2(&[[0.1], [0.2]]);
        let front1_constraints: Option<Array2<f64>> = None;
        let front1_rank = arr1(&[0, 0]);
        let front1_cd = arr1(&[8.0, 9.0]);

        let front2_genes = arr2(&[[4, 5], [6, 7], [8, 9]]);
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

        let mut fronts: Fronts<_, f64> = vec![population1, population2];

        let n_survive = 4; // We want 4 individuals total
        let selector = RankCrowdSurvival;
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

        let expected_genes = arr2(&[[0, 1], [2, 3], [6, 7], [4, 5]]);
        let expected_fitness = arr2(&[[0.1], [0.2], [0.4], [0.3]]);
        assert_eq!(new_population.genes, expected_genes);
        assert_eq!(new_population.fitness, expected_fitness);
    }
}
