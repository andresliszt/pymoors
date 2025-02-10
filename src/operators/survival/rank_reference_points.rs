use std::cmp::Ordering;
use std::fmt::Debug;

use ndarray::{Array1, Array2};

use crate::diversity_metrics::{reference_points_rank_distance, weighted_distance_matrix};
use crate::genetic::{Fronts, FrontsExt, Population};
use crate::operators::{GeneticOperator, SurvivalOperator};

#[derive(Clone, Debug)]
pub struct RankReferencePointsSurvival {
    reference_points: Array2<f64>,
    epsilon: f64,
}

impl GeneticOperator for RankReferencePointsSurvival {
    fn name(&self) -> String {
        "RankReferencePointsSurvival".to_string()
    }
}

impl RankReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>, epsilon: f64) -> Self {
        Self {
            reference_points,
            epsilon,
        }
    }
}

/// This function computes the weighted, normalized Euclidean distance matrix between each solution
/// in the front (fitness matrix) and a set of reference points.
/// (It is already defined in your diversity_metrics module.)
/// Here we assume it is available as:
///    weighted_distance_matrix(fitness: &PopulationFitness, reference: &Array2<f64>) -> Array2<f64>
/// and that reference_points_rank_distance calls it to compute a ranking.
///
/// The splitting front procedure below uses weighted_distance_matrix(fitness, fitness)
/// to compute the internal distances among solutions.

impl SurvivalOperator for RankReferencePointsSurvival {
    fn operate(&self, fronts: &mut Fronts, n_survive: usize) -> Population {
        // We will collect sub-populations (fronts) that survive here.
        let mut chosen_fronts: Vec<Population> = Vec::new();
        let mut n_survivors = 0;

        for front in fronts.iter_mut() {
            let front_size = front.len();
            if n_survivors + front_size <= n_survive {
                // Compute the reference ranking for the entire front using the given reference points.
                let ranking_by_distance =
                    reference_points_rank_distance(&front.fitness, &self.reference_points);
                // Set the diversity metric (crowding distance) for the whole front.
                front
                    .set_diversity(ranking_by_distance)
                    .expect("Failed to set diversity metric");
                // If the entire front fits, add it as is.
                chosen_fronts.push(front.clone());
                n_survivors += front_size;
            } else {
                // Only part of this front will survive (splitting front).
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    let front_size = front.len();
                    // Compute the internal distance matrix among solutions in the front.
                    // Here, we use the fitness matrix as both arguments so that we get pairwise distances.
                    let mut internal_dist =
                        weighted_distance_matrix(&front.fitness, &front.fitness);
                    // Set the diagonal to infinity so that a solution is not considered close to itself.
                    for i in 0..front_size {
                        internal_dist[[i, i]] = f64::INFINITY;
                    }
                    // Get the reference ranking for the front (using the reference points).
                    let reference_ranking =
                        reference_points_rank_distance(&front.fitness, &self.reference_points);
                    let reference_ranking_vec = reference_ranking.to_vec();

                    // Create set S: all indices of solutions in the front,
                    // sorted in ascending order by the reference ranking.
                    let mut s: Vec<usize> = (0..front_size).collect();
                    s.sort_by(|&i, &j| {
                        reference_ranking_vec[i]
                            .partial_cmp(&reference_ranking_vec[j])
                            .unwrap_or(Ordering::Equal)
                    });
                    // Initialize a crowding vector with the initial reference ranking values.
                    let mut crowding = reference_ranking_vec.clone();
                    // Define a penalty Δ as the ceiling of half the front size.
                    let delta = (front_size as f64 / 2.0).ceil();

                    // Iterative procedure:
                    // While S is not empty, select the best (lowest ranked) solution and penalize close neighbors.
                    while !s.is_empty() {
                        let i_star = s[0];
                        s.remove(0); // Remove i_star from S.
                                     // Identify group G: all remaining solutions j in S such that the internal distance
                                     // between i_star and j is less than epsilon.
                        let group: Vec<usize> = s
                            .iter()
                            .cloned()
                            .filter(|&j| internal_dist[[i_star, j]] < self.epsilon)
                            .collect();
                        // Penalize each solution j in group by adding Δ to its reference ranking.
                        for &j in group.iter() {
                            crowding[j] = reference_ranking_vec[j] + delta;
                        }
                        // Remove all indices in G from S.
                        s.retain(|&x| !group.contains(&x));
                    }
                    // Now, sort the indices of the front by the updated (modified) crowding values (ascending).
                    let mut sorted_indices: Vec<usize> = (0..front_size).collect();
                    sorted_indices.sort_by(|&i, &j| {
                        crowding[i]
                            .partial_cmp(&crowding[j])
                            .unwrap_or(Ordering::Equal)
                    });
                    // Select the first 'remaining' indices.
                    let selected_indices: Vec<usize> =
                        sorted_indices.into_iter().take(remaining).collect();
                    // Create a new sub-population from the splitting front using only the selected individuals.
                    let mut selected_front = front.selected(&selected_indices);
                    // Set the updated crowding (diversity) for the selected sub-front.
                    let selected_crowding = Array1::from(
                        selected_indices
                            .iter()
                            .map(|&i| crowding[i])
                            .collect::<Vec<f64>>(),
                    );
                    selected_front
                        .set_diversity(selected_crowding)
                        .expect("Failed to set updated diversity metric");
                    chosen_fronts.push(selected_front);
                }
                // No more slots remain; break out.
                break;
            }
        }
        // Finally, combine the chosen fronts into a single population.
        chosen_fronts.to_population()
    }
}
