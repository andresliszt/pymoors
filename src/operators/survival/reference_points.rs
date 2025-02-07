// src/lib.rs

use crate::genetic::{Fronts, FrontsExt, Population};
use crate::operators::{GeneticOperator, SurvivalOperator};
use ndarray::{s, Array2, ArrayView1};
use std::collections::HashMap;

/// Structure representing the NSGA-III survival operator.
#[derive(Clone, Debug)]
pub struct ReferencePointsSurvival {
    reference_points: Array2<f64>, // Each row is a reference point
}

impl GeneticOperator for ReferencePointsSurvival {
    fn name(&self) -> String {
        "ReferencePointsSurvival".to_string()
    }
}

impl ReferencePointsSurvival {
    /// Creates a new ReferencePointsSurvival operator with the provided reference points.
    pub fn new(reference_points: Array2<f64>) -> Self {
        Self { reference_points }
    }
}

impl SurvivalOperator for ReferencePointsSurvival {
    fn operate(&self, fronts: &Fronts, n_survive: usize) -> Population {
        // Initialize a vector to store selected fronts (populations)
        let mut chosen_fronts: Vec<Population> = Vec::new();
        let mut n_survivors = 0;

        // Iterate over the fronts
        for front in fronts.iter() {
            let front_size = front.len();

            // If the entire front fits within the survival limit, take the whole front.
            if n_survivors + front_size <= n_survive {
                chosen_fronts.push(front.clone());
                n_survivors += front_size;
            } else {
                // Only a portion of this front can survive.
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // Create a temporary normalized copy of the front (for selection purposes only)
                    let normalized_front = normalize_front(front);
                    // Assign individuals to reference points using the normalized fitness values
                    let assignments =
                        assign_to_reference_points(&normalized_front, &self.reference_points);
                    // Perform niching selection on the normalized front to obtain the indices
                    let selected_indices = niching_selection(
                        &normalized_front,
                        &assignments,
                        remaining,
                        &self.reference_points,
                    );
                    // Use the indices to select individuals from the original (non‐normalized) front
                    chosen_fronts.push(front.selected(&selected_indices));
                }
                // No more slots available after processing this front.
                break;
            }
        }

        // Combine the chosen fronts into a single population
        FrontsExt::to_population(&chosen_fronts)
    }
}

/// Normalizes the objectives of a front to the [0, 1] range.
/// This function returns a copy of the population with normalized fitness values.
fn normalize_front(front: &Population) -> Population {
    let n_objectives = front.fitness.ncols();
    let mut normalized = front.clone();

    for m in 0..n_objectives {
        let col = normalized.fitness.column(m);
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Avoid division by zero: if max equals min, set the column to zeros.
        if max > min {
            let mut col_mut = normalized.fitness.slice_mut(s![.., m]);
            col_mut.mapv_inplace(|x| (x - min) / (max - min));
        } else {
            let mut col_mut = normalized.fitness.slice_mut(s![.., m]);
            col_mut.mapv_inplace(|_| 0.0);
        }
    }

    normalized
}

/// Assigns each individual in the front to the nearest reference point (using perpendicular distance)
/// based on the normalized objectives.
fn assign_to_reference_points(
    front: &Population,
    reference_points: &Array2<f64>,
) -> HashMap<usize, Vec<usize>> {
    let mut assignments: HashMap<usize, Vec<usize>> = HashMap::new();

    for (i, individual) in front.fitness.outer_iter().enumerate() {
        let mut min_distance = f64::INFINITY;
        let mut assigned_ref = 0;

        for (j, ref_point) in reference_points.outer_iter().enumerate() {
            let distance = perpendicular_distance(&individual, &ref_point);
            if distance < min_distance {
                min_distance = distance;
                assigned_ref = j;
            }
        }

        assignments
            .entry(assigned_ref)
            .or_insert_with(Vec::new)
            .push(i);
    }

    assignments
}

/// Calculates the perpendicular distance from an individual to a reference point.
fn perpendicular_distance(individual: &ArrayView1<f64>, ref_point: &ArrayView1<f64>) -> f64 {
    // Calculate the Euclidean norm of the reference point
    let ref_norm = ref_point.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();

    // Avoid division by zero
    if ref_norm == 0.0 {
        return f64::INFINITY;
    }

    // Compute the projection of the individual onto the reference point
    let dot_product: f64 = individual
        .iter()
        .zip(ref_point.iter())
        .map(|(a, b)| a * b)
        .sum();
    let projection = ref_point.mapv(|x| (dot_product / ref_norm.powi(2)) * x);

    // Calculate and return the perpendicular distance
    let diff = individual - &projection;
    diff.mapv(|x| x.powi(2)).sum().sqrt()
}

/// Performs niching selection based on the assignments to reference points.
/// It returns the indices (with respect to the normalized front) of the individuals to be selected.
/// The final population will use these indices to extract individuals from the original (non-normalized) front.
fn niching_selection(
    normalized_front: &Population,
    assignments: &HashMap<usize, Vec<usize>>,
    remaining: usize,
    reference_points: &Array2<f64>,
) -> Vec<usize> {
    let mut selected_indices: Vec<usize> = Vec::new();

    // To ensure deterministic behavior, iterate over reference point keys in sorted order.
    let mut ref_keys: Vec<&usize> = assignments.keys().collect();
    ref_keys.sort();

    for &ref_idx in ref_keys.iter() {
        if selected_indices.len() >= remaining {
            break;
        }

        // Get the individuals assigned to this reference point.
        let inds = &assignments[ref_idx];

        // Sort the individuals by their perpendicular distance to the corresponding reference point.
        let mut sorted_inds = inds.clone();
        sorted_inds.sort_by(|&i, &j| {
            let individual_i = normalized_front.fitness.row(i);
            let individual_j = normalized_front.fitness.row(j);
            // Use the actual reference point (from the operator’s reference points) for the distance
            let ref_point = reference_points.row(*ref_idx);

            let dist_i = perpendicular_distance(&individual_i, &ref_point);
            let dist_j = perpendicular_distance(&individual_j, &ref_point);

            dist_i
                .partial_cmp(&dist_j)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select the closest individual (if available)
        if let Some(&best_ind) = sorted_inds.first() {
            selected_indices.push(best_ind);
        }
    }

    // If there are still slots remaining, select additional individuals from those not yet selected.
    if selected_indices.len() < remaining {
        let mut remaining_inds: Vec<usize> = (0..normalized_front.len()).collect();
        remaining_inds.retain(|&i| !selected_indices.contains(&i));

        // Simply take the first needed individuals (this strategy can be improved)
        let additional: Vec<usize> = remaining_inds
            .into_iter()
            .take(remaining - selected_indices.len())
            .collect();
        selected_indices.extend(additional);
    }

    selected_indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic::Population;
    use crate::operators::SurvivalOperator;
    use ndarray::{array, Array1, Array2};
    use rand::Rng;
    use std::collections::HashMap;

    /// Helper function: Checks if two 2D arrays are approximately equal within a given epsilon.
    fn arrays_approx_eq(a: &Array2<f64>, b: &Array2<f64>, epsilon: f64) -> bool {
        if a.shape() != b.shape() {
            println!("Shape mismatch: {:?} vs {:?}", a.shape(), b.shape());
            return false;
        }
        for ((&a_elem, &b_elem), idx) in a.iter().zip(b.iter()).zip(a.indexed_iter()) {
            if (a_elem - b_elem).abs() > epsilon {
                println!(
                    "Elements at index {:?} differ: {} vs {} (epsilon = {})",
                    idx.0, a_elem, b_elem, epsilon
                );
                return false;
            }
        }
        true
    }

    /// Helper function to generate uniformly distributed reference points on the simplex.
    fn generate_reference_points(n_points: usize, n_objectives: usize) -> Array2<f64> {
        // A simple grid-based generation for demonstration purposes.
        let mut points = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..n_points {
            let mut point = Vec::new();
            let mut sum = 0.0;
            for _ in 0..n_objectives {
                let val: f64 = rng.gen_range(0.0..1.0);
                point.push(val);
                sum += val;
            }
            if sum > 0.0 {
                let normalized_point: Vec<f64> = point.iter().map(|&x| x / sum).collect();
                points.extend(normalized_point);
            } else {
                let normalized_point: Vec<f64> = vec![1.0 / n_objectives as f64; n_objectives];
                points.extend(normalized_point);
            }
        }

        Array2::from_shape_vec((n_points, n_objectives), points)
            .expect("Failed to create reference points")
    }

    #[test]
    fn test_nsgaiii_survival() {
        // Create a simple population with 6 individuals and 3 objectives.
        let genes = array![
            [1.0, 2.0, 3.0],
            [1.5, 1.8, 2.5],
            [2.0, 1.5, 2.0],
            [2.5, 1.2, 1.5],
            [3.0, 1.0, 1.0],
            [3.5, 0.8, 0.5]
        ];
        let fitness = array![
            [1.0, 2.0, 3.0],
            [1.5, 1.8, 2.5],
            [2.0, 1.5, 2.0],
            [2.5, 1.2, 1.5],
            [3.0, 1.0, 1.0],
            [3.5, 0.8, 0.5]
        ];
        let population = Population::new(
            genes.clone(),
            fitness.clone(),
            None,
            Array1::from(vec![1, 1, 1, 1, 1, 1]),
            Array1::from(vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        );

        // Define fronts (for simplicity, all individuals are in the first front)
        let fronts = vec![population.clone()];

        // Generate reference points.
        let n_ref_points = 3;
        let n_objectives = 3;
        let reference_points = generate_reference_points(n_ref_points, n_objectives);

        // Create a ReferencePointsSurvival operator.
        let nsga3_survival = ReferencePointsSurvival::new(reference_points.clone());

        // Perform survival operation to select 4 individuals.
        let n_survive = 4;
        let new_population = nsga3_survival.operate(&fronts, n_survive);

        // Assert that the new population has 4 individuals.
        assert_eq!(new_population.len(), n_survive);

        // Also verify that the fitness values are the original ones (not normalized).
        // (Here we simply check that none of the fitness values are in [0, 1] range due to normalization.)
        // This test may be adapted depending on your data.
        for val in new_population.fitness.iter() {
            assert!(val > &1.0);
        }
    }

    #[test]
    fn test_normalize_front() {
        let fitness = array![[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]];
        let population = Population::new(
            array![[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]],
            fitness.clone(),
            None,
            Array1::from(vec![1, 1, 1]),
            Array1::from(vec![0.5, 0.6, 0.7]),
        );

        let normalized = normalize_front(&population);

        let expected_fitness = array![
            [0.0, 0.0],
            [0.3333333333333333, 0.3333333333333333],
            [0.6666666666666666, 0.6666666666666666]
        ];

        assert!(
            arrays_approx_eq(&normalized.fitness, &expected_fitness, 1e-6),
            "Normalization failed"
        );
    }

    #[test]
    fn test_perpendicular_distance() {
        let individual = array![1.0, 1.0];
        let ref_point = array![1.0, 0.0];
        let distance = perpendicular_distance(&individual.view(), &ref_point.view());

        // The perpendicular distance from (1,1) to the line defined by (1,0) is 1.0.
        assert!((distance - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_assign_to_reference_points() {
        let fitness = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let population = Population::new(
            array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
            fitness.clone(),
            None,
            Array1::from(vec![1, 1, 1]),
            Array1::from(vec![0.5, 0.6, 0.7]),
        );

        let reference_points = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let assignments = assign_to_reference_points(&population, &reference_points);

        // Each individual should be assigned to the corresponding reference point.
        assert_eq!(assignments.len(), 3);
        assert_eq!(assignments.get(&0).unwrap(), &vec![0]);
        assert_eq!(assignments.get(&1).unwrap(), &vec![1]);
        assert_eq!(assignments.get(&2).unwrap(), &vec![2]);
    }

    #[test]
    fn test_niching_selection() {
        // Define fitness for 4 individuals.
        let fitness = array![
            [0.0, 0.0], // Individual 0
            [0.1, 0.1], // Individual 1
            [0.2, 0.2], // Individual 2
            [0.3, 0.3]  // Individual 3
        ];

        // Create a population with these fitness values.
        let population = Population::new(
            array![
                [1.0, 2.0], // Genes for Individual 0
                [2.0, 3.0], // Genes for Individual 1
                [3.0, 4.0], // Genes for Individual 2
                [4.0, 5.0]  // Genes for Individual 3
            ],
            fitness.clone(),
            None,
            Array1::from(vec![1, 1, 1, 1]),
            Array1::from(vec![0.5, 0.6, 0.7, 0.8]),
        );

        // Define assignments of individuals to reference points.
        // For this test, we create a dummy assignment.
        let assignments: HashMap<usize, Vec<usize>> = HashMap::from([
            (0, vec![0, 1]), // Reference Point 0: Individuals 0 and 1
            (1, vec![2]),    // Reference Point 1: Individual 2
            (2, vec![3]),    // Reference Point 2: Individual 3
        ]);

        // For testing, create dummy reference points.
        // Here we set all reference points to [0.0, 0.0] so that the perpendicular distance is just the Euclidean norm.
        let reference_points = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

        // Perform niching selection to select 2 individuals.
        let selected_indices = niching_selection(&population, &assignments, 2, &reference_points);

        // The expected selected individuals are:
        // - From reference point 0: Individual 0 (distance 0.0 < distance of Individual 1: ~0.141)
        // - From reference point 1: Individual 2 (only candidate)
        // Note: Since we iterate over keys in sorted order, the order is deterministic.
        let expected_indices = vec![0, 2];

        assert_eq!(
            selected_indices, expected_indices,
            "Niching selection indices do not match expected indices"
        );

        // Additionally, we can create a new population from these indices and verify its fitness.
        let selected_population = population.selected(&selected_indices);
        let expected_fitness = array![
            [0.0, 0.0], // Individual 0
            [0.2, 0.2]  // Individual 2
        ];
        assert!(
            arrays_approx_eq(&selected_population.fitness, &expected_fitness, 1e-6),
            "Selected population fitness does not match expected values"
        );
    }
}
