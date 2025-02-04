use crate::{
    genetic::{Fronts, Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    non_dominated_sorting::{crowding_distance, fast_non_dominated_sorting},
};
use numpy::ndarray::{Array1, Axis};

/// Evaluator struct for calculating fitness and (optionally) constraints,
/// then assembling a `Population`. In addition to the user-provided constraints function,
/// optional lower and upper bounds can be specified for the decision variables (genes).
pub struct Evaluator {
    fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
    constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
    keep_infeasible: bool,
    /// Optional lower bound for each gene.
    lower_bound: Option<f64>,
    /// Optional upper bound for each gene.
    upper_bound: Option<f64>,
}

impl Evaluator {
    /// Creates a new `Evaluator` with a fitness function, an optional constraints function,
    /// a flag to keep infeasible individuals, and optional lower/upper bounds for the genes.
    pub fn new(
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
        keep_infeasible: bool,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        Self {
            fitness_fn,
            constraints_fn,
            keep_infeasible,
            lower_bound,
            upper_bound,
        }
    }

    /// Evaluates the fitness of the population genes (2D ndarray).
    pub fn evaluate_fitness(&self, population_genes: &PopulationGenes) -> PopulationFitness {
        (self.fitness_fn)(population_genes)
    }

    /// Evaluates the constraints (if available).
    /// Returns `None` if no constraints function was provided.
    pub fn evaluate_constraints(
        &self,
        population_genes: &PopulationGenes,
    ) -> Option<PopulationConstraints> {
        self.constraints_fn.as_ref().map(|cf| cf(population_genes))
    }

    /// Builds the fronts from the population genes. If `keep_infeasible` is false,
    /// individuals are filtered out if they do not satisfy:
    ///   - The provided constraints function (all constraint values must be ≤ 0), and
    ///   - The optional lower and upper bounds (each gene must satisfy lower_bound <= gene <= upper_bound).
    pub fn build_fronts(&self, mut genes: PopulationGenes) -> Fronts {
        let mut fitness = self.evaluate_fitness(&genes);
        let mut constraints = self.evaluate_constraints(&genes);

        if !self.keep_infeasible {
            // Create a list of all indices.
            let n = genes.nrows();
            let mut feasible_indices: Vec<usize> = (0..n).collect();

            // Filter individuals that do not satisfy the constraints function (if provided)
            if let Some(ref c) = constraints {
                feasible_indices = feasible_indices
                    .into_iter()
                    .filter(|&i| c.index_axis(Axis(0), i).iter().all(|&val| val <= 0.0))
                    .collect();
            }

            // Further filter individuals based on the optional lower and upper bounds.
            if self.lower_bound.is_some() || self.upper_bound.is_some() {
                feasible_indices = feasible_indices
                    .into_iter()
                    .filter(|&i| {
                        let individual = genes.index_axis(Axis(0), i);
                        let lower_ok = self
                            .lower_bound
                            .map_or(true, |lb| individual.iter().all(|&x| x >= lb));
                        let upper_ok = self
                            .upper_bound
                            .map_or(true, |ub| individual.iter().all(|&x| x <= ub));
                        lower_ok && upper_ok
                    })
                    .collect();
            }

            // Filter all relevant arrays (genes, fitness, and constraints if present)
            genes = genes.select(Axis(0), &feasible_indices);
            fitness = fitness.select(Axis(0), &feasible_indices);
            constraints = constraints.map(|c_array| c_array.select(Axis(0), &feasible_indices));
        }

        let sorted_fronts = fast_non_dominated_sorting(&fitness);
        let mut results: Fronts = Vec::new();

        // For each front (rank = front_index), extract the sub-population.
        for (front_index, indices) in sorted_fronts.iter().enumerate() {
            // Slice out the genes and fitness for just these individuals.
            let front_genes = genes.select(Axis(0), &indices[..]);
            let front_fitness = fitness.select(Axis(0), &indices[..]);

            // If constraints exist, slice them out too.
            let front_constraints = constraints
                .as_ref()
                .map(|c| c.select(Axis(0), &indices[..]));

            // Calculate crowding distance for just these individuals.
            let cd_front = crowding_distance(&front_fitness);

            // Create a rank Array1 (one rank value per individual in the front).
            let rank_arr = Array1::from_elem(indices.len(), front_index);

            // Build a `Population` representing this entire front.
            let population_front = Population {
                genes: front_genes,
                fitness: front_fitness,
                constraints: front_constraints,
                rank: rank_arr,
                crowding_distance: cd_front,
            };

            results.push(population_front);
        }

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numpy::ndarray::{array, concatenate, Axis};

    // Fitness function: Sphere function (sum of squares for each individual).
    fn fitness_fn(genes: &PopulationGenes) -> PopulationFitness {
        genes
            .map_axis(Axis(1), |individual| {
                individual.iter().map(|&x| x * x).sum::<f64>()
            })
            .insert_axis(Axis(1))
    }

    // Constraints function:
    //   Constraint 1: sum of genes - 10 ≤ 0.
    //   Constraint 2: each gene ≥ 0 (represented as -x ≤ 0).
    fn constraints_fn(genes: &PopulationGenes) -> PopulationConstraints {
        let sum_constraint = genes
            .sum_axis(Axis(1))
            .mapv(|sum| sum - 10.0)
            .insert_axis(Axis(1));
        let non_neg_constraints = genes.mapv(|x| -x);
        concatenate(
            Axis(1),
            &[sum_constraint.view(), non_neg_constraints.view()],
        )
        .unwrap()
    }

    #[test]
    fn test_evaluator_evaluate_fitness() {
        let evaluator = Evaluator::new(Box::new(fitness_fn), None, true, None, None);

        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]];
        let fitness = evaluator.evaluate_fitness(&population_genes);
        let expected_fitness = array![
            [5.0],  // 1^2 + 2^2 = 5
            [25.0], // 3^2 + 4^2 = 25
            [0.0],  // 0^2 + 0^2 = 0
        ];

        assert_eq!(fitness, expected_fitness);
    }

    #[test]
    fn test_evaluator_evaluate_constraints() {
        let evaluator = Evaluator::new(
            Box::new(fitness_fn),
            Some(Box::new(constraints_fn)),
            true,
            None,
            None,
        );

        let population_genes = array![
            [1.0, 2.0], // Feasible (sum=3, sum-10=-7; genes ≥ 0)
            [3.0, 4.0], // Feasible (sum=7, sum-10=-3; genes ≥ 0)
            [5.0, 6.0]  // Infeasible (sum=11, sum-10=1>0)
        ];

        if let Some(constraints_array) = evaluator.evaluate_constraints(&population_genes) {
            let expected_constraints = array![
                // Each row: [sum - 10, -gene1, -gene2, ...]
                [-7.0, -1.0, -2.0],
                [-3.0, -3.0, -4.0],
                [1.0, -5.0, -6.0],
            ];
            assert_eq!(constraints_array, expected_constraints);

            // Verify feasibility: true if all constraint values are ≤ 0.
            let feasibility: Vec<bool> = constraints_array
                .outer_iter()
                .map(|row| row.iter().all(|&val| val <= 0.0))
                .collect();
            let expected_feasibility = vec![true, true, false];
            assert_eq!(feasibility, expected_feasibility);
        } else {
            panic!("Constraints function should not be None");
        }
    }

    #[test]
    fn test_evaluator_build_fronts_without_bounds() {
        let evaluator = Evaluator::new(
            Box::new(fitness_fn),
            Some(Box::new(constraints_fn)),
            true,
            None,
            None,
        );

        // Create a small population with 3 individuals.
        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Build fronts.
        let fronts = evaluator.build_fronts(population_genes);

        // Verify total number of individuals across all fronts equals 3.
        let total_individuals: usize = fronts.iter().map(|f| f.genes.nrows()).sum();
        assert_eq!(
            total_individuals, 3,
            "Total individuals across all fronts should be 3."
        );

        // Check that each front has matching shapes for rank and crowding distance.
        for (front_index, front_pop) in fronts.iter().enumerate() {
            let n = front_pop.genes.nrows();
            assert_eq!(
                front_pop.rank.len(),
                n,
                "Rank length should match number of individuals in front"
            );
            assert_eq!(
                front_pop.crowding_distance.len(),
                n,
                "Crowding distance length should match individuals in front"
            );
            for &r in front_pop.rank.iter() {
                assert_eq!(
                    r, front_index,
                    "Each individual's rank should match the front index"
                );
            }
            if let Some(ref c) = front_pop.constraints {
                assert_eq!(
                    c.nrows(),
                    n,
                    "Constraints rows should match number of individuals"
                );
            }
        }
    }

    #[test]
    fn test_evaluator_build_fronts_with_infeasible_and_bounds() {
        // Use constraints function and bounds. Also, keep_infeasible is false.
        // The bounds here require that every gene must be between 0 and 5.
        let evaluator = Evaluator::new(
            Box::new(fitness_fn),
            Some(Box::new(constraints_fn)),
            false,
            Some(0.0),
            Some(5.0),
        );

        // Create a population with 3 individuals:
        //   - [1.0, 2.0]: Feasible (sum=3, all genes between 0 and 5)
        //   - [3.0, 4.0]: Feasible (sum=7, all genes between 0 and 5)
        //   - [6.0, 1.0]: Infeasible (fails upper bound, since 6.0 > 5.0)
        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [6.0, 1.0],];

        // Build fronts.
        let fronts = evaluator.build_fronts(population_genes);

        // Expecting only 2 feasible individuals due to bounds filtering.
        let total_individuals: usize = fronts.iter().map(|f| f.genes.nrows()).sum();
        assert_eq!(
            total_individuals, 2,
            "Total individuals across all fronts should be 2 due to bounds filtering."
        );
    }
}
