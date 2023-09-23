use crate::{
    genetic::{Fronts, Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    non_dominated_sorting::{crowding_distance, fast_non_dominated_sorting},
};
use numpy::ndarray::{Array1, Axis};

/// Evaluator struct for calculating fitness and (optionally) constraints,
/// then assembling a `Population`.

pub struct Evaluator {
    fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
    constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
    keep_infeasible: bool,
}

impl Evaluator {
    /// Creates a new `Evaluator` with a fitness function, an optional constraints function, and a flag to keep infeasible individuals.
    pub fn new(
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
        keep_infeasible: bool,
    ) -> Self {
        Self {
            fitness_fn,
            constraints_fn,
            keep_infeasible,
        }
    }

    /// Evaluates the fitness of the population genes (2D ndarray).
    pub fn evaluate_fitness(&self, population_genes: &PopulationGenes) -> PopulationFitness {
        (self.fitness_fn)(population_genes)
    }

    /// Evaluates the constraints (if available).
    /// Returns `None` if no constraints function was given.
    pub fn evaluate_constraints(
        &self,
        population_genes: &PopulationGenes,
    ) -> Option<PopulationConstraints> {
        self.constraints_fn.as_ref().map(|cf| cf(population_genes))
    }

    pub fn build_fronts(&self, mut genes: PopulationGenes) -> Fronts {
        let mut fitness = self.evaluate_fitness(&genes);

        let mut constraints = self.evaluate_constraints(&genes);

        if !self.keep_infeasible {
            if let Some(ref c) = constraints {
                let feasible_indices: Vec<usize> = c
                    .outer_iter()
                    .enumerate()
                    .filter_map(|(i, row)| {
                        if row.iter().all(|&val| val <= 0.0) {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();

                // Filter *all* relevant arrays, including genes:
                genes = genes.select(Axis(0), &feasible_indices);
                fitness = fitness.select(Axis(0), &feasible_indices);
                constraints = constraints.map(|c_array| c_array.select(Axis(0), &feasible_indices));
            }
        }

        let sorted_fronts = fast_non_dominated_sorting(&fitness);

        let mut results: Fronts = Vec::new();

        // For each front (rank = front_index), extract the sub-population
        for (front_index, indices) in sorted_fronts.iter().enumerate() {
            // Slice out the genes and fitness for just these individuals
            let front_genes = genes.select(Axis(0), &indices[..]);
            let front_fitness = fitness.select(Axis(0), &indices[..]);

            // If constraints exist, slice them out too
            let front_constraints = constraints
                .as_ref()
                .map(|c| c.select(Axis(0), &indices[..]));

            // crowding_distance for just these individuals
            let cd_front = crowding_distance(&front_fitness);

            // Create a rank Array1 (one rank value per individual in the front)
            let rank_arr = Array1::from_elem(indices.len(), front_index);

            // Build a `Population` representing this entire front
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

    // Fitness function
    fn fitness_fn(genes: &PopulationGenes) -> PopulationFitness {
        // Sphere function: sum of squares for each individual
        genes
            .map_axis(Axis(1), |individual| {
                individual.iter().map(|&x| x * x).sum::<f64>()
            })
            .insert_axis(Axis(1))
    }

    // Constraints function
    fn constraints_fn(genes: &PopulationGenes) -> PopulationConstraints {
        // Constraint 1: sum of genes - 10 ≤ 0
        let sum_constraint = genes
            .sum_axis(Axis(1))
            .mapv(|sum| sum - 10.0)
            .insert_axis(Axis(1));

        // Constraint 2: each gene ≥ 0 (represented as -x ≤ 0)
        let non_neg_constraints = genes.mapv(|x| -x);

        // Combine constraints into one array of shape (n_individuals, n_constraints)
        concatenate(
            Axis(1),
            &[sum_constraint.view(), non_neg_constraints.view()],
        )
        .unwrap()
    }

    #[test]
    fn test_evaluator_evaluate_fitness() {
        let evaluator = Evaluator::new(Box::new(fitness_fn), None, true);

        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0],];

        let fitness = evaluator.evaluate_fitness(&population_genes);
        let expected_fitness = array![
            [5.0],  // 1^2 + 2^2
            [25.0], // 3^2 + 4^2
            [0.0],  // 0^2 + 0^2
        ];

        assert_eq!(fitness, expected_fitness);
    }

    #[test]
    fn test_evaluator_evaluate_constraints() {
        let evaluator = Evaluator::new(Box::new(fitness_fn), Some(Box::new(constraints_fn)), true);

        let population_genes = array![
            [1.0, 2.0], // Feasible (sum=3, sum-10=-7; genes >= 0)
            [3.0, 4.0], // Feasible (sum=7, sum-10=-3; genes >= 0)
            [5.0, 6.0], // Infeasible (sum=11, sum-10=1>0)
        ];

        if let Some(constraints_array) = evaluator.evaluate_constraints(&population_genes) {
            let expected_constraints = array![
                // For each row: [sum-10, -gene1, -gene2, ...]
                [-7.0, -1.0, -2.0],
                [-3.0, -3.0, -4.0],
                [1.0, -5.0, -6.0],
            ];
            assert_eq!(constraints_array, expected_constraints);

            // Optionally verify feasibility
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
    fn test_evaluator_build_fronts() {
        let evaluator = Evaluator::new(Box::new(fitness_fn), Some(Box::new(constraints_fn)), true);

        // We'll create a small population with 3 individuals
        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        // Build multiple fronts
        let fronts = evaluator.build_fronts(population_genes);

        // Expecting 3 individuals distributed across one or more fronts:
        // single-objective (sphere) + strictly dominated solutions
        // can often yield each individual in its own front if each solution strictly dominates
        // or is dominated. But let's just confirm front(s) logic.

        // Let's verify total number of individuals across all fronts = 3
        let total_individuals: usize = fronts.iter().map(|f| f.genes.nrows()).sum();
        assert_eq!(
            total_individuals, 3,
            "Total individuals across all fronts should be 3."
        );

        // We can also check that the constraints, rank, crowding_distance shape match
        for (front_index, front_pop) in fronts.iter().enumerate() {
            // Each front has front_pop.genes, front_pop.fitness, front_pop.constraints
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

            // For each individual in this front, the rank array should be `front_index`
            for &r in front_pop.rank.iter() {
                assert_eq!(
                    r, front_index,
                    "Each individual's rank should match the front index"
                );
            }

            // If constraints exist, shape should match
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
    fn test_evaluator_build_fronts_with_infeasible() {
        let evaluator = Evaluator::new(Box::new(fitness_fn), Some(Box::new(constraints_fn)), false);

        // We'll create a small population with 3 individuals
        let population_genes = array![
            [1.0, 2.0], // Feasible
            [3.0, 4.0], // Feasible
            [5.0, 6.0], // Infeasible
        ];

        // Build multiple fronts
        let fronts = evaluator.build_fronts(population_genes);

        // Expecting 2 individuals distributed across one or more fronts:
        // single-objective (sphere) + strictly dominated solutions
        // can often yield each individual in its own front if each solution strictly dominates
        // or is dominated. But let's just confirm front(s) logic.

        // Let's verify total number of individuals across all fronts = 2
        let total_individuals: usize = fronts.iter().map(|f| f.genes.nrows()).sum();
        assert_eq!(
            total_individuals, 2,
            "Total individuals across all fronts should be 2."
        );

        // We can also check that the constraints, rank, crowding_distance shape match
        for (front_index, front_pop) in fronts.iter().enumerate() {
            // Each front has front_pop.genes, front_pop.fitness, front_pop.constraints
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

            // For each individual in this front, the rank array should be `front_index`
            for &r in front_pop.rank.iter() {
                assert_eq!(
                    r, front_index,
                    "Each individual's rank should match the front index"
                );
            }

            // If constraints exist, shape should match
            if let Some(ref c) = front_pop.constraints {
                assert_eq!(
                    c.nrows(),
                    n,
                    "Constraints rows should match number of individuals"
                );
            }
        }
    }
}
