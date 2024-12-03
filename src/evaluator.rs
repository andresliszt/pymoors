use crate::genetic::{PopulationConstraints, PopulationFitness, PopulationGenes};

use ndarray::Array1;
use num_traits::Zero;
use std::marker::PhantomData;

pub struct Evaluator<Dna, F, G> {
    fitness_fn: Box<dyn Fn(&PopulationGenes<Dna>) -> PopulationFitness<F> + Send + Sync>,
    constraints_fn:
        Option<Box<dyn Fn(&PopulationGenes<Dna>) -> PopulationConstraints<G> + Send + Sync>>,
    _phantom: PhantomData<Dna>,
}

impl<Dna, F, G> Evaluator<Dna, F, G>
where
    Dna: Clone + 'static,
    F: Clone + 'static,
    G: Clone + PartialOrd + Zero + 'static,
{
    pub fn new(
        fitness_fn: Box<dyn Fn(&PopulationGenes<Dna>) -> PopulationFitness<F> + Send + Sync>,
        constraints_fn: Option<
            Box<dyn Fn(&PopulationGenes<Dna>) -> PopulationConstraints<G> + Send + Sync>,
        >,
    ) -> Self {
        Self {
            fitness_fn,
            constraints_fn,
            _phantom: PhantomData,
        }
    }

    /// Evaluates the fitness of the population.
    pub fn evaluate_fitness(
        &self,
        population_genes: &PopulationGenes<Dna>,
    ) -> PopulationFitness<F> {
        (self.fitness_fn)(population_genes)
    }

    /// Evaluates the constraints for the entire population.
    ///
    /// # Returns
    ///
    /// - `Option<(PopulationConstraints<G>, Array1<bool>)>`:
    ///   - `PopulationConstraints<G>`: Each row corresponds to an individual's constraint evaluations.
    ///   - `Array1<bool>`: Indicates feasibility (`true` if all constraints ≤ 0).
    pub fn evaluate_constraints(
        &self,
        population_genes: &PopulationGenes<Dna>,
    ) -> Option<(PopulationConstraints<G>, Array1<bool>)> {
        if let Some(constraints_fn) = &self.constraints_fn {
            let constraints_array = (constraints_fn)(population_genes);

            // Determine feasibility for each individual
            let feasibility_array = constraints_array
                .outer_iter()
                .map(|constraint_values| constraint_values.iter().all(|c| *c <= G::zero()))
                .collect::<Array1<bool>>();

            Some((constraints_array, feasibility_array))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Axis};

    // Fitness function
    fn fitness_fn(genes: &PopulationGenes<f64>) -> PopulationFitness<f64> {
        // Sphere function: sum of squares for each individual
        let fitness_values = genes
            .map_axis(Axis(1), |individual| {
                individual.iter().map(|&x| x * x).sum::<f64>()
            })
            .insert_axis(Axis(1));
        fitness_values
    }

    // Constraints function
    fn constraints_fn(genes: &PopulationGenes<f64>) -> PopulationConstraints<f64> {
        // Constraint: sum of genes - 10 ≤ 0
        let sum_constraint = genes
            .sum_axis(Axis(1))
            .mapv(|sum| sum - 10.0)
            .insert_axis(Axis(1));

        // Constraint: genes ≥ 0 (represented as -x ≤ 0)
        let non_neg_constraints = genes.mapv(|x| -x);

        // Combine constraints into one array
        ndarray::concatenate(
            Axis(1),
            &[sum_constraint.view(), non_neg_constraints.view()],
        )
        .unwrap()
    }

    #[test]
    fn test_evaluator_evaluate_fitness() {
        let evaluator = Evaluator::<f64, f64, f64>::new(Box::new(fitness_fn), None);

        let population_genes = array![[1.0, 2.0], [3.0, 4.0], [0.0, 0.0],];

        let fitness = evaluator.evaluate_fitness(&population_genes);

        let expected_fitness = array![[5.0], [25.0], [0.0],];

        assert_eq!(fitness, expected_fitness);
    }

    #[test]
    fn test_evaluator_evaluate_constraints() {
        let evaluator =
            Evaluator::<f64, f64, f64>::new(Box::new(fitness_fn), Some(Box::new(constraints_fn)));

        let population_genes = array![
            [1.0, 2.0], // Feasible
            [3.0, 4.0], // Feasible
            [5.0, 6.0], // Infeasible (sum > 10)
        ];

        if let Some((constraints_array, feasibility_array)) =
            evaluator.evaluate_constraints(&population_genes)
        {
            let expected_constraints =
                array![[-7.0, -1.0, -2.0], [-3.0, -3.0, -4.0], [1.0, -5.0, -6.0],];

            let expected_feasibility = array![true, true, false];

            assert_eq!(constraints_array, expected_constraints);
            assert_eq!(feasibility_array, expected_feasibility);
        } else {
            panic!("Constraints function should not be None");
        }
    }

    #[test]
    fn test_evaluator_with_integer_constraints() {
        // Define fitness and constraints functions with integer types
        fn fitness_fn_int(genes: &PopulationGenes<i32>) -> PopulationFitness<i32> {
            let fitness_values = genes
                .map_axis(Axis(1), |individual| {
                    individual.iter().map(|&x| x * x).sum::<i32>()
                })
                .insert_axis(Axis(1));
            fitness_values
        }

        fn constraints_fn_int(genes: &PopulationGenes<i32>) -> PopulationConstraints<i32> {
            let sum_constraint = genes
                .sum_axis(Axis(1))
                .mapv(|sum| sum - 10)
                .insert_axis(Axis(1));

            let non_neg_constraints = genes.mapv(|x| -x);

            ndarray::concatenate(
                Axis(1),
                &[sum_constraint.view(), non_neg_constraints.view()],
            )
            .unwrap()
        }

        let evaluator = Evaluator::<i32, i32, i32>::new(
            Box::new(fitness_fn_int),
            Some(Box::new(constraints_fn_int)),
        );

        let population_genes = array![
            [1, 2], // Feasible
            [3, 4], // Feasible
            [5, 6], // Infeasible (sum > 10)
        ];

        if let Some((constraints_array, feasibility_array)) =
            evaluator.evaluate_constraints(&population_genes)
        {
            let expected_constraints = array![[-7, -1, -2], [-3, -3, -4], [1, -5, -6],];

            let expected_feasibility = array![true, true, false];

            assert_eq!(constraints_array, expected_constraints);
            assert_eq!(feasibility_array, expected_feasibility);
        } else {
            panic!("Constraints function should not be None");
        }
    }
}
