use numpy::ndarray::{concatenate, Axis};
use rand::thread_rng;
use rand::Rng;

use crate::{
    evaluator::Evaluator,
    genetic::{FrontsExt, Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    helpers::duplicates::PopulationCleaner,
    helpers::printer::print_minimum_objectives,
    operators::{
        evolve::Evolve, CrossoverOperator, MutationOperator, SamplingOperator, SelectionOperator,
        SurvivalOperator,
    },
};

mod macros;
pub mod nsga2;
pub mod nsga3;

pub struct MultiObjectiveAlgorithm {
    population: Population,
    survivor: Box<dyn SurvivalOperator>,
    evolve: Evolve,
    evaluator: Evaluator,
    pop_size: usize,
    n_offsprings: usize,
    num_iterations: usize,
    verbose: bool,
    n_vars: usize,
}

impl MultiObjectiveAlgorithm {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sampler: Box<dyn SamplingOperator>,
        selector: Box<dyn SelectionOperator>,
        survivor: Box<dyn SurvivalOperator>,
        crossover: Box<dyn CrossoverOperator>,
        mutation: Box<dyn MutationOperator>,
        duplicates_cleaner: Option<Box<dyn PopulationCleaner>>,
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        n_vars: usize,
        pop_size: usize,
        n_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        keep_infeasible: bool,
        verbose: bool,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
        // Optional lower and upper bounds for each gene.
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Self {
        let mut rng = thread_rng();
        let mut genes = sampler.operate(pop_size, n_vars, &mut rng);

        // Create the evolution operator.
        let evolve = Evolve::new(
            selector,
            crossover,
            mutation,
            duplicates_cleaner,
            mutation_rate,
            crossover_rate,
        );

        // Clean duplicates if the cleaner is enabled.
        genes = evolve.clean_duplicates(genes, None);

        let evaluator = Evaluator::new(
            fitness_fn,
            constraints_fn,
            keep_infeasible,
            lower_bound,
            upper_bound,
        );
        let population = evaluator.build_fronts(genes).flatten_fronts();
        Self {
            population,
            survivor,
            evolve,
            evaluator,
            pop_size,
            n_offsprings,
            num_iterations,
            verbose,
            n_vars,
        }
    }

    fn next<R: Rng>(&mut self, rng: &mut R) {
        // Obtain offspring genes.
        let offspring_genes =
            match self
                .evolve
                .evolve(&self.population, self.n_offsprings as usize, 200, rng)
            {
                Ok(genes) => genes,
                Err(e) => {
                    eprintln!("Error during evolution: {:?}", e);
                    return;
                }
            };

        // Validate that the number of columns in offspring_genes matches n_vars.
        assert_eq!(
            offspring_genes.ncols(),
            self.n_vars,
            "Number of columns in offspring_genes ({}) does not match n_vars ({})",
            offspring_genes.ncols(),
            self.n_vars
        );

        // Combine the current population with the offspring.
        let combined_genes = concatenate(
            Axis(0),
            &[self.population.genes.view(), offspring_genes.view()],
        )
        .expect("Failed to concatenate current population genes with offspring genes");
        // Build fronts from the combined genes.
        let fronts = self.evaluator.build_fronts(combined_genes);
        // Select the new population from the fronts.
        self.population = self.survivor.operate(&fronts, self.pop_size);
    }

    pub fn run(&mut self) {
        let mut rng = thread_rng();
        let mut current_iter = 0;
        while current_iter < self.num_iterations {
            self.next(&mut rng);
            current_iter += 1;
            if self.verbose {
                print_minimum_objectives(&self.population, current_iter);
            }
        }
    }
}
