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
}

impl MultiObjectiveAlgorithm {
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
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
    ) -> Self {
        // build the initial population from its genes
        let mut rng = thread_rng();
        let genes = sampler.operate(pop_size, n_vars, &mut rng);
        let pop_size = genes.len();
        let evolve = Evolve::new(
            selector,
            crossover,
            mutation,
            duplicates_cleaner,
            mutation_rate,
            crossover_rate,
        );
        let evaluator = Evaluator::new(fitness_fn, constraints_fn, keep_infeasible);
        let population = evaluator.build_fronts(genes).flatten_fronts();
        Self {
            population,
            survivor,
            evolve,
            evaluator,
            pop_size,
            n_offsprings,
            num_iterations,
        }
    }

    fn _next<R: Rng>(&mut self, rng: &mut R) {
        let genes = match self
            .evolve
            .evolve(&self.population, self.n_offsprings as usize, 200, rng)
        {
            Ok(genes) => genes,
            Err(e) => {
                eprintln!("Error during evolution: {:?}", e);
                return;
            }
        };

        let fronts = self.evaluator.build_fronts(genes);
        self.population = self.survivor.operate(&fronts, self.pop_size);
    }

    pub fn run(&mut self) {
        let mut rng = thread_rng();
        let mut current_iter = 0;
        while current_iter < self.num_iterations {
            self._next(&mut rng);
            current_iter += 1;
            print_minimum_objectives(&self.population, current_iter);
        }
    }
}
