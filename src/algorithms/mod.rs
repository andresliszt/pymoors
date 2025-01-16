use rand::thread_rng;
use rand::Rng;

use crate::{
    evaluator::Evaluator,
    genetic::{FrontsExt, Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    operators::{
        evolve::Evolve, CrossoverOperator, MutationOperator, SelectionOperator, SurvivalOperator, SamplingOperator
    },
};

pub mod nsga2;

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
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
        pop_size: usize,
        n_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        // build the initial population from its genes
        let mut rng = thread_rng();
        let genes = sampler.operate(pop_size, &mut rng);
        let pop_size = genes.len();
        let evolve = Evolve::new(selector, crossover, mutation, mutation_rate, crossover_rate);
        let evaluator = Evaluator::new(fitness_fn, constraints_fn);
        let population = evaluator.build_fronts(&genes).flatten_fronts();
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
        // Do the mating
        let genes = match self
            .evolve
            .evolve(&self.population, self.n_offsprings as usize, 100, rng)
        {
            Ok(genes) => genes,
            Err(e) => {
                eprintln!("Error during evolution: {:?}", e);
                return;
            }
        };
        // Compute the fronts
        let fronts = self.evaluator.build_fronts(&genes);
        // Let the survival operator operate
        self.population = self.survivor.operate(&fronts, self.pop_size);
    }

    pub fn run(&mut self) {
        let mut rng = thread_rng();
        let mut current_iter = 0;
        while current_iter < self.num_iterations {
            self._next(&mut rng);
            current_iter += 1;
        }
    }
}
