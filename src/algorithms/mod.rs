use numpy::ndarray::{concatenate, Axis};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::error::Error;
use std::fmt;

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;

use crate::algorithms::py_errors::NoFeasibleIndividualsError;
use crate::{
    evaluator::Evaluator,
    genetic::{FrontsExt, Population, PopulationConstraints, PopulationFitness, PopulationGenes},
    helpers::duplicates::PopulationCleaner,
    helpers::printer::print_minimum_objectives,
    operators::{
        evolve::Evolve, evolve::EvolveError, CrossoverOperator, MutationOperator, SamplingOperator,
        SelectionOperator, SurvivalOperator,
    },
};

mod macros;
pub mod nsga2;
pub mod nsga3;
pub mod py_errors;
pub mod rnsga2;

#[derive(Debug)]
pub enum MultiObjectiveAlgorithmError {
    Evolve(EvolveError),
    NoFeasibleIndividuals,
}

impl fmt::Display for MultiObjectiveAlgorithmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiObjectiveAlgorithmError::Evolve(msg) => {
                write!(f, "Error during evolution: {}", msg)
            }
            MultiObjectiveAlgorithmError::NoFeasibleIndividuals => {
                write!(f, "No feasible individuals found")
            }
        }
    }
}

impl From<EvolveError> for MultiObjectiveAlgorithmError {
    fn from(e: EvolveError) -> Self {
        MultiObjectiveAlgorithmError::Evolve(e)
    }
}

/// Once a new error is created to be exposed to the python side
/// the match must be updatetd
impl From<MultiObjectiveAlgorithmError> for PyErr {
    fn from(err: MultiObjectiveAlgorithmError) -> PyErr {
        match err {
            MultiObjectiveAlgorithmError::NoFeasibleIndividuals => {
                NoFeasibleIndividualsError::new_err(err.to_string())
            }
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

impl Error for MultiObjectiveAlgorithmError {}

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
    rng: StdRng,
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
        seed: Option<u64>,
    ) -> Result<Self, MultiObjectiveAlgorithmError> {
        let mut rng = seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64);
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

        let fronts = evaluator.build_fronts(genes);

        if fronts.is_empty() {
            return Err(MultiObjectiveAlgorithmError::NoFeasibleIndividuals);
        }

        let population = fronts.to_population();

        Ok(Self {
            population,
            survivor,
            evolve,
            evaluator,
            pop_size,
            n_offsprings,
            num_iterations,
            verbose,
            n_vars,
            rng,
        })
    }

    fn next(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        // Obtain offspring genes.
        let offspring_genes = self
            .evolve
            .evolve(&self.population, self.n_offsprings, 200, &mut self.rng)
            .map_err::<MultiObjectiveAlgorithmError, _>(Into::into)?;

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
        let mut fronts = self.evaluator.build_fronts(combined_genes);

        // Check if there are no feasible individuals
        if fronts.is_empty() {
            return Err(MultiObjectiveAlgorithmError::NoFeasibleIndividuals);
        }

        // Select the new population
        self.population = self.survivor.operate(&mut fronts, self.pop_size);
        Ok(())
    }

    pub fn run(&mut self) -> Result<(), MultiObjectiveAlgorithmError> {
        for current_iter in 0..self.num_iterations {
            match self.next() {
                Ok(()) => {
                    if self.verbose {
                        print_minimum_objectives(&self.population, current_iter + 1);
                    }
                }
                Err(MultiObjectiveAlgorithmError::Evolve(EvolveError::EmptyMatingResult {
                    message,
                    ..
                })) => {
                    println!("Warning: {}. Terminating the algorithm early.", message);
                    break;
                }
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}
