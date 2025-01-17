use pyo3::prelude::*;
use numpy::PyArray2;
use numpy::IntoPyArray;

use crate::algorithms::MultiObjectiveAlgorithm;
use crate::genetic::{PopulationConstraints, PopulationFitness, PopulationGenes};
use crate::helpers::functions::{
    create_population_constraints_closure, create_population_fitness_closure,
};
use crate::helpers::parser::{
    unwrap_crossover_operator, unwrap_mutation_operator, unwrap_sampling_operator,
};
use crate::operators::selection::RankAndCrowdingSelection;
use crate::operators::survival::RankCrowdingSurvival;
use crate::operators::{CrossoverOperator, MutationOperator, SamplingOperator};

pub struct Nsga2 {
    algorithm: MultiObjectiveAlgorithm,
}

impl Nsga2 {
    pub fn new(
        sampler: Box<dyn SamplingOperator>,
        crossover: Box<dyn CrossoverOperator>,
        mutation: Box<dyn MutationOperator>,
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        n_vars: usize,
        pop_size: usize,
        n_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
    ) -> Self {
        let selector = Box::new(RankAndCrowdingSelection::new());
        let survivor = Box::new(RankCrowdingSurvival::new());
        let algorithm = MultiObjectiveAlgorithm::new(
            sampler,
            selector,
            survivor,
            crossover,
            mutation,
            fitness_fn,
            n_vars,
            pop_size,
            n_offsprings,
            num_iterations,
            mutation_rate,
            crossover_rate,
            constraints_fn,
        );
        Self { algorithm }
    }

    pub fn run(&mut self) {
        self.algorithm.run();
    }
}

#[pyclass(name = "Nsga2", unsendable)]
pub struct PyNsga2 {
    pub inner: Nsga2,
}

#[pymethods]
impl PyNsga2 {
    /// Python constructor:
    /// nsga2 = Nsga2(
    ///   sampler, crossover, mutation, fitness_fn,
    ///   constraints_fn=None,
    ///   pop_size=100, n_offsprings=50, num_iterations=200,
    ///   mutation_rate=0.1, crossover_rate=0.8
    /// )
    #[new]
    pub fn new(
        sampler: PyObject,
        crossover: PyObject,
        mutation: PyObject,
        fitness_fn: PyObject,
        n_vars: usize,
        pop_size: usize,
        n_offsprings: usize,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        constraints_fn: Option<PyObject>,
    ) -> PyResult<Self> {
        // 1) Unwrap the genetic operators
        let sampler_box = unwrap_sampling_operator(sampler)?;
        let crossover_box = unwrap_crossover_operator(crossover)?;
        let mutation_box = unwrap_mutation_operator(mutation)?;

        // 2) Build the MANDATORY population-level fitness closure
        let fitness_closure = create_population_fitness_closure(fitness_fn)?;

        // 3) Build OPTIONAL population-level constraints closure
        //    If None in Python, we store None in Rust. Otherwise, wrap it in Some.
        let constraints_closure = if let Some(py_obj) = constraints_fn {
            Some(create_population_constraints_closure(py_obj)?)
        } else {
            None
        };

        // 4) Call the Rust Nsga2::new
        let nsga2 = Nsga2::new(
            sampler_box,
            crossover_box,
            mutation_box,
            fitness_closure,
            n_vars,
            pop_size,
            n_offsprings,
            num_iterations,
            mutation_rate,
            crossover_rate,
            constraints_closure,
        );

        Ok(Self { inner: nsga2 })
    }

    /// Expose the .run() method to Python
    pub fn run(&mut self) {
        self.inner.run();
    }

    /// Expose the population genes as a numpy array to Python
    pub fn get_population_genes<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.inner.algorithm.population.genes.to_owned().into_pyarray(py)
    }

    /// Expose the population fitness as a numpy array to Python
    pub fn get_population_fitness<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
        self.inner.algorithm.population.fitness.to_owned().into_pyarray(py)
    }
}
