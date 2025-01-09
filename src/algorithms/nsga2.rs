use crate::algorithms::MultiObjectiveAlgorithm;
use crate::operators::selection::RankAndCrowdingSelection;
use crate::operators::survival::RankCrowdingSurvival;
use crate::genetic::{PopulationGenes, PopulationFitness, PopulationConstraints};
use crate::operators::{CrossoverOperator, MutationOperator};
use std::fmt::Debug;

pub struct Nsga2<Cross, Mut>
where
    Cross: CrossoverOperator,
    Mut: MutationOperator,
{
    algorithm: MultiObjectiveAlgorithm<RankAndCrowdingSelection, RankCrowdingSurvival, Cross, Mut>,
}

impl<Cross, Mut> Nsga2<Cross, Mut>
where
    Cross: CrossoverOperator,
    Mut: MutationOperator,
{
    pub fn new(
        genes: PopulationGenes,
        crossover: Cross,
        mutation: Mut,
        fitness_fn: Box<dyn Fn(&PopulationGenes) -> PopulationFitness>,
        constraints_fn: Option<Box<dyn Fn(&PopulationGenes) -> PopulationConstraints>>,
        n_offsprings: i32,
        num_iterations: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        let selector = RankAndCrowdingSelection::new();
        let survivor = RankCrowdingSurvival::new();
        let algorithm = MultiObjectiveAlgorithm::new(
            genes,
            selector,
            survivor,
            crossover,
            mutation,
            fitness_fn,
            constraints_fn,
            n_offsprings,
            num_iterations,
            mutation_rate,
            crossover_rate,
        );
        Self { algorithm }
    }

    pub fn run(&mut self) {
        self.algorithm.run();
    }
}
