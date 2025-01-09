use crate::genetic::{
    Fronts, Genes, Individual, Population, PopulationFitness, PopulationGenes,
};
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::Debug;

pub mod crossover;
pub mod evolve;
pub mod mutation;
pub mod sampling;
pub mod selection;
pub mod survival;

pub trait GeneticOperator: Clone + Debug {
    fn name(&self) -> String;
}

pub trait SamplingOperator: GeneticOperator {
    /// Samples a single individual.
    fn sample_individual<R>(&self, rng: &mut R) -> Genes
    where
        R: Rng + Sized;

    /// Samples a population of individuals.
    fn operate<R>(&self, population_size: usize, rng: &mut R) -> PopulationGenes
    where
        R: Rng + Sized,
    {
        let mut population = Vec::with_capacity(population_size);

        // Sample individuals and collect them
        for _ in 0..population_size {
            let individual = self.sample_individual(rng);
            population.push(individual);
        }

        // Determine the number of genes per individual
        let num_genes = population[0].len();

        // Flatten the population into a single vector
        let flat_population: Vec<f64> = population
            .into_iter()
            .flat_map(|individual| individual.into_iter())
            .collect();

        // Create the shape: (number of individuals, number of genes)
        let shape = (population_size, num_genes);

        // Use from_shape_vec to create PopulationGenes
        let population_genes = PopulationGenes::from_shape_vec(shape, flat_population)
            .expect("Failed to create PopulationGenes from vector");

        population_genes
    }
}

pub trait MutationOperator: GeneticOperator {
    /// Mutates a single individual and returns the mutated individual.
    fn mutate<R>(&self, individual: &Genes, rng: &mut R) -> Genes
    where
        R: Rng + Sized;

    /// Selects individuals for mutation based on the mutation rate.
    fn _select_individuals_for_mutation<R>(
        &self,
        population_size: usize,
        mutation_rate: f64,
        rng: &mut R,
    ) -> Vec<bool>
    where
        R: Rng + Sized,
    {
        (0..population_size)
            .map(|_| rng.gen::<f64>() < mutation_rate)
            .collect()
    }

    /// Applies the mutation operator to the population.
    fn operate<R>(
        &self,
        population: &PopulationGenes,
        mutation_rate: f64,
        rng: &mut R,
    ) -> PopulationGenes
    where
        R: Rng + Sized,
    {
        // Step 1: Generate a boolean mask for mutation
        let mask: Vec<bool> =
            self._select_individuals_for_mutation(population.len(), mutation_rate, rng);

        // Step 2: Create a new population with mutated individuals
        let mut new_population = population.clone();
        new_population
            .outer_iter_mut()
            .enumerate()
            .for_each(|(i, mut individual)| {
                if mask[i] {
                    let mutated = self.mutate(&individual.to_owned(), rng);
                    individual.assign(&mutated);
                }
            });

        new_population
    }
}

pub trait CrossoverOperator: GeneticOperator {
    const N_OFFSPRINGS_PER_CROSSOVER: usize = 2;

    /// Performs crossover between two parents to produce two offspring.
    fn crossover<R>(
        &self,
        parent_a: &Genes,
        parent_b: &Genes,
        rng: &mut R,
    ) -> (Genes, Genes)
    where
        R: Rng + Sized;

    /// Applies the crossover operator to the population.
    /// Takes two parent populations and returns two offspring populations.
    fn operate<R>(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        rng: &mut R,
    ) -> PopulationGenes
    where
        R: Rng + Sized,
    {
        let population_size = parents_a.nrows();
        assert_eq!(
            population_size,
            parents_b.nrows(),
            "Parent populations must be of the same size"
        );

        let num_genes = parents_a.ncols();
        assert_eq!(
            num_genes,
            parents_b.ncols(),
            "Parent individuals must have the same number of genes"
        );

        // Prepare flat vectors to collect offspring genes
        let mut flat_offspring =
            Vec::with_capacity(Self::N_OFFSPRINGS_PER_CROSSOVER * population_size * num_genes);

        for i in 0..population_size {
            let parent_a = parents_a.row(i).to_owned();
            let parent_b = parents_b.row(i).to_owned();
            let (child_a, child_b) = self.crossover(&parent_a, &parent_b, rng);

            // Extend the flat vectors with the offspring genes
            flat_offspring.extend(child_a.into_iter());
            flat_offspring.extend(child_b.into_iter());
        }

        // Create PopulationGenes directly from the flat vectors
        let offspring_population = PopulationGenes::from_shape_vec(
            (
                Self::N_OFFSPRINGS_PER_CROSSOVER * population_size,
                num_genes,
            ),
            flat_offspring,
        )
        .expect("Failed to create offspring_population_a");
        offspring_population
    }
}

// Enum to represent the result of a tournament duel.
pub enum DuelResult {
    LeftWins,
    RightWins,
    Tie,
}

pub trait SelectionOperator: GeneticOperator {
    const PRESSURE: usize = 2;
    const N_PARENTS_PER_CROSSOVER: usize = 2;

    /// Selects random participants from the population for the tournaments.
    /// If `n_crossovers * pressure` is greater than the population size, it will create multiple permutations
    /// to ensure there are enough random indices.
    fn _select_participants<R>(
        &self,
        pop_size: usize,
        n_crossovers: usize,
        rng: &mut R,
    ) -> Vec<Vec<usize>>
    where
        R: Rng + Sized,
    {
        // Note that we have fixed n_parents = 2 and pressure = 2
        let total_needed = n_crossovers * Self::N_PARENTS_PER_CROSSOVER * Self::PRESSURE;
        let mut all_indices = Vec::with_capacity(total_needed);

        let n_perms = (total_needed + pop_size - 1) / pop_size; // Ceil division
        for _ in 0..n_perms {
            let mut perm: Vec<usize> = (0..pop_size).collect();
            perm.shuffle(rng);
            all_indices.extend_from_slice(&perm);
        }

        all_indices.truncate(total_needed);

        // Now split all_indices into chunks of size 2
        let mut result = Vec::with_capacity(n_crossovers);
        for chunk in all_indices.chunks(2) {
            // chunk is a slice of length 2
            result.push(vec![chunk[0], chunk[1]]);
        }

        result
    }

    /// Tournament between 2 individuals.
    fn tournament_duel(&self, p1: &Individual, p2: &Individual) -> DuelResult;

    fn operate<R>(
        &self,
        population: &Population,
        n_crossovers: usize,
        rng: &mut R,
    ) -> (Population, Population)
    where
        R: Rng + Sized,
    {
        let pop_size = population.len();

        let participants = self._select_participants(pop_size, n_crossovers, rng);

        let mut winners = Vec::with_capacity(n_crossovers);

        // For binary tournaments:
        // Each row of 'participants' is [p1, p2]
        for row in &participants {
            let ind_a = population.get(row[0]);
            let ind_b = population.get(row[1]);
            let duel_result = self.tournament_duel(&ind_a, &ind_b);
            let winner = match duel_result {
                DuelResult::LeftWins => row[0],
                DuelResult::RightWins => row[1],
                DuelResult::Tie => row[1], // TODO: use random?
            };
            winners.push(winner);
        }

        // Split winners into two halves
        let mid = winners.len() / 2;
        let first_half = &winners[..mid];
        let second_half = &winners[mid..];

        // Create two new populations based on the split
        let population_a = population.selected(first_half);
        let population_b = population.selected(second_half);

        (population_a, population_b)
    }
}

pub trait SurvivalOperator: GeneticOperator {
    /// Selects the individuals that will survive to the next generation.
    fn operate(&self, fronts: &Fronts, n_survive: usize) -> Population;
}
