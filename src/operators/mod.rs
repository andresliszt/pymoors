use crate::genetic::{
    Fronts, Individual, IndividualGenes, IndividualGenesMut, Population, PopulationFitness,
    PopulationGenes,
};
use crate::random::RandomGenerator;
use numpy::ndarray::{Array1, Axis};
use rand::prelude::SliceRandom;
use std::fmt::Debug;

pub mod crossover;
pub mod evolve;
pub mod mutation;
pub mod py_operators;
pub mod sampling;
pub mod selection;
pub mod survival;

/// Keep these traits as object safe because python implementation needs dyn

pub trait GeneticOperator: Debug {
    fn name(&self) -> String;
}

pub trait SamplingOperator: GeneticOperator {
    /// Samples a single individual.
    fn sample_individual(&self, n_vars: usize, rng: &mut dyn RandomGenerator) -> IndividualGenes;

    /// Samples a population of individuals.
    fn operate(
        &self,
        pop_size: usize,
        n_vars: usize,
        rng: &mut dyn RandomGenerator,
    ) -> PopulationGenes {
        let mut population = Vec::with_capacity(pop_size);

        // Sample individuals and collect them
        for _ in 0..pop_size {
            let individual = self.sample_individual(n_vars, rng);
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
        let shape = (pop_size, num_genes);

        // Use from_shape_vec to create PopulationGenes
        let population_genes = PopulationGenes::from_shape_vec(shape, flat_population)
            .expect("Failed to create PopulationGenes from vector");

        population_genes
    }
}

/// MutationOperator defines an in-place mutation where the individual is modified directly.
pub trait MutationOperator: GeneticOperator {
    /// Mutates a single individual in place.
    ///
    /// # Arguments
    ///
    /// * `individual` - The individual to mutate, provided as a mutable view.
    /// * `rng` - A random number generator.
    fn mutate<'a>(&self, individual: IndividualGenesMut<'a>, rng: &mut dyn RandomGenerator);

    /// Selects individuals for mutation based on the mutation rate.
    fn select_individuals_for_mutation(
        &self,
        pop_size: usize,
        mutation_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) -> Vec<bool> {
        (0..pop_size).map(|_| rng.gen_bool(mutation_rate)).collect()
    }

    /// Applies the mutation operator to the entire population in place.
    ///
    /// # Arguments
    ///
    /// * `population` - The population as a mutable 2D array (each row represents an individual).
    /// * `mutation_rate` - The probability that an individual is mutated.
    /// * `rng` - A random number generator.
    fn operate(
        &self,
        population: &mut PopulationGenes,
        mutation_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) {
        // Get the number of individuals (i.e. the number of rows).
        let pop_size = population.len_of(Axis(0));
        // Generate a boolean mask for which individuals will be mutated.
        let mask: Vec<bool> = self.select_individuals_for_mutation(pop_size, mutation_rate, rng);

        // Iterate over the population using outer_iter_mut to get a mutable view for each row.
        for (i, mut individual) in population.outer_iter_mut().enumerate() {
            if mask[i] {
                // Pass a mutable view of the individual to the mutate method.
                self.mutate(individual.view_mut(), rng);
            }
        }
    }
}

pub trait CrossoverOperator: GeneticOperator {
    fn n_offsprings_per_crossover(&self) -> usize {
        2
    }

    /// Performs crossover between two parents to produce two offspring.
    fn crossover(
        &self,
        parent_a: &IndividualGenes,
        parent_b: &IndividualGenes,
        rng: &mut dyn RandomGenerator,
    ) -> (IndividualGenes, IndividualGenes);

    /// Applies the crossover operator to the population.
    /// Takes two parent populations and returns two offspring populations.
    /// Includes a `crossover_rate` to determine which pairs undergo crossover.
    fn operate(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        crossover_rate: f64,
        rng: &mut dyn RandomGenerator,
    ) -> PopulationGenes {
        let pop_size = parents_a.nrows();
        assert_eq!(
            pop_size,
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
            Vec::with_capacity(self.n_offsprings_per_crossover() * pop_size * num_genes);

        for i in 0..pop_size {
            let parent_a = parents_a.row(i).to_owned();
            let parent_b = parents_b.row(i).to_owned();

            if rng.gen_proability() <= crossover_rate {
                // Perform crossover
                let (child_a, child_b) = self.crossover(&parent_a, &parent_b, rng);
                flat_offspring.extend(child_a.into_iter());
                flat_offspring.extend(child_b.into_iter());
            } else {
                // Keep parents as offspring
                flat_offspring.extend(parent_a.into_iter());
                flat_offspring.extend(parent_b.into_iter());
            }
        }

        // Create PopulationGenes directly from the flat vectors
        let offspring_population = PopulationGenes::from_shape_vec(
            (self.n_offsprings_per_crossover() * pop_size, num_genes),
            flat_offspring,
        )
        .expect("Failed to create offspring population");
        offspring_population
    }
}

// Enum to represent the result of a tournament duel.
#[derive(Debug, PartialEq, Eq)]
pub enum DuelResult {
    LeftWins,
    RightWins,
    Tie,
}

pub trait SelectionOperator: GeneticOperator {
    fn pressure(&self) -> usize {
        2
    }

    fn n_parents_per_crossover(&self) -> usize {
        2
    }

    /// Selects random participants from the population for the tournaments.
    /// If `n_crossovers * pressure` is greater than the population size, it will create multiple permutations
    /// to ensure there are enough random indices.
    fn _select_participants(
        &self,
        pop_size: usize,
        n_crossovers: usize,
        rng: &mut dyn RandomGenerator,
    ) -> Vec<Vec<usize>> {
        // Note that we have fixed n_parents = 2 and pressure = 2
        let total_needed = n_crossovers * self.n_parents_per_crossover() * self.pressure();
        let mut all_indices = Vec::with_capacity(total_needed);

        let n_perms = (total_needed + pop_size - 1) / pop_size; // Ceil division
        for _ in 0..n_perms {
            let mut perm: Vec<usize> = (0..pop_size).collect();
            perm.shuffle(rng.rng());
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
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut dyn RandomGenerator,
    ) -> DuelResult;

    fn operate(
        &self,
        population: &Population,
        n_crossovers: usize,
        rng: &mut dyn RandomGenerator,
    ) -> (Population, Population) {
        let pop_size = population.len();

        let participants = self._select_participants(pop_size, n_crossovers, rng);

        let mut winners = Vec::with_capacity(n_crossovers);

        // For binary tournaments:
        // Each row of 'participants' is [p1, p2]
        for row in &participants {
            let ind_a = population.get(row[0]);
            let ind_b = population.get(row[1]);
            let duel_result = self.tournament_duel(&ind_a, &ind_b, rng);
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

/// Controls how the diversity (crowding) metric is compared during tournament selection.
#[derive(Clone, Debug)]
pub enum SurvivalScoringComparison {
    /// Larger survival scoring (e.g crowding sitance) is preferred.
    Maximize,
    /// Smaller survival scoring (crowding) metric is preferred.
    Minimize,
}

/// Enum to provide context to the survival score computation.
#[derive(Clone, Copy, Debug)]
pub enum FrontContext {
    First,     // The first (nondominated) front.
    Inner,     // Subsequent fronts that fit entirely.
    Splitting, // The front that must be partially selected (splitting).
}

/// The SurvivalOperator trait extends GeneticOperator and requires that concrete operators
/// provide a method for computing the survival score from a front's fitness.
pub trait SurvivalOperator: GeneticOperator {
    /// Returns whether the survival scoring should be maximized or minimized.
    fn scoring_comparison(&self) -> SurvivalScoringComparison {
        SurvivalScoringComparison::Maximize
    }

    /// Computes the survival score for a given front's fitness.
    /// This is the only method that needs to be overridden by each survival operator.
    fn survival_score(
        &self,
        front_fitness: &PopulationFitness,
        context: FrontContext,
        rng: &mut dyn RandomGenerator,
    ) -> Array1<f64>;

    /// Selects the individuals that will survive to the next generation.
    /// The default implementation uses the survival score to select individuals.
    fn operate(
        &self,
        fronts: &mut Fronts,
        n_survive: usize,
        rng: &mut dyn RandomGenerator,
    ) -> Population {
        // Drain all fronts and enumerate them.
        let drained = fronts.drain(..).enumerate();
        let mut survivors: Option<Population> = None;
        let mut n_survivors = 0;

        for (i, mut front) in drained {
            // Save the length of the current front before any move.
            let front_len = front.len();

            // Determine the context: first front is First, later fronts are Inner.
            let context = if i == 0 {
                FrontContext::First
            } else {
                FrontContext::Inner
            };

            if n_survivors + front_len <= n_survive {
                // The entire front fits.
                let score = self.survival_score(&front.fitness, context, rng);
                front
                    .set_survival_score(score)
                    .expect("Failed to set survival score");
                survivors = match survivors {
                    None => Some(front),
                    Some(existing) => Some(Population::merge(&existing, &front)),
                };
                n_survivors += front_len;
            } else {
                // Splitting front: only part of the front is needed.
                let remaining = n_survive - n_survivors;
                if remaining > 0 {
                    // Use Splitting context regardless of i.
                    let score = self.survival_score(&front.fitness, FrontContext::Splitting, rng);
                    front
                        .set_survival_score(score)
                        .expect("Failed to set survival score for splitting front");
                    // Clone survival_score vector for sorting.
                    let scores = front
                        .survival_score
                        .clone()
                        .expect("No survival score set for splitting front");
                    // Get indices for the current front.
                    let mut indices: Vec<usize> = (0..front_len).collect();
                    indices.sort_by(|&i, &j| match self.scoring_comparison() {
                        SurvivalScoringComparison::Maximize => scores[j]
                            .partial_cmp(&scores[i])
                            .unwrap_or(std::cmp::Ordering::Equal),
                        SurvivalScoringComparison::Minimize => scores[i]
                            .partial_cmp(&scores[j])
                            .unwrap_or(std::cmp::Ordering::Equal),
                    });
                    // Select exactly the required number of individuals.
                    let selected_indices: Vec<usize> =
                        indices.into_iter().take(remaining).collect();
                    let partial = front.selected(&selected_indices);
                    survivors = match survivors {
                        None => Some(partial),
                        Some(existing) => Some(Population::merge(&existing, &partial)),
                    };
                }
                break;
            }
        }
        survivors.expect("No survivors were selected")
    }
}
