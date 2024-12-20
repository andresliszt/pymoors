use crate::{
    evaluator::Evaluator,
    genetic::{Fitness, Population, PopulationGenes},
    operators::{CrossoverOperator, MutationOperator, SelectionOperator}
};
use rand::Rng;
use std::{fmt::Debug, marker::PhantomData};

pub struct Evolve<Dna, F, S, C, M>
where
    Dna: Clone + Debug,
    F: Fitness,
    M: MutationOperator<Dna>,
    C: CrossoverOperator<Dna>,
    S: SelectionOperator<Dna, F>,
{
    _f: PhantomData<F>,
    _d: PhantomData<Dna>,
    evaluator: Evaluator<Dna, F>,
    selection: S,
    crossover: C,
    mutation: M,
    mutation_rate: f64,
    crossover_rate: f64,
}

#[derive(Debug)]
pub enum EvolveError {
    EmptyMatingResult {
        message: String,
        current_offspring_count: usize,
        required_offsprings: usize,
    },
}

impl<Dna, F, S, C, M> Evolve<Dna, F, S, C, M>
where
    Dna: Clone + Debug,
    F: Fitness,
    M: MutationOperator<Dna>,
    C: CrossoverOperator<Dna>,
    S: SelectionOperator<Dna, F>,
{
    fn new(
        evaluator: Evaluator<Dna, F>,
        selection: S,
        crossover: C,
        mutation: M,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        Self {
            _f: PhantomData,
            _d: PhantomData,
            evaluator,
            selection,
            crossover,
            mutation,
            mutation_rate,
            crossover_rate,
        }
    }

    fn _mating<R>(
        &self,
        parents_a: &PopulationGenes<Dna>,
        parents_b: &PopulationGenes<Dna>,
        rng: &mut R,
    ) -> PopulationGenes<Dna>
    where
        R: Rng + Sized,
    {
        // Do the crossover
        let offsprings = self.crossover.operate(&parents_a, &parents_b, rng);
        // Do the mutation. Note that mutation mutates inplace
        let offsprings = self.mutation.operate(&offsprings, self.mutation_rate, rng);
        // Evaluate new population
        return offsprings;
    }

    pub fn evolve<R>(
        &self,
        population: &Population<Dna, F>,
        n_offsprings: usize,
        max_iter: usize,
        rng: &mut R,
    ) -> Result<PopulationGenes<Dna>, EvolveError>
    where
        R: Rng + Sized,
    {
        let num_genes = population.genes.ncols();
        let mut all_offsprings = Vec::with_capacity(n_offsprings);
        let mut iterations = 0;

        while all_offsprings.len() < n_offsprings && iterations < max_iter {
            let remaining = n_offsprings - all_offsprings.len();

            // Select parents
            let (parents_a, parents_b) = self.selection.operate(population, remaining, rng);

            // Generate new offspring
            let new_offsprings = self._mating(&parents_a.genes, &parents_b.genes, rng);

            // Clone rows into the vector to ensure ownership
            all_offsprings.extend(new_offsprings.outer_iter().map(|row| row.to_owned()));

            iterations += 1;
        }

        // If we exit the loop without achieving n_offsprings, adjust the number
        let achieved_offsprings = all_offsprings.len();

        if achieved_offsprings == 0 {
            return Err(EvolveError::EmptyMatingResult {
                message: "No offspring were generated.".to_string(),
                current_offspring_count: 0,
                required_offsprings: n_offsprings,
            });
        }

        // Convert the Vec back into an Array2
        let offspring_data: Vec<Dna> = all_offsprings.into_iter().flat_map(|row| row).collect();

        let offspring_array =
            PopulationGenes::from_shape_vec((achieved_offsprings, num_genes), offspring_data)
                .expect("Failed to create offspring array");

        if achieved_offsprings < n_offsprings {
            println!(
                "Warning: Only {} offspring were generated out of the desired {}.",
                achieved_offsprings, n_offsprings
            );
        }

        Ok(offspring_array)
    }
}
