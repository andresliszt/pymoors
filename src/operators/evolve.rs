use rand::RngCore;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
    genetic::{Population, PopulationGenes},
    helpers::duplicates::PopulationCleaner,
    operators::{CrossoverOperator, MutationOperator, SelectionOperator},
};

#[derive(Debug)]
pub struct Evolve {
    selection: Box<dyn SelectionOperator>,
    crossover: Box<dyn CrossoverOperator>,
    mutation: Box<dyn MutationOperator>,
    pub duplicates_cleaner: Option<Box<dyn PopulationCleaner>>,
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

impl Evolve {
    pub fn new(
        selection: Box<dyn SelectionOperator>,
        crossover: Box<dyn CrossoverOperator>,
        mutation: Box<dyn MutationOperator>,
        duplicates_cleaner: Option<Box<dyn PopulationCleaner>>,
        mutation_rate: f64,
        crossover_rate: f64,
    ) -> Self {
        Self {
            selection,
            crossover,
            mutation,
            duplicates_cleaner,
            mutation_rate,
            crossover_rate,
        }
    }

    /// Single-step crossover + mutation for a batch of selected parents.
    fn mating_batch(
        &self,
        parents_a: &PopulationGenes,
        parents_b: &PopulationGenes,
        rng: &mut dyn RngCore,
    ) -> PopulationGenes {
        // 1) Perform crossover in one batch
        let offsprings = self
            .crossover
            .operate(parents_a, parents_b, self.crossover_rate, rng);
        // 2) Perform mutation in one batch (often in-place)
        let offsprings = self.mutation.operate(&offsprings, self.mutation_rate, rng);

        offsprings
    }

    /// Generates up to `n_offsprings` in multiple iterations (up to `max_iter`).
    /// We accumulate offspring in a buffer and only remove duplicates once at the end.
    /// Prints separate durations for mating and duplicates cleaning, plus iteration count.
    pub fn evolve(
        &self,
        population: &Population,
        n_offsprings: usize,
        max_iter: usize,
        rng: &mut dyn RngCore,
    ) -> Result<PopulationGenes, EvolveError> {
        // We'll accumulate offspring rows in a Vec<Vec<f64>>
        let mut all_offsprings: Vec<Vec<f64>> = Vec::with_capacity(n_offsprings);
        let num_genes = population.genes.ncols();
        let mut iterations = 0;

        // Repeatedly create batches until we reach n_offsprings or exhaust max_iter
        while all_offsprings.len() < n_offsprings && iterations < max_iter {
            let remaining = n_offsprings - all_offsprings.len();

            // Select parents in a batch (size = `remaining` or some multiple)
            let (parents_a, parents_b) = self.selection.operate(population, remaining, rng);

            // Create offspring from these parents
            let new_offsprings = self.mating_batch(&parents_a.genes, &parents_b.genes, rng);

            // Extend our accumulator with the new rows
            for row in new_offsprings.outer_iter() {
                all_offsprings.push(row.to_vec());
            }

            // Print the number of offspring generated in this iteration
            println!(
                "Iteration {}: Generated {} offspring",
                iterations + 1,
                new_offsprings.nrows()
            );

            // Clean duplicates if a cleaner is provided
            if let Some(ref cleaner) = self.duplicates_cleaner {
                // Mide el tiempo del duplicates_cleaner
                let cleaner_start = Instant::now();

                // Convert Vec<Vec<f64>> into a single Array2
                let offspring_data: Vec<f64> =
                    all_offsprings.clone().into_iter().flatten().collect();
                let mut offspring_array = PopulationGenes::from_shape_vec(
                    (all_offsprings.len(), num_genes),
                    offspring_data,
                )
                .expect("Failed to create offspring array from the accumulated data");

                // Clean duplicates
                offspring_array = cleaner.remove(&offspring_array);

                // Update all_offsprings with the cleaned data
                all_offsprings = offspring_array
                    .outer_iter()
                    .map(|row| row.to_vec())
                    .collect();

                let cleaner_duration = cleaner_start.elapsed();
                println!("Duplicates cleaner took: {:?}", cleaner_duration);
            }

            iterations += 1;
        }

        println!("Total iterations: {}", iterations);

        if all_offsprings.is_empty() {
            // We never generated anything
            return Err(EvolveError::EmptyMatingResult {
                message: "No offspring were generated.".to_string(),
                current_offspring_count: 0,
                required_offsprings: n_offsprings,
            });
        }

        // Convert Vec<Vec<f64>> into a single Array2
        let all_offsprings_len = all_offsprings.len();
        let offspring_data: Vec<f64> = all_offsprings.into_iter().flatten().collect();

        // Build the final matrix
        let offspring_array =
            PopulationGenes::from_shape_vec((all_offsprings_len, num_genes), offspring_data)
                .expect("Failed to create offspring array from the accumulated data");

        // If we still have nothing after cleaning, return an error
        if offspring_array.nrows() == 0 {
            return Err(EvolveError::EmptyMatingResult {
                message: "No offspring were generated after removing duplicates.".to_string(),
                current_offspring_count: 0,
                required_offsprings: n_offsprings,
            });
        }

        // Warn if we didn't achieve the desired number
        if offspring_array.nrows() < n_offsprings {
            println!(
                "Warning: Only {} offspring were generated out of the desired {}.",
                offspring_array.nrows(),
                n_offsprings
            );
        }

        Ok(offspring_array)
    }
}
