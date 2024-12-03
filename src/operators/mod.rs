use crate::genetic::{Genes, GenesMut, PopulationGenes};
use rand::Rng;
use std::fmt::Debug;

pub mod mutation;
pub mod sampling;
pub mod survival;
pub mod crossover;

pub trait GeneticOperator: Clone + Debug {
    fn name(&self) -> String;
}

pub trait SamplingOperator<Dna>: GeneticOperator
where
    Dna: Clone + Debug + PartialEq + Send + Sync,
{
    /// Samples a single individual.
    fn sample_individual<R>(&self, rng: &mut R) -> Genes<Dna>
    where
        R: Rng + Sized;

    /// Samples a population of individuals.
    fn operate<R>(&self, population_size: usize, rng: &mut R) -> PopulationGenes<Dna>
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
        let flat_population: Vec<Dna> = population
            .into_iter()
            .flat_map(|individual| individual.into_iter())
            .collect();

        // Create the shape: (number of individuals, number of genes)
        let shape = (population_size, num_genes);

        // Use from_shape_vec to create PopulationGenes<Dna::Item>
        let population_genes = PopulationGenes::from_shape_vec(shape, flat_population)
            .expect("Failed to create PopulationGenes from vector");

        population_genes
    }
}

pub trait MutationOperator<Dna>: GeneticOperator
where
    Dna: Clone + Debug + PartialEq + Send + Sync,
{
    /// Mutates a single individual.
    fn mutate<R>(&self, individual: &mut GenesMut<Dna>, rng: &mut R)
    where
        R: Rng + Sized;

    /// Selects ¿´+dindividuals for mutation based on the mutation rate.
    fn select_individuals_for_mutation<R>(
        &self,
        population_size: usize,
        individual_mutation_rate: f64,
        rng: &mut R,
    ) -> Vec<usize>
    where
        R: Rng + Sized,
    {
        let dist = rand::distributions::Uniform::new(0.0, 1.0);
        (0..population_size)
            .filter(|_| rng.sample(dist) < individual_mutation_rate)
            .collect()
    }

    /// Applies the mutation operator to the population.
    fn operate<R>(
        &self,
        population: &mut PopulationGenes<Dna>,
        individual_mutation_rate: f64,
        rng: &mut R,
    ) where
        R: Rng + Sized,
    {
        let selected_indices =
            self.select_individuals_for_mutation(population.nrows(), individual_mutation_rate, rng);

        for &idx in &selected_indices {
            let mut individual = population.row_mut(idx);
            self.mutate(&mut individual, rng);
        }
    }
}


pub trait CrossoverOperator<Dna>: GeneticOperator
where
    Dna: Clone + Debug + PartialEq + Send + Sync,
{
    /// Performs crossover between two parents to produce two offspring.
    fn crossover<R>(
        &self,
        parent_a: &Genes<Dna>,
        parent_b: &Genes<Dna>,
        rng: &mut R,
    ) -> (Genes<Dna>, Genes<Dna>)
    where
        R: Rng + Sized;

    /// Applies the crossover operator to the population.
    /// Takes two parent populations and returns two offspring populations.
    fn operate<R>(
        &self,
        parents_a: &PopulationGenes<Dna>,
        parents_b: &PopulationGenes<Dna>,
        rng: &mut R,
    ) -> (PopulationGenes<Dna>, PopulationGenes<Dna>)
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
        let mut flat_offspring_a = Vec::with_capacity(population_size * num_genes);
        let mut flat_offspring_b = Vec::with_capacity(population_size * num_genes);

        for i in 0..population_size {
            let parent_a = parents_a.row(i).to_owned();
            let parent_b = parents_b.row(i).to_owned();
            let (child_a, child_b) = self.crossover(&parent_a, &parent_b, rng);

            // Extend the flat vectors with the offspring genes
            flat_offspring_a.extend(child_a.into_iter());
            flat_offspring_b.extend(child_b.into_iter());
        }

        // Create PopulationGenes<Dna> directly from the flat vectors
        let offspring_population_a = PopulationGenes::from_shape_vec(
            (population_size, num_genes),
            flat_offspring_a,
        )
        .expect("Failed to create offspring_population_a");

        let offspring_population_b = PopulationGenes::from_shape_vec(
            (population_size, num_genes),
            flat_offspring_b,
        )
        .expect("Failed to create offspring_population_b");

        (offspring_population_a, offspring_population_b)
    }
}
