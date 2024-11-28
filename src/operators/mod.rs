use crate::genetic::{Children, IndividualMut, ParentsMut, Population};
use rand::Rng;
use std::fmt::Debug;

pub mod mutation;

pub trait GeneticOperator: Clone + Debug {
    fn name(&self) -> String;
}

pub trait MutationOperator<Dna>: GeneticOperator
where
    Dna: Clone + Debug + PartialEq + Send + Sync,
{
    /// Mutates a single individual.
    fn mutate<R>(&self, individual: &mut IndividualMut<Dna>, rng: &mut R)
    where
        R: Rng + Sized;

    /// Selects individuals for mutation based on the mutation rate.
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
        population: &mut Population<Dna>,
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

// pub trait CrossOverOperator<Dna>: GeneticOperator
// where
//     Dna: Clone + Debug + PartialEq + Send + Sync,
// {
//     /// Mutates a single individual.
//     fn crossover<R>(&self, parents: &Parents<Dna>, rng: &mut R) -> Children<Dna>
//     where
//         R: Rng + Sized;

//     /// Selects individuals for mutation based on the mutation rate.
//     fn selector<R>(&self, population_size: usize, mutation_rate: f64, rng: &mut R) -> Vec<usize>
//     where
//         R: Rng + Sized,
//     {
//         let dist = rand::distributions::Uniform::new(0.0, 1.0);
//         (0..population_size)
//             .filter(|_| rng.sample(dist) < mutation_rate)
//             .collect()
//     }

//     /// Applies the mutation operator to the population.
//     fn operate<R>(
//         &self,
//         population: &Population<Dna>,
//         mutation_rate: f64,
//         rng: &mut R,
//     ) -> Population<Dna>
//     where
//         R: Rng + Sized,
//     {
//         let selected_indices = self.selector(population.nrows(), mutation_rate, rng);
//         if selected_indices.is_empty() {
//             return population.clone();
//         }

//         let mut mutated_population = population.clone();

//         for &idx in &selected_indices {
//             let individual = mutated_population.row(idx).to_owned();
//             let mutated_individual = self.mutate(&individual, rng);
//             mutated_population.row_mut(idx).assign(&mutated_individual);
//         }

//         mutated_population
//     }
// }
