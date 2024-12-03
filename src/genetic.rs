use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::Add;

use ndarray::{Array1, Array2, ArrayViewMut1, Axis};

/// Represents an individual in the population.
/// Each `Genes` is an `Array1<Dna>`, where `Dna` is the type of the genes.
pub type Genes<Dna> = Array1<Dna>;
pub type GenesMut<'a, Dna> = ArrayViewMut1<'a, Dna>;

/// The `Parents` type represents the input for a binary genetic operator, such as a crossover operator.
/// It is a tuple of two 2-dimensional arrays (`Array2<Dna>`) where `Parents.0[i]` will be operated with
/// `Parents.1[i]` for each `i` in the population size. Each array corresponds to one "parent" group
/// participating in the operation.
pub type Parents<Dna> = (Array1<Dna>, Array1<Dna>);
pub type ParentsMut<'a, Dna> = (ArrayViewMut1<'a, Dna>, ArrayViewMut1<'a, Dna>);
/// The `Children` type defines the output of a binary genetic operator, such as the crossover operator.
/// It is a tuple of two 2-dimensional arrays (`Array2<Dna>`) where each array represents the resulting
/// offspring derived from the corresponding parent arrays in `Parents`.
pub type Children<Dna> = (Array1<Dna>, Array1<Dna>);
pub type ChildrenMut<'a, Dna> = (ArrayViewMut1<'a, Dna>, ArrayViewMut1<'a, Dna>);

/// The `PopulationGenes` type defines the current set of individuals in the population.
/// It is represented as a 2-dimensional array (`Array2<Dna>`), where each row corresponds to an individual.
pub type PopulationGenes<Dna> = Array2<Dna>;

pub trait FitnessValue: PartialOrd + Copy + Add<Output = Self> + Debug {}

impl<T> FitnessValue for T where T: PartialOrd + Copy + Add<Output = Self> + Debug {}
/// Fitness associated to one Genes
pub type IndividualFitness<F> = Array1<F>;
/// PopulationGenes Fitness
pub type PopulationFitness<F> = Array2<F>;

pub type IndividualConstraints<F> = Array1<F>;

pub type PopulationConstraints<F> = Array2<F>;

/// The `Population` struct containing genes, fitness, rank, and crowding distance.
/// `rank` and `crowding_distance` are optional and may be set during the process.
pub struct Population<Dna, F>
where
    F: FitnessValue,
{
    pub genes: PopulationGenes<Dna>,
    pub fitness: PopulationFitness<F>,
    pub rank: Option<Array1<usize>>,
    pub crowding_distance: Option<Array1<f64>>,
}

impl<Dna, F> Population<Dna, F>
where
    Dna: Clone,
    F: FitnessValue + Clone,
{
    /// Creates a new `Population` instance with the given genes and fitness.
    /// `rank` and `crowding_distance` are initially `None`.
    pub fn new(genes: PopulationGenes<Dna>, fitness: PopulationFitness<F>) -> Self {
        Self {
            genes,
            fitness,
            rank: None,
            crowding_distance: None,
        }
    }

    /// Retains only the individuals at the specified indices,
    /// modifying the current `Population` in place.
    pub fn select(&mut self, indices: &Array1<usize>) {
        // Convert indices to a slice for compatibility
        let indices_slice = indices
            .as_slice()
            .expect("Indices should be contiguous in memory");

        // Filter genes based on indices
        self.genes = self.genes.select(Axis(0), indices_slice).to_owned();

        // Filter fitness based on indices
        self.fitness = self.fitness.select(Axis(0), indices_slice).to_owned();

        // Filter rank if available
        if let Some(ref mut rank) = self.rank {
            *rank = rank.select(Axis(0), indices_slice).to_owned();
        }

        // Filter crowding_distance if available
        if let Some(ref mut crowding_distance) = self.crowding_distance {
            *crowding_distance = crowding_distance.select(Axis(0), indices_slice).to_owned();
        }
    }
}
