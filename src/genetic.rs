use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::Add;

use ndarray::{concatenate, Array1, Array2, ArrayViewMut1, Axis};

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

pub trait Fitness: PartialOrd + Copy + Add<Output = Self> + Debug {}

impl<F> Fitness for F where F: PartialOrd + Copy + Add<Output = Self> + Debug {}

/// Fitness associated to one Genes
pub type IndividualFitness<F> = Array1<F>;
/// PopulationGenes Fitness
pub type PopulationFitness<F> = Array2<F>;

pub trait ConstraintsValue: PartialOrd + Copy + Add<Output = Self> + Debug {}

impl<G> ConstraintsValue for G where G: PartialOrd + Copy + Add<Output = Self> + Debug {}

pub type IndividualConstraints<G> = Array1<G>;

pub type PopulationConstraints<G> = Array2<G>;

pub struct Individual<Dna, F>
where
    F: Fitness,
{
    pub genes: Genes<Dna>,
    pub fitness: IndividualFitness<F>,
    pub constraints: Option<IndividualConstraints<f64>>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl<Dna, F> Individual<Dna, F>
where
    F: Fitness,
{
    pub fn new(
        genes: Genes<Dna>,
        fitness: IndividualFitness<F>,
        constraints: Option<IndividualConstraints<f64>>,
        rank: usize,
        crowding_distance: f64,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            crowding_distance,
        }
    }

    pub fn is_feasible(&self) -> bool {
        match &self.constraints {
            None => true,
            Some(c) => {
                let sum: f64 = c.iter().sum();
                sum <= 0.0
            }
        }
    }
}

/// The `Population` struct containing genes, fitness, rank, and crowding distance.
/// `rank` and `crowding_distance` are optional and may be set during the process.
pub struct Population<Dna, F>
where
    F: Fitness,
{
    pub genes: PopulationGenes<Dna>,
    pub fitness: PopulationFitness<F>,
    pub constraints: Option<PopulationConstraints<f64>>,
    pub rank: Array1<usize>,
    pub crowding_distance: Array1<f64>,
}

impl<Dna, F> Population<Dna, F>
where
    Dna: Clone,
    F: Fitness + Clone,
{
    /// Creates a new `Population` instance with the given genes and fitness.
    /// `rank` and `crowding_distance` are initially `None`.
    pub fn new(
        genes: PopulationGenes<Dna>,
        fitness: PopulationFitness<F>,
        constraints: Option<PopulationConstraints<f64>>,
        rank: Array1<usize>,
        crowding_distance: Array1<f64>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            crowding_distance,
        }
    }

    /// Retrieves an `Individual` from the population by index.
    pub fn get(&self, idx: usize) -> Individual<Dna, F> {
        let constraints = self.constraints.as_ref().map(|c| c.row(idx).to_owned());

        Individual::new(
            self.genes.row(idx).to_owned(),
            self.fitness.row(idx).to_owned(),
            constraints,
            self.rank[idx],
            self.crowding_distance[idx],
        )
    }
    /// Retains only the individuals at the specified indices,
    /// modifying the current `Population` in place.
    /// TODO: Change indices to be Vec<usize>
    pub fn select(&mut self, indices: &Array1<usize>) {
        // Convert indices to a slice for compatibility
        let indices_slice = indices
            .as_slice()
            .expect("Indices should be contiguous in memory");

        // Filter genes based on indices
        self.genes = self.genes.select(Axis(0), indices_slice).to_owned();
        // Filter fitness based on indices
        self.fitness = self.fitness.select(Axis(0), indices_slice).to_owned();
        // Filter rank
        self.rank = self.rank.select(Axis(0), indices_slice).to_owned();
        // Filter crowding_distance
        self.crowding_distance = self
            .crowding_distance
            .select(Axis(0), indices_slice)
            .to_owned()
    }

    /// Returns a new `Population` containing only the individuals at the specified indices.
    /// Indices may be repeated, resulting in repeated individuals in the new population.
    pub fn selected(&self, indices: &[usize]) -> Population<Dna, F> {
        let genes = self.genes.select(Axis(0), indices);
        let fitness = self.fitness.select(Axis(0), indices);
        let rank = self.rank.select(Axis(0), indices);
        let crowding_distance = self.crowding_distance.select(Axis(0), indices);

        let constraints = self
            .constraints
            .as_ref()
            .map(|c| c.select(Axis(0), indices));

        Population::new(genes, fitness, constraints, rank, crowding_distance)
    }

    /// Returns the number of individuals in this population.
    pub fn len(&self) -> usize {
        self.genes.nrows()
    }
}

pub type Fronts<Dna, F> = Vec<Population<Dna, F>>;

/// An extension trait for the `Fronts<Dna, F>` type to add a `.flatten()` method
/// that combines multiple fronts into a single `Population<Dna, F>`.
pub trait FrontsExt<Dna, F>
where
    F: Fitness,
{
    fn flatten_fronts(&self) -> Population<Dna, F>;
}

impl<Dna, F> FrontsExt<Dna, F> for Vec<Population<Dna, F>>
where
    Dna: Clone,
    F: Fitness + Clone, // Make sure F: PartialOrd, Copy, etc. is in your Fitness trait
{
    fn flatten_fronts(&self) -> Population<Dna, F> {
        if self.is_empty() {
            panic!("Cannot flatten empty fronts!");
        }

        let has_constraints = self[0].constraints.is_some();

        let mut genes_views = Vec::new();
        let mut fitness_views = Vec::new();
        let mut rank_views = Vec::new();
        let mut cd_views = Vec::new();
        let mut constraints_views = Vec::new();

        for front in self.iter() {
            genes_views.push(front.genes.view());
            fitness_views.push(front.fitness.view());
            rank_views.push(front.rank.view());
            cd_views.push(front.crowding_distance.view());

            if has_constraints {
                let c = front
                    .constraints
                    .as_ref()
                    .expect("Inconsistent constraints among fronts");
                constraints_views.push(c.view());
            }
        }

        let merged_genes =
            concatenate(Axis(0), &genes_views[..]).expect("Error concatenating genes");
        let merged_fitness =
            concatenate(Axis(0), &fitness_views[..]).expect("Error concatenating fitness");

        // **Concatenate** (Axis(0)) for 1D arrays rank & cd:
        let merged_rank =
            concatenate(Axis(0), &rank_views[..]).expect("Error concatenating rank arrays"); // 1D result
        let merged_cd = concatenate(Axis(0), &cd_views[..]).expect("Error concatenating cd arrays"); // 1D result

        let merged_constraints = if has_constraints {
            Some(
                concatenate(Axis(0), &constraints_views[..])
                    .expect("Error concatenating constraints"),
            )
        } else {
            None
        };

        Population {
            genes: merged_genes,
            fitness: merged_fitness,
            constraints: merged_constraints,
            rank: merged_rank,
            crowding_distance: merged_cd,
        }
    }
}
