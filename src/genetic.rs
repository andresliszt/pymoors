use numpy::ndarray::{concatenate, Array1, Array2, ArrayViewMut1, Axis};

/// Represents an individual in the population.
/// Each `Genes` is an `Array1<f64>`.
pub type Genes = Array1<f64>;
pub type GenesMut<'a> = ArrayViewMut1<'a, f64>;

/// The `Parents` type represents the input for a binary genetic operator, such as a crossover operator.
/// It is a tuple of two 2-dimensional arrays (`Array2<f64>`) where `Parents.0[i]` will be operated with
/// `Parents.1[i]` for each `i` in the population size. Each array corresponds to one "parent" group
/// participating in the operation.
pub type Parents = (Array1<f64>, Array1<f64>);

/// The `Children` type defines the output of a binary genetic operator, such as the crossover operator.
/// It is a tuple of two 2-dimensional arrays (`Array2<f64>`) where each array represents the resulting
/// offspring derived from the corresponding parent arrays in `Parents`.
pub type Children = (Array1<f64>, Array1<f64>);

/// The `PopulationGenes` type defines the current set of individuals in the population.
/// It is represented as a 2-dimensional array (`Array2<f64>`), where each row corresponds to an individual.
pub type PopulationGenes = Array2<f64>;

/// Fitness associated to one Genes
pub type IndividualFitness = Array1<f64>;
/// PopulationGenes Fitness
pub type PopulationFitness = Array2<f64>;

pub type IndividualConstraints = Array1<f64>;

pub type PopulationConstraints = Array2<f64>;

pub struct Individual {
    pub genes: Genes,
    pub fitness: IndividualFitness,
    pub constraints: Option<IndividualConstraints>,
    pub rank: usize,
    pub crowding_distance: f64,
}

impl Individual {
    pub fn new(
        genes: Genes,
        fitness: IndividualFitness,
        constraints: Option<IndividualConstraints>,
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
#[derive(Debug)]
pub struct Population {
    pub genes: PopulationGenes,
    pub fitness: PopulationFitness,
    pub constraints: Option<PopulationConstraints>,
    pub rank: Array1<usize>,
    pub crowding_distance: Array1<f64>,
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            genes: self.genes.clone(),
            fitness: self.fitness.clone(),
            constraints: self.constraints.clone(),
            rank: self.rank.clone(),
            crowding_distance: self.crowding_distance.clone(),
        }
    }
}

impl Population {
    /// Creates a new `Population` instance with the given genes and fitness.
    /// `rank` and `crowding_distance` are initially `None`.
    pub fn new(
        genes: PopulationGenes,
        fitness: PopulationFitness,
        constraints: Option<PopulationConstraints>,
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
    pub fn get(&self, idx: usize) -> Individual {
        let constraints = self.constraints.as_ref().map(|c| c.row(idx).to_owned());

        Individual::new(
            self.genes.row(idx).to_owned(),
            self.fitness.row(idx).to_owned(),
            constraints,
            self.rank[idx],
            self.crowding_distance[idx],
        )
    }

    /// Returns a new `Population` containing only the individuals at the specified indices.
    /// Indices may be repeated, resulting in repeated individuals in the new population.
    pub fn selected(&self, indices: &[usize]) -> Population {
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
    /// Returns a new `Population` containing only the individuals with rank = 0.
    pub fn best(&self) -> Population {
        let indices: Vec<usize> = self
            .rank
            .iter()
            .enumerate()
            .filter_map(|(i, &rank)| if rank == 0 { Some(i) } else { None })
            .collect();
        self.selected(&indices)
    }
}

pub type Fronts = Vec<Population>;

/// An extension trait for the `Fronts` type to add a `.flatten()` method
/// that combines multiple fronts into a single `Population`.
pub trait FrontsExt {
    fn flatten_fronts(&self) -> Population;
}

impl FrontsExt for Vec<Population> {
    fn flatten_fronts(&self) -> Population {
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
