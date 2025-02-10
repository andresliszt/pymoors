use numpy::ndarray::{concatenate, Array1, Array2, ArrayViewMut1, Axis};

/// Represents an individual in the population.
/// Each `Genes` is an `Array1<f64>`.
pub type Genes = Array1<f64>;
pub type GenesMut<'a> = ArrayViewMut1<'a, f64>;

/// Represents an individual with genes, fitness, constraints (if any),
/// rank, and an optional diversity metric.
pub struct Individual {
    pub genes: Genes,
    pub fitness: Array1<f64>,
    pub constraints: Option<Array1<f64>>,
    pub rank: usize,
    pub diversity_metric: Option<f64>,
}

impl Individual {
    pub fn new(
        genes: Genes,
        fitness: Array1<f64>,
        constraints: Option<Array1<f64>>,
        rank: usize,
        diversity_metric: Option<f64>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            diversity_metric,
        }
    }

    pub fn is_feasible(&self) -> bool {
        match &self.constraints {
            None => true,
            Some(c) => c.iter().sum::<f64>() <= 0.0,
        }
    }
}

/// Type aliases to work with populations.
pub type PopulationGenes = Array2<f64>;
pub type PopulationFitness = Array2<f64>;
pub type PopulationConstraints = Array2<f64>;

/// The `Population` struct contains genes, fitness, constraints (if any),
/// rank, and optionally a diversity metric vector.
#[derive(Debug)]
pub struct Population {
    pub genes: PopulationGenes,
    pub fitness: PopulationFitness,
    pub constraints: Option<PopulationConstraints>,
    pub rank: Array1<usize>,
    pub diversity_metric: Option<Array1<f64>>,
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            genes: self.genes.clone(),
            fitness: self.fitness.clone(),
            constraints: self.constraints.clone(),
            rank: self.rank.clone(),
            diversity_metric: self.diversity_metric.clone(),
        }
    }
}

impl Population {
    /// Creates a new `Population` instance with the given genes, fitness, constraints, and rank.
    /// The `diversity_metric` field is set to `None` by default.
    pub fn new(
        genes: PopulationGenes,
        fitness: PopulationFitness,
        constraints: Option<PopulationConstraints>,
        rank: Array1<usize>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            diversity_metric: None, // Initialized to None by default.
        }
    }

    /// Retrieves an `Individual` from the population by index.
    pub fn get(&self, idx: usize) -> Individual {
        let constraints = self.constraints.as_ref().map(|c| c.row(idx).to_owned());
        let diversity = self.diversity_metric.as_ref().map(|dm| dm[idx]);
        Individual::new(
            self.genes.row(idx).to_owned(),
            self.fitness.row(idx).to_owned(),
            constraints,
            self.rank[idx],
            diversity,
        )
    }

    /// Returns a new `Population` containing only the individuals at the specified indices.
    pub fn selected(&self, indices: &[usize]) -> Population {
        let genes = self.genes.select(Axis(0), indices);
        let fitness = self.fitness.select(Axis(0), indices);
        let rank = self.rank.select(Axis(0), indices);
        let diversity_metric = self
            .diversity_metric
            .as_ref()
            .map(|dm| dm.select(Axis(0), indices));
        let constraints = self
            .constraints
            .as_ref()
            .map(|c| c.select(Axis(0), indices));

        Population::new(genes, fitness, constraints, rank).with_diversity(diversity_metric)
    }

    /// Returns the number of individuals in the population.
    pub fn len(&self) -> usize {
        self.genes.nrows()
    }

    /// Returns a new `Population` containing only the individuals with rank = 0.
    pub fn best(&self) -> Population {
        let indices: Vec<usize> = self
            .rank
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| if r == 0 { Some(i) } else { None })
            .collect();
        self.selected(&indices)
    }

    /// Auxiliary method to chain the assignment of `diversity_metric` in the `selected` method.
    fn with_diversity(mut self, diversity_metric: Option<Array1<f64>>) -> Self {
        self.diversity_metric = diversity_metric;
        self
    }

    /// Updates the population's `diversity_metric` field.
    ///
    /// This method validates that the provided `diversity` vector has the same number of elements
    /// as individuals in the population. If not, it returns an error.
    pub fn set_diversity(&mut self, diversity: Array1<f64>) -> Result<(), String> {
        if diversity.len() != self.len() {
            return Err(format!(
                "The diversity vector has length {} but the population contains {} individuals.",
                diversity.len(),
                self.len()
            ));
        }
        self.diversity_metric = Some(diversity);
        Ok(())
    }
}

/// Type alias for a vector of `Population` representing multiple fronts.
pub type Fronts = Vec<Population>;

/// An extension trait for `Fronts` that adds a `.to_population()` method
/// which flattens multiple fronts into a single `Population`.
pub trait FrontsExt {
    fn to_population(&self) -> Population;
}

impl FrontsExt for Vec<Population> {
    fn to_population(&self) -> Population {
        if self.is_empty() {
            panic!("Cannot flatten empty fronts!");
        }

        let has_constraints = self[0].constraints.is_some();
        let has_diversity = self[0].diversity_metric.is_some();

        let mut genes_views = Vec::new();
        let mut fitness_views = Vec::new();
        let mut rank_views = Vec::new();
        let mut diversity_views = Vec::new();
        let mut constraints_views = Vec::new();

        for front in self.iter() {
            genes_views.push(front.genes.view());
            fitness_views.push(front.fitness.view());
            rank_views.push(front.rank.view());
            if has_diversity {
                let dm = front
                    .diversity_metric
                    .as_ref()
                    .expect("Inconsistent diversity_metric among fronts");
                diversity_views.push(dm.view());
            }
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
        let merged_rank =
            concatenate(Axis(0), &rank_views[..]).expect("Error concatenating rank arrays");

        let merged_diversity = if has_diversity {
            Some(
                concatenate(Axis(0), &diversity_views[..])
                    .expect("Error concatenating diversity arrays"),
            )
        } else {
            None
        };

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
            diversity_metric: merged_diversity,
        }
    }
}
