use numpy::ndarray::{concatenate, Array1, Array2, ArrayViewMut1, Axis};

/// Represents an individual in the population.
/// Each `IndividualGenes` is an `Array1<f64>`.
pub type IndividualGenes = Array1<f64>;
pub type IndividualGenesMut<'a> = ArrayViewMut1<'a, f64>;

/// Represents an individual with genes, fitness, constraints (if any),
/// rank, and an optional diversity metric.
pub struct Individual {
    pub genes: IndividualGenes,
    pub fitness: Array1<f64>,
    pub constraints: Option<Array1<f64>>,
    pub rank: usize,
    pub survival_score: Option<f64>,
}

impl Individual {
    pub fn new(
        genes: IndividualGenes,
        fitness: Array1<f64>,
        constraints: Option<Array1<f64>>,
        rank: usize,
        survival_score: Option<f64>,
    ) -> Self {
        Self {
            genes,
            fitness,
            constraints,
            rank,
            survival_score,
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
    pub survival_score: Option<Array1<f64>>,
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            genes: self.genes.clone(),
            fitness: self.fitness.clone(),
            constraints: self.constraints.clone(),
            rank: self.rank.clone(),
            survival_score: self.survival_score.clone(),
        }
    }
}

impl Population {
    /// Creates a new `Population` instance with the given genes, fitness, constraints, and rank.
    /// The `survival_score` field is set to `None` by default.
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
            survival_score: None, // Initialized to None by default.
        }
    }

    /// Retrieves an `Individual` from the population by index.
    pub fn get(&self, idx: usize) -> Individual {
        let constraints = self.constraints.as_ref().map(|c| c.row(idx).to_owned());
        let diversity = self.survival_score.as_ref().map(|dm| dm[idx]);
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
        let survival_score = self
            .survival_score
            .as_ref()
            .map(|dm| dm.select(Axis(0), indices));
        let constraints = self
            .constraints
            .as_ref()
            .map(|c| c.select(Axis(0), indices));

        Population::new(genes, fitness, constraints, rank).with_diversity(survival_score)
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

    /// Auxiliary method to chain the assignment of `survival_score` in the `selected` method.
    fn with_diversity(mut self, survival_score: Option<Array1<f64>>) -> Self {
        self.survival_score = survival_score;
        self
    }

    /// Updates the population's `survival_score` field.
    ///
    /// This method validates that the provided `diversity` vector has the same number of elements
    /// as individuals in the population. If not, it returns an error.
    pub fn set_survival_score(&mut self, diversity: Array1<f64>) -> Result<(), String> {
        if diversity.len() != self.len() {
            return Err(format!(
                "The diversity vector has length {} but the population contains {} individuals.",
                diversity.len(),
                self.len()
            ));
        }
        self.survival_score = Some(diversity);
        Ok(())
    }

    pub fn merge(population1: &Population, population2: &Population) -> Population {
        // Concatenate genes (assumed to be an Array2).
        let merged_genes = concatenate(
            Axis(0),
            &[population1.genes.view(), population2.genes.view()],
        )
        .expect("Failed to merge genes");

        // Concatenate fitness (assumed to be an Array2).
        let merged_fitness = concatenate(
            Axis(0),
            &[population1.fitness.view(), population2.fitness.view()],
        )
        .expect("Failed to merge fitness");

        // Concatenate rank (Array1).
        let merged_rank = concatenate(Axis(0), &[population1.rank.view(), population2.rank.view()])
            .expect("Failed to merge rank");

        // Merge constraints: both must be Some or both must be None.
        let merged_constraints = match (&population1.constraints, &population2.constraints) {
            (Some(c1), Some(c2)) => Some(
                concatenate(Axis(0), &[c1.view(), c2.view()]).expect("Failed to merge constraints"),
            ),
            (None, None) => None,
            _ => panic!("Mismatched population constraints: one is set and the other is None"),
        };

        // Merge survival_score: both must be Some or both must be None.
        let merged_survival_score = match (&population1.survival_score, &population2.survival_score)
        {
            (Some(s1), Some(s2)) => Some(
                concatenate(Axis(0), &[s1.view(), s2.view()])
                    .expect("Failed to merge survival scores"),
            ),
            (None, None) => None,
            _ => panic!("Mismatched population survival scores: one is set and the other is None"),
        };

        // Create the new Population with merged fields.
        Population::new(
            merged_genes,
            merged_fitness,
            merged_constraints,
            merged_rank,
        )
        .with_diversity(merged_survival_score)
    }
}

/// Type alias for a vector of `Population` representing multiple fronts.
pub type Fronts = Vec<Population>;

/// An extension trait for `Fronts` that adds a `.to_population()` method
/// which flattens multiple fronts into a single `Population`.
pub trait FrontsExt {
    fn to_population(self) -> Population;
}

impl FrontsExt for Vec<Population> {
    fn to_population(self) -> Population {
        self.into_iter()
            .reduce(|pop1, pop2| Population::merge(&pop1, &pop2))
            .expect("Error when merging population vector")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_individual_is_feasible() {
        // Individual with no constraints should be feasible.
        let ind1 = Individual::new(array![1.0, 2.0], array![0.5, 1.0], None, 0, None);
        assert!(
            ind1.is_feasible(),
            "Individual with no constraints should be feasible"
        );

        // Individual with constraints summing to <= 0 is feasible.
        let ind2 = Individual::new(
            array![1.0, 2.0],
            array![0.5, 1.0],
            Some(array![-1.0, 0.0]),
            0,
            None,
        );
        assert!(
            ind2.is_feasible(),
            "Constraints sum -1.0 should be feasible"
        );

        // Individual with constraints summing to > 0 is not feasible.
        let ind3 = Individual::new(
            array![1.0, 2.0],
            array![0.5, 1.0],
            Some(array![1.0, 0.1]),
            0,
            None,
        );
        assert!(
            !ind3.is_feasible(),
            "Constraints sum 1.1 should not be feasible"
        );
    }

    #[test]
    fn test_population_new_get_selected_len() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let rank = array![0, 1];
        let pop = Population::new(genes.clone(), fitness.clone(), None, rank.clone());

        // Test len()
        assert_eq!(pop.len(), 2, "Population should have 2 individuals");

        // Test get()
        let ind0 = pop.get(0);
        assert_eq!(ind0.genes, genes.row(0).to_owned());
        assert_eq!(ind0.fitness, fitness.row(0).to_owned());
        assert_eq!(ind0.rank, 0);

        // Test selected()
        let selected = pop.selected(&[1]);
        assert_eq!(
            selected.len(),
            1,
            "Selected population should have 1 individual"
        );
        let ind_selected = selected.get(0);
        assert_eq!(ind_selected.genes, array![3.0, 4.0]);
        assert_eq!(ind_selected.fitness, array![1.5, 2.0]);
        assert_eq!(ind_selected.rank, 1);
    }

    #[test]
    fn test_population_best() {
        // Create a population with three individuals and varying ranks.
        let genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]];
        // First and third individuals have rank 0, second has rank 1.
        let rank = array![0, 1, 0];
        let pop = Population::new(genes, fitness, None, rank);
        let best = pop.best();
        // Expect best population to contain only individuals with rank 0.
        assert_eq!(best.len(), 2, "Best population should have 2 individuals");
        for i in 0..best.len() {
            let ind = best.get(i);
            assert_eq!(
                ind.rank, 0,
                "All individuals in best population should have rank 0"
            );
        }
    }

    #[test]
    fn test_set_survival_score() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let rank = array![0, 1];
        let mut pop = Population::new(genes, fitness, None, rank);

        // Set a survival score vector with correct length.
        let diversity = array![0.1, 0.2];
        assert!(pop.set_survival_score(diversity.clone()).is_ok());
        assert_eq!(pop.survival_score.unwrap(), diversity);
    }

    #[test]
    fn test_set_survival_score_err() {
        // Create a population with two individuals.
        let genes = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness = array![[0.5, 1.0], [1.5, 2.0]];
        let rank = array![0, 1];
        let mut pop = Population::new(genes, fitness, None, rank);

        // Setting a survival score vector with incorrect length should error.
        let wrong_diversity = array![0.1];
        assert!(pop.set_survival_score(wrong_diversity).is_err());
    }

    #[test]
    fn test_population_merge() {
        // Create two populations.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let rank1 = array![0, 0];
        let pop1 = Population::new(genes1, fitness1, None, rank1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let rank2 = array![1, 1];
        let pop2 = Population::new(genes2, fitness2, None, rank2);

        let merged = Population::merge(&pop1, &pop2);
        assert_eq!(
            merged.len(),
            4,
            "Merged population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Merged genes do not match");

        let expected_fitness = array![[0.5, 1.0], [1.5, 2.0], [2.5, 3.0], [3.5, 4.0]];
        assert_eq!(
            merged.fitness, expected_fitness,
            "Merged fitness does not match"
        );

        let expected_rank = array![0, 0, 1, 1];
        assert_eq!(merged.rank, expected_rank, "Merged rank does not match");
    }

    #[test]
    fn test_fronts_ext_to_population() {
        // Create two fronts.
        let genes1 = array![[1.0, 2.0], [3.0, 4.0]];
        let fitness1 = array![[0.5, 1.0], [1.5, 2.0]];
        let rank1 = array![0, 0];
        let pop1 = Population::new(genes1, fitness1, None, rank1);

        let genes2 = array![[5.0, 6.0], [7.0, 8.0]];
        let fitness2 = array![[2.5, 3.0], [3.5, 4.0]];
        let rank2 = array![1, 1];
        let pop2 = Population::new(genes2, fitness2, None, rank2);

        let fronts: Fronts = vec![pop1.clone(), pop2.clone()];
        let merged = fronts.to_population();

        assert_eq!(
            merged.len(),
            4,
            "Flattened population should have 4 individuals"
        );

        let expected_genes = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        assert_eq!(merged.genes, expected_genes, "Flattened genes do not match");
    }
}
