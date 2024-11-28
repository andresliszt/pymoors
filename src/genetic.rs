use numpy::ndarray::{Array1, Array2, ArrayViewMut1, ArrayViewMut2};

/// Represents an individual in the population.
/// Each `Individual` is an `Array1<Dna>`, where `Dna` is the type of the genes.
pub type Individual<Dna> = Array1<Dna>;
pub type IndividualMut<'a, Dna> = ArrayViewMut1<'a, Dna>;

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

/// The `Population` type defines the current set of individuals in the population.
/// It is represented as a 2-dimensional array (`Array2<Dna>`), where each row corresponds to an individual.
pub type Population<Dna> = Array2<Dna>;

/// Fitness associated to one Individual
pub trait Fitness: PartialOrd + Copy + Debug + Send + Sync {}
