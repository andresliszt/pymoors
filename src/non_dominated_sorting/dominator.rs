use numpy::ndarray::{Array2, ArrayView1, ArrayView2, Zip};

/// Handles Pareto dominated solutions
pub struct Dominator {}

impl Dominator {
    /// Compares two array of objectives `f1` and `f2` and determine if `f1`
    /// dominates `f2` or vice versa. We say that `f1 `dominates `f2` if for all
    /// index `i` `f1[i] < f2[i]`. Analogous when `f2` dominates `f1`.
    pub fn who_dominates(f1: ArrayView1<'_, f64>, f2: ArrayView1<'_, f64>) -> i8 {
        if f1.len() != f2.len() {
            panic!(
                "Both array of objectives must have same length. Got len f1 {} and len f2 {}",
                f1.len(),
                f2.len()
            )
        } else {
            if Zip::from(&f1).and(&f2).all(|&f1, &f2| f1 < f2) {
                return 1;
            }
            if Zip::from(&f1).and(&f2).all(|&f1, &f2| f1 > f2) {
                return -1;
            }
            0
        }
    }

    pub fn domination_matrix(population_objectives: ArrayView2<'_, f64>) -> Array2<i8> {
        // get population number
        let n_population = population_objectives.shape()[0];
        // initial domination matrix full zeros
        let mut matrix = Array2::<i8>::zeros((n_population, n_population));
        // build dominance
        for i in 0..n_population {
            for j in 0..n_population {
                matrix[[i, j]] = Dominator::who_dominates(
                    population_objectives.row(i).view(),
                    population_objectives.row(j).view(),
                );
                matrix[[j, i]] = -matrix[[i, j]]
            }
        }
        matrix
    }
}
