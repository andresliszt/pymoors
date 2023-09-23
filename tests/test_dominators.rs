use pymoors::non_dominated_sorting::dominator::Dominator;

use numpy::ndarray::{Array1, Array2};
use rstest::*;

#[rstest]
#[case(Array1::<f64>::from(vec![1.0,2.0]), Array1::<f64>::from(vec![1.0,2.0]), 0)]
#[case(Array1::<f64>::from(vec![1.0,2.0]), Array1::<f64>::from(vec![2.0,1.0]), 0)]
#[case(Array1::<f64>::from(vec![1.0,1.0]), Array1::<f64>::from(vec![2.0,2.0]), 1)]
#[case(Array1::<f64>::from(vec![1.0,2.0]), Array1::<f64>::from(vec![-1.0,-2.0]), -1)]
fn test_who_dominates(#[case] f1: Array1<f64>, #[case] f2: Array1<f64>, #[case] expected: i8) {
    assert_eq!(Dominator::who_dominates(f1.view(), f2.view()), expected)
}

#[rstest]
#[case(Array2::from(vec![[1.0, 2.0],[0.0, 0.0]]), Array2::from(vec![[0, -1], [1, 0]]))]
#[case(Array2::from(vec![[1.0, 2.0, 3.0],[2.0, 3.0, 4.0]]), Array2::from(vec![[0, 1], [-1, 0]]))]
#[case(Array2::from(vec![[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]]), Array2::from(vec![[0, 0], [0, 0]]))]
fn test_domination_matrix(#[case] population_objectives: Array2<f64>, #[case] expected: Array2<i8>) {
    assert_eq!(
        Dominator::domination_matrix(population_objectives.view()),
        expected
    )
}
