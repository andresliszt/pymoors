use std::collections::HashSet;
use std::iter::repeat;

use numpy::ndarray::{ArrayView1, ArrayView2, Axis, Zip};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

/// Handles Pareto dominated solutions
fn who_dominates(f1: ArrayView1<'_, f64>, f2: ArrayView1<'_, f64>) -> i8 {
    if Zip::from(&f1).and(&f2).all(|&f1, &f2| f1 < f2) {
        return 1;
    }
    if Zip::from(&f1).and(&f2).all(|&f1, &f2| f1 > f2) {
        return -1;
    }
    0
}

fn get_current_front(
    population_fitness: ArrayView2<'_, f64>,
    remainder_indexes: &Vec<usize>,
) -> Vec<usize> {
    // Get population size
    let population_size = population_fitness.shape()[0];
    // Create an empty vector for the current Pareto Front
    let mut current_front: Vec<usize> = Vec::new();
    // Create empty container for dominated individuals. This is a vector of vectors, where
    // is_dominated[i] is a vector (possibly empty) of all individuals who dominates individual i
    // If is_dominated[i] is an empty vector, then individual i belongs to the Pareto Front
    let mut is_dominated: Vec<Vec<usize>> =
        Vec::from_iter(repeat(Vec::new()).take(population_size));

    for i in 0..population_size {
        for j in (i + 1)..population_size {
            let relation = who_dominates(population_fitness.row(i), population_fitness.row(j));
            // relation = 1 means that individual i dominates individual j
            if relation == 1 {
                is_dominated[j].push(i)
            }
            // relation = -1 means that individual j dominates individual i
            else if relation == -1 {
                is_dominated[i].push(j)
            }
        }
        if is_dominated[i].len() == 0 {
            current_front.push(remainder_indexes[i]);
        }
    }

    current_front
}

fn update_remainder_individuals<'a, 'b>(
    remainder_individuals: &'a mut Vec<usize>,
    current_front: &'b Vec<usize>,
) -> &'a mut Vec<usize> {
    let current_front_set: HashSet<usize> = current_front.iter().cloned().collect();
    remainder_individuals.retain(|x| !current_front_set.contains(x));
    remainder_individuals
}

fn fast_non_dominated_sorting(population_fitness: ArrayView2<f64>) -> Vec<Vec<usize>> {
    // Get population size
    let population_size: usize = population_fitness.shape()[0];
    // Container for the fronts
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    //
    let mut initial_population_idx: Vec<usize> = Vec::from_iter(0..population_size);
    // Get the first front
    let current_front: Vec<usize> = get_current_front(population_fitness, &initial_population_idx);
    // Check for remainder individuals that haven't been assiged yet
    let remainder_individuals: &mut Vec<usize> =
        update_remainder_individuals(&mut initial_population_idx, &current_front);
    // push the first front
    fronts.push(Vec::from_iter(current_front));
    // if there are not more individuals finish
    if remainder_individuals.len() == 0 {
        return fronts;
    }
    // Iterate until all individuals are assigned
    loop {
        // filter out population objectives with individuals that haven't been assigned
        let current_front: Vec<usize> = get_current_front(
            population_fitness
                .select(Axis(0), &Vec::from_iter(remainder_individuals.clone()))
                .view(),
            &Vec::from_iter(remainder_individuals.clone()),
        );
        let remainder_individuals: &mut Vec<usize> =
            update_remainder_individuals(remainder_individuals, &current_front);
        // Push the front
        fronts.push(Vec::from_iter(current_front));
        // Once all individuals have been assigned we break
        if remainder_individuals.len() == 0 {
            break;
        }
    }
    fronts
}

/// Python wrapper for fast_non_dominated_sorting algorithm
#[pyfunction]
#[pyo3(name = "fast_non_dominated_sorting")]
pub fn fast_non_dominated_sorting_py(population_fitness: PyReadonlyArray2<f64>) -> Vec<Vec<usize>> {
    fast_non_dominated_sorting(population_fitness.as_array())
}
