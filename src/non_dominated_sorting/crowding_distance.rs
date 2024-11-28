use std::f64::INFINITY;

use numpy::ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

fn argsort(arr: ArrayView1<f64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..arr.len()).collect();
    indices.sort_unstable_by(move |&i, &j| arr[i].partial_cmp(&arr[j]).unwrap());
    indices
}

// TODO: Consider the case where front has 3 elements or less
// TODO: What happens if f_max - f_min = 0

pub fn crodwing_distance(front: ArrayView2<f64>) -> Array1<f64> {
    let number_individuals = front.shape()[0];
    let number_objectives = front.shape()[1];
    // Container for the distance
    let mut distance: Array1<f64> = Array1::zeros(number_individuals);
    // Update corner values to inf
    distance[0] = INFINITY;
    distance[number_individuals - 1] = INFINITY;
    // Iterate over objective space
    for objective_index in 0..number_objectives {
        let objective = front.column(objective_index);
        // Sort the objective values and get the indices
        let sorted_indices = argsort(objective);
        // get max and min of the current ofjective
        let objective_max = objective[sorted_indices[0]];
        let objective_min = objective[sorted_indices[sorted_indices.len() - 1]];
        // Iterate over individuals using sorted_index
        for individual_index in 1..(number_individuals - 1) {
            distance[individual_index] += (front
                [[individual_index + 1, sorted_indices[objective_index]]]
                + front[[individual_index - 1, sorted_indices[objective_index]]])
                / (objective_max - objective_min)
        }
    }
    distance
}

/// Python wrapper for crodwing distnace
#[pyfunction]
#[pyo3(name = "crowding_distance")]
pub fn crodwing_distance_py<'py>(
    py: Python<'py>,
    front: PyReadonlyArray2<f64>,
) -> &'py PyArray1<f64> {
    crodwing_distance(front.as_array()).into_pyarray(py)
}
