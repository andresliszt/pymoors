use numpy::ndarray::Array1;
use pyo3::prelude::*;
use rand::{Rng, RngCore};

use crate::genetic::Genes;
use crate::operators::{CrossoverOperator, GeneticOperator};

#[derive(Clone, Debug)]
pub struct OrderCrossover;

impl OrderCrossover {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for OrderCrossover {
    fn name(&self) -> String {
        "OrderCrossover".to_string()
    }
}

impl CrossoverOperator for OrderCrossover {
    fn crossover(
        &self,
        parent_a: &Genes,
        parent_b: &Genes,
        rng: &mut dyn RngCore,
    ) -> (Genes, Genes) {
        let len = parent_a.len();
        assert_eq!(len, parent_b.len());

        // Choose 2 cut points
        let mut p1 = rng.gen_range(0..len);
        let mut p2 = rng.gen_range(0..len);
        if p1 > p2 {
            std::mem::swap(&mut p1, &mut p2);
        }

        // Initialize children with some default
        let mut child_a = Array1::from_elem(len, f64::NAN);
        let mut child_b = Array1::from_elem(len, f64::NAN);

        // copy [p1..p2] from A-> childA, B-> childB
        for i in p1..p2 {
            child_a[i] = parent_a[i];
            child_b[i] = parent_b[i];
        }

        // Fill remainder of child_a from parent_b in order
        {
            let mut fill_index = p2 % len; // start filling after p2
            for i in 0..len {
                // index in parent_b
                let idx_b = (p2 + i) % len;
                let val_b = parent_b[idx_b];
                // skip if already in child_a
                if !child_a.iter().any(|&x| x == val_b) {
                    child_a[fill_index] = val_b;
                    fill_index = (fill_index + 1) % len;
                }
            }
        }

        // Fill remainder of child_b from parent_a in order
        {
            let mut fill_index = p2 % len;
            for i in 0..len {
                let idx_a = (p2 + i) % len;
                let val_a = parent_a[idx_a];
                if !child_b.iter().any(|&x| x == val_a) {
                    child_b[fill_index] = val_a;
                    fill_index = (fill_index + 1) % len;
                }
            }
        }

        (child_a, child_b)
    }
}

/// Crossover operator for permutation-based individuals using Order Crossover (OX)
#[pyclass(name = "OrderCrossover")]
#[derive(Clone)]
pub struct PyOrderCrossover {
    pub inner: OrderCrossover,
}

#[pymethods]
impl PyOrderCrossover {
    #[new]
    fn new() -> Self {
        Self {
            inner: OrderCrossover::new(),
        }
    }
}
