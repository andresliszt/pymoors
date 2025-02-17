# Pymoors macros

This repository contains macros necessary to avoid code duplication in the pymoors project. They aim to register operators and algorithms defined in Rust in Python using pyo3. An example of how to register is as follows:

```rust
use crate::operators::{GenesMut, GeneticOperator, MutationOperator};
use crate::random::RandomGenerator;

use pymoors_macros::py_operator;

#[py_operator("mutation")]
#[derive(Clone, Debug)]
/// Mutation operator that flips bits in a binary individual with a specified mutation rate
pub struct BitFlipMutation {
    pub gene_mutation_rate: f64,
}

impl BitFlipMutation {
    pub fn new(gene_mutation_rate: f64) -> Self {
        Self { gene_mutation_rate }
    }
}

impl GeneticOperator for BitFlipMutation {
    fn name(&self) -> String {
        "BitFlipMutation".to_string()
    }
}

impl MutationOperator for BitFlipMutation {
    fn mutate<'a>(&self, mut individual: GenesMut<'a>, rng: &mut dyn RandomGenerator) {
        for gene in individual.iter_mut() {
            if rng.gen_bool(self.gene_mutation_rate) {
                *gene = if *gene == 0.0 { 1.0 } else { 0.0 };
            }
        }
    }
}
```

When using `#[py_operator("mutation")]`, the implementation of the `BitFlipMutation` struct in Python is automatically created, getting the docstring from the rust stuct doc. Then, in Python, you can simply call:

```python
from pymoors import BitFlipMutation

import numpy as np

population = np.array([[1.0,0.0,0.0], [0.0,1.0,1.0]])

mutation = BitFlipMutation(gene_mutation_rate=0.99, seed = 1)
>>> mutation.mutate(population)
array([[0., 1., 1.],
       [1., 0., 0.]])
>>> mutation.__doc__
"Mutation operator that flips bits in a binary individual with a specified mutation rate"
```
