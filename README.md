# pymoors
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)
![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![Black](https://img.shields.io/badge/Code%20Style-Black-000000.svg)
[![codecov](https://codecov.io/gh/andresliszt/pymoors/graph/badge.svg)](https://codecov.io/gh/andresliszt/pymoors)

## Overview

This project aims to solve multi-objective optimization problems using genetic algorithms. It is a hybrid implementation combining Python and Rust, leveraging the power of [PyO3](https://github.com/PyO3/pyo3) to bridge the two languages. This project was inspired in the amazing python project [pymoo](https://github.com/anyoptimization/pymoo).

## Features

- **Multi-Objective Optimization**: Implementations of popular multi-objective genetic algorithms like NSGA-II and NSGA-III.
- **Hybrid Python and Rust**: Core computational components are written in Rust for performance, while the user interface and high-level API are provided in Python.
- **Extensible Operators**: Support for various sampling, mutation, crossover, and survival operators.
- **Constraint Handling**: Ability to handle constraints in optimization problems.


## Installation (NOT AVAILABLE YET)

To install the package you simply can do

```sh
pip install pymoors
```

## Solving a Small Knapsack Problem with NSGA-II

Here is an example of how to use the `pymoors` package to solve a small knapsack problem using the NSGA-II algorithm:

```python
import numpy as np
from pymoors import (
    Nsga2,
    BitFlipMutation,
    SinglePointBinaryCrossover,
    RandomSamplingBinary,
    ExactDuplicatesCleaner,
)

# Define the problem data
WEIGHTS = np.array([12, 2, 1, 4, 10], dtype=float)
VALUES = np.array([4, 2, 1, 5, 3], dtype=float)
CAPACITY = 15.0

def fitness_knapsack(population_genes):
    total_values = population_genes @ VALUES
    total_weights = population_genes @ WEIGHTS
    fitness_matrix = np.column_stack((-total_values, total_weights))
    return fitness_matrix

def constraints_knapsack(population_genes):
    total_weights = population_genes @ WEIGHTS
    return (total_weights - CAPACITY)[:, np.newaxis]

# Create the NSGA-II algorithm instance
algorithm = Nsga2(
    sampler=RandomSamplingBinary(),
    crossover=SinglePointBinaryCrossover(),
    mutation=BitFlipMutation(gene_mutation_rate=0.5),
    fitness_fn=fitness_knapsack,
    constraints_fn=constraints_knapsack,
    duplicates_cleaner=ExactDuplicatesCleaner(),
    n_vars=5,  # 5 items
    pop_size=100,  # population size
    n_offsprings=50,  # offsprings per generation
    num_iterations=100,  # generation count
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
)

# Run the algorithm
algorithm.run()

# Retrieve the final population of solutions
final_population = algorithm.population
# Retrieve solutions with rank = 0
best: list[Individual] = final_population.best

>>> best[0]
Individual(genes=array([0., 0., 0., 1., 0.]), fitness=array([-5.,  4.]), rank=0, constraints=array([-11.]))
```

## Constraints handling


In **pymoors**, constraints are provided as a **Python function** that operates at the **population level**—i.e., it takes a 2D NumPy array (`(N, d)` shape, where each row corresponds to one individual) and returns a 2D array (`(N, k)`) of constraint evaluations. Each individual is considered **feasible** if **all** its constraint values are **strictly less than 0**.


Below is a simplified example constraint function for a 2-gene problem, where $(x_1 + 2x_2 \leq 10)$. We return an array of shape `(N, 1)`:

```python
import numpy as np

from pymoors.typing import TwoDArray

def constraints_fn(population_genes: TwoDArray) -> TwoDArray:
    """
    Each row in population_genes corresponds to one individual.
    Suppose our constraint is x1 + 2*x2 <= 10 (two genes per individual).
    We'll return an array of shape (N,1), where values < 0 indicate feasibility.
    """
    lhs = population_genes[:, 0] + 2.0 * population_genes[:, 1]
    return (lhs - 10.0)[:, np.newaxis]
```

To handle approximate equality constraints, one common trick is to transform them into inequality constraints. For example, if you have an equality ($g(x) = 0$) and you want  tolerance, define: $g_{\text{ineq}}(x) = |g(x)| - \epsilon < 0.$

#### Example: Enforcing $( x_1 + 2x_2 = 5 )$ with ($\epsilon = 0.1$)

```python
import numpy as np

from pymoors.typing import TwoDArray

def equality_constraint_fn(population_genes: TwoDArray) -> TwoDArray:
    """
    Enforce the approximate equality constraint: x1 + 2*x2 ≈ 5
    Using the transformation: abs(h(x)) - epsilon < 0
    """
    h = population_genes[:, 0] + 2.0 * population_genes[:, 1] - 5.0  # h(x) = x1 + 2*x2 - 5
    epsilon = 0.1
    return (np.abs(h) - epsilon)[:, np.newaxis]  # Feasible if < 0
```

## Improvements

1. **Fast Non-Dominated Sorting**
   A bottleneck lies in the fast non-dominated sorting algorithm (defined in `src/non_dominated_sorting/dominator.rs`). Any improvement to this algorithm would significantly impact overall performance.

2. **Duplicate Elimination**
   Another bottleneck is the duplicate-removal step. In Python, SciPy offers a very performant `cdist` function to compute pairwise distances (e.g., Euclidean). The analogous step in this project still falls short compared to SciPy's performance.

3. **Population-Level Operators**
   Each operator trait (e.g., `CrossoverOperator`) currently acts on individual pairs of solutions, exemplified by:
   ```rust
   fn crossover(
       &self,
       parent_a: &Genes,
       parent_b: &Genes,
       rng: &mut dyn RngCore,
   ) -> (Genes, Genes);
    ```
4. **Reduce clone calls** Another opportunity is to reduce excessive `clone` calls wherever possible, as they can add overhead to the genetic operators and auxiliary functions.
5. **Better user interface** Improve user exposed population `Population` class to manipulate optimal solutions found.
6. **Better docstring on .pyi file** Improve and sync the _pymoors.pyi with the Rust code.
6. **Better printer** Currently the printer is only showing the min of each objective.

## TODO

1. Enable CI/CD.
2. Separate python implementation from Rust. This project can be a stand alone Rust crate .
3. Enable ability to pass genetic operator from python (crossover, mutation and sampling as python functions).
4. Coverage, dev tools, linters, formatters, makefile, etc.
5. Write a better Readme
