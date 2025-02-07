# Algorithms

In pymoors, the algorithms are implemented as classes that are exposed on the Python side with a set of useful attributes. These attributes include the final population and the optimal or best set of individuals found during the optimization process.

For example, after running an algorithm like NSGA2, you can access:
- **Final Population:** The complete set of individuals from the last generation.
- **Optimum Set:** Typically, the best individuals (e.g., those with rank 0) that form the current approximation of the Pareto front.

This design abstracts away the complexities of the underlying Rust implementation and provides an intuitive, Pythonic interface for setting up, executing, and analyzing multi-objective optimization problems.

## Mathematical Formulation of a Multi-Objective Optimization Problem with Constraints

Consider the following optimization problem:

\[
\begin{aligned}
\min_{x_1, x_2} \quad & f_1(x_1,x_2) = x_1^2 + x_2^2 \\
\min_{x_1, x_2} \quad & f_2(x_1,x_2) = (x_1-1)^2 + x_2^2 \\
\text{subject to} \quad & x_1 + x_2 \leq 1, \\
& x_1 \geq 0,\quad x_2 \geq 0.
\end{aligned}
\]

### Theoretical Solution

The feasible region is defined by the constraints:
- \( x_1 + x_2 \leq 1 \), which creates a triangular region in the first quadrant with vertices at \((0,0)\), \((1,0)\), and \((0,1)\).
- \( x_1 \geq 0 \) and \( x_2 \geq 0 \).

For the objectives:
- The first objective, \( f_1(x_1,x_2) = x_1^2 + x_2^2 \), is minimized at \((0,0)\).
- The second objective, \( f_2(x_1,x_2) = (x_1-1)^2 + x_2^2 \), is minimized at \((1,0)\).

Since these two minimizers lie at different vertices of the feasible region, there is an inherent trade-off between the objectives. The Pareto front is formed by the set of non-dominated solutions along the boundary of the feasible region.

A common approach to derive the Pareto front is to consider solutions along the line \( x_1 + x_2 = 1 \) (with \(x_1, x_2 \in [0,1]\)). By parameterizing the boundary as \( x_2 = 1 - x_1 \), we can express the objectives as functions of \( x_1 \):

\[
\begin{aligned}
f_1(x_1, 1-x_1) &= x_1^2 + (1-x_1)^2, \\
f_2(x_1, 1-x_1) &= (x_1-1)^2 + (1-x_1)^2.
\end{aligned}
\]

Thus, the Pareto front is given by:

\[
\left\{ \left(f_1(x_1, 1-x_1),\, f_2(x_1, 1-x_1)\right) \mid x_1 \in [0,1] \right\}.
\]

This continuous set of solutions along the boundary represents the trade-off between minimizing \( f_1 \) and \( f_2 \) within the given constraints.


Below is how you can formulate and solve this problem in pymoors:

```python
import numpy as np
from pymoors import (
    Nsga2,
    RandomSamplingFloat,
    GaussianMutation,
    ExponentialCrossover,
    CloseDuplicatesCleaner
)
from pymoors.typing import TwoDArray

# Define the fitness function
def fitness(genes: TwoDArray) -> TwoDArray:
    x1 = genes[:, 0]
    x2 = genes[:, 1]
    # Objective 1: f1(x1,x2) = x1^2 + x2^2
    f1 = x1**2 + x2**2
    # Objective 2: f2(x1,x2) = (x1-1)^2 + x2**2
    f2 = (x1 - 1)**2 + x2**2
    return np.column_stack([f1, f2])

# Define the constraints function
def constraints(genes: TwoDArray) -> TwoDArray:
    x1 = genes[:, 0]
    x2 = genes[:, 1]
    # Constraint 1: x1 + x2 <= 1
    g1 = x1 + x2 - 1
    # Convert to 2D array
    return g1.reshape(-1, 1)

# Set up the NSGA2 algorithm with the above definitions
algorithm = Nsga2(
    sampler=RandomSamplingFloat(min=0, max=1),
    crossover=ExponentialCrossover(exponential_crossover_rate=0.75),
    mutation=GaussianMutation(gene_mutation_rate=0.1, sigma=0.01),
    fitness_fn=fitness,
    constraints_fn=constraints,  # Pass the constraints function
    duplicates_cleaner=CloseDuplicatesCleaner(epsilon=1e-5),
    n_vars=2,
    pop_size=200,
    n_offsprings=200,
    num_iterations=200,
    mutation_rate=0.1,
    crossover_rate=0.9,
    keep_infeasible=False,
    lower_bound=0
)

# Run the algorithm
algorithm.run()

```
