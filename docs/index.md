---
hide:
  - navigation
  - toc
---

## Why moo-rs?

<div class="moo-features">
  <div class="moo-feature">
    <span class="moo-feature__icon">⚡</span>
    <h3>Rust performance</h3>
    <p>Core algorithms implemented in pure Rust — no Python bottlenecks, no GIL, no overhead.</p>
  </div>
  <div class="moo-feature">
    <span class="moo-feature__icon">🐍</span>
    <h3>Python ergonomics</h3>
    <p>Full-featured Python bindings via <a href="https://github.com/PyO3/pyo3">pyo3</a>. NumPy arrays in, NumPy arrays out.</p>
  </div>
  <div class="moo-feature">
    <span class="moo-feature__icon">🔌</span>
    <h3>Pluggable operators</h3>
    <p>Swap crossover, mutation, selection and survival operators freely — or bring your own.</p>
  </div>
  <div class="moo-feature">
    <span class="moo-feature__icon">🎯</span>
    <h3>Many-objective ready</h3>
    <p>From classic NSGA-II to reference-vector methods like NSGA-III and REVEA — all included.</p>
  </div>
</div>

---

## Available Multi-Objective Algorithms

A concise index of the currently available algorithms.

<div class="moo-algo-grid">
  <a class="moo-algo-card" href="user_guide/algorithms/nsga2.html">
    <span class="moo-algo-card__name">NSGA-II</span>
    <span class="moo-algo-card__desc">Baseline Pareto-based MOEA with fast non-dominated sorting and crowding distance. Robust, widely used for 2–3 objectives.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/nsga3.html">
    <span class="moo-algo-card__name">NSGA-III</span>
    <span class="moo-algo-card__desc">Many-objective extension of NSGA-II using reference points to maintain diversity and guide convergence.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/ibea.html">
    <span class="moo-algo-card__name">IBEA</span>
    <span class="moo-algo-card__desc">Indicator-Based EA that optimizes a quality indicator (e.g., hypervolume/ε-indicator) to drive selection.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/spea2.html">
    <span class="moo-algo-card__name">SPEA-II</span>
    <span class="moo-algo-card__desc">Strength Pareto EA with enhanced fitness assignment, density estimation (k-NN), and external archive.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/agemoea.html">
    <span class="moo-algo-card__name">AGE-MOEA</span>
    <span class="moo-algo-card__desc">Approximation-guided MOEA that directly improves the Pareto-front approximation via set-level indicators.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/rnsga2.html">
    <span class="moo-algo-card__name">RNSGA-II</span>
    <span class="moo-algo-card__desc">Reference-point oriented NSGA-II variant; biases the search toward regions of interest while preserving diversity.</span>
  </a>
  <a class="moo-algo-card" href="user_guide/algorithms/revea.html">
    <span class="moo-algo-card__name">REVEA</span>
    <span class="moo-algo-card__desc">Reference vector/region–guided evolutionary algorithm using directional vectors to balance diversity and convergence.</span>
  </a>
  <a class="moo-algo-card moo-algo-card--custom" href="user_guide/algorithms/custom/custom.html">
    <span class="moo-algo-card__name">Custom Defined Algorithms</span>
    <span class="moo-algo-card__desc">User defined algorithms by defining selection and survival operators.</span>
  </a>
</div>

---

## Introduction to Multi-Objective Optimization

**Multi-objective optimization** refers to a set of techniques and methods designed to solve problems where *multiple objectives* must be satisfied simultaneously. These objectives are often *conflicting*, meaning that improving one may deteriorate another. For instance, one might seek to **minimize** production costs while **maximizing** product quality at the same time.

### General Formulation

A multi-objective optimization problem can be formulated in a generic mathematical form. If we have \(k\) objective functions to optimize, it can be expressed as:

\[
\begin{aligned}
&\min_x \quad (f_1(x), f_2(x), \dots, f_k(x)) \\
&\text{subject to:} \\
&g_i(x) \leq 0, \quad i = 1, \dots, m \\
&h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
\]

Where:
- \( x \) represents the set of decision variables.
- \( f_i(x) \) are the objective functions.
- \( g_i(x) \leq 0 \) and \( h_j(x) = 0 \) represent the **constraints_fn** of the problem (e.g., resource limits, quality requirements, etc.).

Unlike single-objective optimization, here we seek to optimize *all* objectives simultaneously. However, in practice, there is no single "best" solution for *all* objectives. Instead, we look for a set of solutions known as the **Pareto front** or **Pareto set**.

## Advantages for Multi-Objective Optimization

1. **Natural Handling of Multiple Objectives**: By operating on a population of solutions, GAs can maintain an approximation to the **Pareto front** during execution.
2. **Flexibility**: They can be easily adapted to different kinds of problems (discrete, continuous, constrained, etc.).
3. **Robustness**: They tend to perform well in the presence of *noise* or uncertainty in the problem, offering acceptable performance under less-than-ideal conditions.

## Beauty and Misbehavior Optimization Problem

In this unique optimization problem, there is only one individual who optimizes both beauty and misbehavior at the same time: my little dog Arya!

<div style="text-align: center;">
  <img src="images/arya.png" alt="Arya" width="500" />
</div>

Arya not only captivates with her beauty, but she also misbehaves in the most adorable way possible. This problem serves as a reminder that sometimes the optimal solution is as heartwarming as it is delightfully mischievous.
