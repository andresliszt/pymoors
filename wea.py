from pymoors import Nsga2, RandomSamplingBinary, UniformBinaryCrossover, BitFlipMutation

import numpy as np


def evaluate_population(population):
    """
    Calcula dos objetivos para una población de individuos binarios.

    Objetivo 1: OneMax - Número total de 1's en cada individuo.
    Objetivo 2: LeadingOnes - Número de 1's consecutivos al inicio de cada individuo.

    Parameters:
    -----------
    population : np.ndarray
        Array binario de forma (n_population, n_vars), donde cada fila representa un individuo.

    Returns:
    --------
    objectives : np.ndarray
        Array de forma (n_population, 2) donde cada fila contiene [OneMax, LeadingOnes] para el individuo correspondiente.
    """
    population = np.asarray(population)

    # Objetivo 1: OneMax
    one_max = population.sum(axis=1)

    # Objetivo 2: LeadingOnes
    zeros = population == 0
    first_zero_indices = np.argmax(zeros, axis=1)
    all_ones = np.all(population == 1, axis=1)
    leading_ones = first_zero_indices.copy()
    leading_ones[all_ones] = population.shape[1]

    objectives = np.vstack((one_max, leading_ones)).T
    return objectives


algorithm = Nsga2(
    sampler=RandomSamplingBinary(),
    crossover=UniformBinaryCrossover(),
    mutation=BitFlipMutation(gene_mutation_rate=0.3),
    fitness_fn=evaluate_population,
    pop_size=100,
    n_offsprings=100,
    num_iterations=100,
    mutation_rate=0.1,
    crossover_rate=0.9,
)

algorithm.run()

import pdb; pdb.set_trace()

