use crate::unwrap_operator;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

unwrap_operator!(
    unwrap_mutation_operator,
    crate::operators::MutationOperator,
    "Unsupported or unknown mutation operator object",
    [
        crate::operators::py_operators::PyBitFlipMutation,
        crate::operators::py_operators::PySwapMutation,
        crate::operators::py_operators::PyGaussianMutation,
        crate::operators::py_operators::PyDisplacementMutation,
        crate::operators::py_operators::PyScrambleMutation,
    ]
);

unwrap_operator!(
    unwrap_crossover_operator,
    crate::operators::CrossoverOperator,
    "Unsupported or unknown crossover operator object",
    [
        crate::operators::py_operators::PySinglePointBinaryCrossover,
        crate::operators::py_operators::PyUniformBinaryCrossover,
        crate::operators::py_operators::PyOrderCrossover,
        crate::operators::py_operators::PyExponentialCrossover,
        crate::operators::py_operators::PySimulatedBinaryCrossover,
    ]
);

unwrap_operator!(
    unwrap_sampling_operator,
    crate::operators::SamplingOperator,
    "Unsupported or unknown sampling operator",
    [
        crate::operators::py_operators::PyRandomSamplingFloat,
        crate::operators::py_operators::PyRandomSamplingInt,
        crate::operators::py_operators::PyRandomSamplingBinary,
        crate::operators::py_operators::PyPermutationSampling
    ]
);

unwrap_operator!(
    unwrap_duplicates_cleaner,
    crate::helpers::duplicates::PopulationCleaner,
    "Unsupported or unknown duplicates cleaner",
    [
        crate::helpers::duplicates::PyCloseDuplicatesCleaner,
        crate::helpers::duplicates::PyExactDuplicatesCleaner
    ]
);
