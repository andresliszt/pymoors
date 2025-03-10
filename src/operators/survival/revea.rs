use std::borrow::Cow;

use ndarray::{s, Array1, Array2};
use ndarray_stats::QuantileExt;

use crate::genetic::{Fronts, Population, PopulationFitness};
use crate::helpers::extreme_points::get_ideal_from_fronts;
use crate::operators::survival::helpers::HyperPlaneNormalization;
use crate::operators::{GeneticOperator, SurvivalOperator};
use crate::algorithms::AlgorithmContext;
use crate::random::RandomGenerator;

/// Implementation of the survival operator for the NSGA3 algorithm presented in the paper
/// An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sorting Approach

#[derive(Clone, Debug)]
pub struct ReveaReferencePointsSurvival {
    reference_points: Array2<f64>,
}

impl GeneticOperator for ReveaReferencePointsSurvival {
    fn name(&self) -> String {
        "ReveaReferencePointsSurvival".to_string()
    }
}

impl ReveaReferencePointsSurvival {
    pub fn new(reference_points: Array2<f64>) -> Self {
        Self { reference_points }
    }
}

impl SurvivalOperator for ReveaReferencePointsSurvival {
    fn set_survival_score(
        &self,
        _fronts: &mut crate::genetic::Fronts,
        _rng: &mut dyn RandomGenerator,
        algorithm_context: &AlgorithmContext
    ) {
        unimplemented!("REVEA doesn't use survival score. It uses random tournament which doesn't depend on the score")
    }

    // fn operate(
    //         &self,
    //         fronts: &mut Fronts,
    //         n_survive: usize,
    //         rng: &mut dyn RandomGenerator,
    //         algorithm_context: &AlgorithmContext,
    //     ) -> Population {

    //         // this is the global ideal point computed over all fronts
    //         let z_min = get_ideal_from_fronts(&fronts);
    //         fronts[0]
    // }


}
