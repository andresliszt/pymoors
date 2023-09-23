use rand::{Rng, RngCore};
use std::fmt::Debug;

use crate::genetic::Individual;
use crate::operators::{DuelResult, GeneticOperator, SelectionOperator};

#[derive(Clone, Debug)]
pub struct RandomSelection {}

impl RandomSelection {
    pub fn new() -> Self {
        Self {}
    }
}

impl GeneticOperator for RandomSelection {
    fn name(&self) -> String {
        "RandomSelection".to_string()
    }
}

impl SelectionOperator for RandomSelection {
    fn tournament_duel(
        &self,
        p1: &Individual,
        p2: &Individual,
        rng: &mut dyn RngCore,
    ) -> DuelResult {
        let p1_feasible = p1.is_feasible();
        let p2_feasible = p2.is_feasible();

        // If exactly one is feasible, that one automatically wins:
        if p1_feasible && !p2_feasible {
            DuelResult::LeftWins
        } else if p2_feasible && !p1_feasible {
            DuelResult::RightWins
        } else {
            // Otherwise, both are feasible or both are infeasible => random winner.
            if rng.gen_bool(0.5) {
                DuelResult::LeftWins
            } else {
                DuelResult::RightWins
            }
        }
    }
}
