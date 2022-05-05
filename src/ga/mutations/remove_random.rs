use log::trace;
use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::{Problem, Vessel},
    solution::routing::RoutingSolution,
};

#[derive(Debug, Clone)]
pub struct RemoveRandom {
    rng: rand::rngs::StdRng,
}

impl RemoveRandom {
    pub fn new() -> Self {
        Self {
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}

impl Mutation for RemoveRandom {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying RemoveRandom to {:?}", solution);
        // Note: there always be at least one vessel in a `Problem`, and
        // 0..=x is always non-empty when x is an unsigned type
        let v = problem.indices::<Vessel>().choose(&mut self.rng).unwrap();

        match (1..solution[v].len()).choose(&mut self.rng) {
            Some(x) => {
                solution.mutate()[v].mutate().remove(x);
            }
            None => (),
        }
    }
}
