use itertools::iproduct;
use rand::{
    prelude::{IteratorRandom, StdRng},
    SeedableRng,
};

use crate::{
    ga::{initialization::GreedyWithBlinks, Mutation},
    problem::{self, Node},
    solution::Visit,
};

/// A ruin and recreate method that removes a single vessel's path, and inserts in in a greedily using greedy insertion with blinks
pub struct Vessel {
    rng: StdRng,
    blink_rate: f64,
    epsilon: (f64, f64),
}

impl Vessel {
    pub fn new(blink_rate: f64) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            blink_rate,
            epsilon: (1.0, 1.0),
        }
    }
}

impl Mutation for Vessel {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        // Choose a random vessel index.
        let vessel = (0..problem.vessels().len()).choose(&mut self.rng).unwrap();
        // Ruin it by removing all visits
        let mut mutator = solution.mutate();
        mutator[vessel].mutate().clear();
        drop(mutator);

        // Reconstruct using greedy with blinks
        let available = problem.vessels()[vessel].available_from();
        let candidates = iproduct!(
            problem.indices::<Node>(),
            available + 1..problem.timesteps()
        )
        .map(|(node, time)| (vessel, Visit { node, time }))
        .collect();

        // Gredily construct a new vessel plan based
        let greedy = GreedyWithBlinks::new(self.blink_rate);
        greedy.converge(solution, self.epsilon, candidates);
    }
}
