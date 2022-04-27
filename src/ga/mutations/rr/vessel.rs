use itertools::iproduct;
use rand::{
    prelude::{IteratorRandom, StdRng},
    Rng, SeedableRng,
};

use crate::{
    ga::{initialization::GreedyWithBlinks, Mutation},
    problem::Node,
    solution::Visit,
};

pub trait Dropout<T> {
    fn dropout<F: Fn(T) -> bool>(&mut self, eligible: F, removal_rate: f64);
}

impl<T> Dropout<T> for Vec<T>
where
    T: Copy,
{
    fn dropout<F: Fn(T) -> bool>(&mut self, eligible: F, removal_rate: f64) {
        let mut i = 0;
        while i < self.len() {
            if eligible(self[i]) && rand::thread_rng().gen_bool(removal_rate) {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// A ruin and recreate method that removes a single vessel's path, and inserts in in a greedily using greedy insertion with blinks
pub struct Vessel {
    rng: StdRng,
    blink_rate: f64,
    removal_rate: f64,
    epsilon: (f64, f64),
}

impl Vessel {
    pub fn new(blink_rate: f64, removal_rate: f64) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            blink_rate,
            removal_rate,
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
        // Ruin it by removing a part of the vessel's plan
        {
            let mut mutator = solution.mutate();
            let mut plan = mutator[vessel].mutate();
            let origin = plan.origin();
            plan.dropout(|x| x != origin, self.removal_rate);
        }

        // Find the insertion candidates
        let available = problem.vessels()[vessel].available_from();
        let nodes = problem.indices::<Node>();
        let time = available + 1..problem.timesteps();
        let candidates = iproduct!(nodes, time)
            .map(|(node, time)| (vessel, Visit { node, time }))
            .collect();

        // Gredily construct a new vessel plan based on greedy insertion with blinks
        let greedy = GreedyWithBlinks::new(self.blink_rate);
        greedy.converge(solution, self.epsilon, candidates);
    }
}
