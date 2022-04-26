use itertools::iproduct;
use rand::{
    prelude::{IteratorRandom, StdRng},
    SeedableRng,
};

use crate::{
    ga::{initialization::GreedyWithBlinks, Mutation},
    problem::{Node, Vessel},
    solution::{routing::Plan, Visit},
};

use super::Dropout;

/// A ruin and recreate method that removes visits within a random time period, and repairs the solution by greedily inserting visits using greedy insertion with blinks
pub struct Period {
    rng: StdRng,
    blink_rate: f64,
    removal_rate: f64,
    epsilon: (f64, f64),
    max_size: usize,
}

impl Period {
    pub fn new(blink_rate: f64, removal_rate: f64, max_size: usize) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            blink_rate,
            removal_rate,
            epsilon: (1.0, 1.0),
            max_size,
        }
    }
}

impl Mutation for Period {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        let t = problem.timesteps();
        let half = self.max_size / 2;
        let middle = (0..t).choose(&mut self.rng).unwrap();
        let period = middle.max(half) - half..(middle + half).min(t);

        // Ruin the solution
        {
            let mut solution = solution.mutate();
            for mut plan in solution.iter_mut().map(Plan::mutate) {
                plan.dropout(|i, v| i != 0 && period.contains(&v.time), self.removal_rate);
            }
        }

        // Recreate it again
        let candidates = iproduct!(
            problem.indices::<Vessel>(),
            problem.indices::<Node>(),
            period
        )
        .filter(|&(v, n, t)| t > problem.vessels()[v].available_from())
        .map(|(v, node, time)| (v, Visit { node, time }))
        .collect();

        GreedyWithBlinks::new(self.blink_rate).converge(solution, self.epsilon, candidates)
    }
}
