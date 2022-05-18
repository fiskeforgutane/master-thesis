use log::trace;
use rand::prelude::*;

use crate::{
    ga::{Mutation, Fitness},
    problem::{Node, Problem, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

#[derive(Debug, Clone)]
pub struct AddRandom {
    rng: rand::rngs::StdRng,
}

impl AddRandom {
    pub fn new() -> AddRandom {
        AddRandom {
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}

impl Mutation for AddRandom {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        trace!("Applying AddRandom to {:?}", solution);
        // The number of timesteps
        let t = problem.timesteps();
        // Note: there always be at least one vessel in a `Problem`, and
        // 0..=x is always non-empty when x is an unsigned type
        let v = problem.indices::<Vessel>().choose(&mut self.rng).unwrap();
        let node = problem.indices::<Node>().choose(&mut self.rng).unwrap();

        // This is the valid range of times we can insert
        let available = problem.vessels()[v].available_from();
        let valid = available + 1..t;
        let time = valid.choose(&mut self.rng).unwrap();
        let plan = &solution[v];

        // We try to find the point in time that is closest to `time` and insert the visit there
        // We could restrict `0..t`, but it shouldn't really matter,
        // since the *only* time we will go past is if there is exactly one visit at every time step.
        for delta in 0..t {
            let up = (time + delta).min(t - 1);
            let down = time.max(delta + available + 1) - delta;

            for t in [up, down] {
                if plan.binary_search_by_key(&t, |v| v.time).is_err() {
                    let mut solution = solution.mutate();
                    let mut plan = solution[v].mutate();
                    plan.push(Visit { node, time: t });
                    return;
                }
            }
        }
    }
}
