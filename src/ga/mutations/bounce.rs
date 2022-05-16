use log::trace;
use pyo3::pyclass;
use rand::prelude::*;

use crate::{
    ga::{Fitness, Mutation},
    problem::{Problem, VesselIndex},
    solution::routing::RoutingSolution,
};

/// How to apply the Bounce
#[pyclass]
#[derive(Debug, Clone)]
pub enum BounceMode {
    All,
    Random,
}

/// A mutation that cause visits that are impossible to due to travel time to "bounce off" each other
/// in such a way that they (hopefully) get enought space to become travel time feasible.
pub struct Bounce {
    rng: rand::rngs::StdRng,
    pub passes: usize,
    mode: BounceMode,
}

impl Bounce {
    pub fn apply_bounce_pass(
        problem: &Problem,
        vessel: VesselIndex,
        solution: &mut RoutingSolution,
    ) {
        let first = solution[vessel][0];
        let last = solution.artificial_end(vessel);
        let boat = &problem.vessels()[vessel];
        let max_t = problem.timesteps() - 1;

        let mut mutator = solution.mutate();
        let mut plan = mutator[vessel].mutate();

        for i in 1..plan.len() {
            // Note: `one.len() == i` and `two.len() == plan.len() - i`, by construction
            let (one, two) = plan.split_at_mut(i);
            let prev = one.last().unwrap_or(&first);
            // `i < plan.len()` => `two.len() > 0`
            let (two, three) = two.split_at_mut(1);
            let current = &mut two[0];
            // Note: `last` if None iff plan.len() == 0, in which case we wouldn't be in this loop
            let next = three.first().unwrap_or(last.as_ref().unwrap());

            // The time required to travel from `prev` to `current`, and from `current` to `next`
            let t1 = problem.travel_time(prev.node, current.node, boat);
            let t2 = problem.travel_time(current.node, next.node, boat);
            // The amount of warp we have in each direction.
            // Note: this is really just max(t1 - (dest.time - src.time), 0) formulated to avoid underflow of usize.
            let w1 = t1 - (current.time - prev.time).min(t1);
            let w2 = t2 - (next.time - current.time).min(t2);

            match (
                current.time - prev.time,
                next.time - current.time,
                w1.cmp(&w2),
            ) {
                // If there is more warp to the future than the past, we'll move `current` one timestep towards
                // the start if possible (this is effectively time = max(time - 1, 0) for usize)
                (2.., _, std::cmp::Ordering::Less) => {
                    current.time = current.time - 1.min(current.time)
                }
                // If there is more warp towards the past than the future we'll try to push the `current` visit
                // one step forward.
                (_, 2.., std::cmp::Ordering::Greater) => {
                    current.time = (current.time + 1).min(max_t)
                }

                // If they are equal we will not move any of them
                // Same goes if moving one of them would violate the "no simultaneous visits" requirement.
                (_, _, _) => (),
            }
        }
    }

    pub fn new(passes: usize, mode: BounceMode) -> Bounce {
        Bounce {
            rng: rand::rngs::StdRng::from_entropy(),
            passes,
            mode,
        }
    }
}

impl Mutation for Bounce {
    fn apply(
        &mut self,
        problem: &Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
        _: &dyn Fitness,
    ) {
        trace!("Applying Bounce({:?}) to {:?}", self.mode, solution);
        for _ in 0..self.passes {
            match self.mode {
                BounceMode::All => {
                    for (v, _) in problem.vessels().iter().enumerate() {
                        Self::apply_bounce_pass(problem, v, solution);
                    }
                }
                BounceMode::Random => {
                    let vessel = problem
                        .vessels()
                        .choose(&mut self.rng)
                        .expect("there must be vessels")
                        .index();
                    Self::apply_bounce_pass(problem, vessel, solution)
                }
            }
        }
    }
}
