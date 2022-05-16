use std::collections::HashSet;

use log::{trace, warn};

use rand::prelude::*;

use crate::{
    ga::{Fitness, Mutation},
    problem::Problem,
    solution::Visit,
};

// How we're going to perform the twerking.
#[derive(Debug, Clone, Copy)]
pub enum TwerkMode {
    Random,
    All,
}
/// "Shake" up the time at which each vessel's visits are performed.
#[derive(Debug, Clone)]
pub struct Twerk {
    rng: rand::rngs::StdRng,
    mode: TwerkMode,
}

impl Twerk {
    pub fn some_random_person() -> Twerk {
        Twerk {
            rng: rand::rngs::StdRng::from_entropy(),
            mode: TwerkMode::Random,
        }
    }

    pub fn everybody() -> Twerk {
        Twerk {
            rng: rand::rngs::StdRng::from_entropy(),
            mode: TwerkMode::All,
        }
    }
}

impl Twerk {
    pub fn those_hips<R: rand::Rng>(
        rng: &mut R,
        vessel: usize,
        problem: &Problem,
        plan: &mut [Visit],
    ) {
        // Note: assumes that the visits are sorted in ascending order by time, which is normally enforced by the mutation guard.
        // However, if this is called after some other mutation that breaks that guarantee we might have to fix it here
        let total_time = plan
            .windows(2)
            .map(|w| w[1].time - w[0].time)
            .sum::<usize>();

        // When the vessel is first available
        let available = problem.vessels()[vessel].available_from();
        // The "average time" between two visits.
        let avg = total_time / plan.len().max(1);
        // We don't want the visits to cross too often, so we'll try to keep it such that they sheldom cross
        let max_delta = (avg / 3) as isize;
        // The highest allowed timestep
        let t_max = (problem.timesteps() - 1) as isize;
        // The times that are "in use"
        let mut times = plan.iter().map(|v| v.time).collect::<HashSet<_>>();

        for visit in plan {
            let delta = rng.gen_range(-max_delta..=max_delta);
            let new = (visit.time as isize + delta).clamp(available as isize + 1, t_max) as usize;

            if !times.contains(&new) {
                times.remove(&visit.time);
                times.insert(new);
                visit.time = new;
            }
        }
    }
}

impl Mutation for Twerk {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
        _: &dyn Fitness,
    ) {
        trace!("Applying Twerk({:?}) to {:?}", self.mode, solution);
        let rng = &mut self.rng;

        let mut mutator = solution.mutate();

        match self.mode {
            TwerkMode::Random => match (0..mutator.len()).choose(rng) {
                Some(vessel) => {
                    Twerk::those_hips(rng, vessel, problem, &mut mutator[vessel].mutate()[1..])
                }
                None => warn!("unable to twerk"),
            },
            TwerkMode::All => {
                for (vessel, plan) in mutator.iter_mut().enumerate() {
                    Twerk::those_hips(rng, vessel, problem, &mut plan.mutate()[1..])
                }
            }
        }
    }
}
