use itertools::Itertools;
use log::warn;
use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::{Problem, VesselIndex},
    solution::{routing::RoutingSolution, Visit},
};

pub fn choose_proportional_by_key<'a, I, T, F, R>(it: I, f: F, mut rng: R) -> T
where
    I: IntoIterator<Item = T> + 'a,
    for<'b> &'b I: IntoIterator<Item = &'b T>,
    T: 'a,
    F: for<'c> Fn(&'c T) -> f64,
    R: Rng,
{
    let total: f64 = (&it).into_iter().map(&f).sum();
    let threshold = rng.gen_range(0.0..=total);

    let mut sum = 0.0;
    for x in it.into_iter() {
        sum += f(&x);
        if sum >= threshold {
            return x;
        }
    }

    unreachable!()
}

// How we're going to perform the twerking.
pub enum TwerkMode {
    Random,
    All,
}
/// "Shake" up the time at which each vessel's visits are performed.
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
    pub fn those_hips<R: rand::Rng>(rng: &mut R, problem: &Problem, plan: &mut [Visit]) {
        // Note: assumes that the visits are sorted in ascending order by time, which is normally enforced by the mutation guard.
        // However, if this is called after some other mutation that breaks that guarantee we might have to fix it here
        let total_time = plan
            .windows(2)
            .map(|w| w[1].time - w[0].time)
            .sum::<usize>();

        // The "average time" between two visits.
        let avg = total_time / plan.len().max(1);
        // We don't want the visits to cross too often, so we'll try to keep it such that they sheldom cross
        let max_delta = (avg / 3) as isize;
        // The highest allowed timestep
        let t_max = (problem.timesteps() - 1) as isize;

        for visit in plan {
            let delta = rng.gen_range(-max_delta..=max_delta);
            let new = visit.time as isize + delta;
            visit.time = new.clamp(0, t_max) as usize;
        }
    }
}

impl Mutation for Twerk {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        let rng = &mut self.rng;

        let mut plans = solution.mutate();

        match self.mode {
            TwerkMode::Random => match plans.choose_mut(rng) {
                Some(plan) => Twerk::those_hips(rng, problem, &mut plan.mutate()),
                None => warn!("unable to twerk"),
            },
            TwerkMode::All => {
                for plan in plans.iter_mut() {
                    Twerk::those_hips(rng, problem, &mut plan.mutate())
                }
            }
        }
    }
}

/// How to apply the Bounce
pub enum BounceMode {
    All,
    Random,
}

/// A mutation that cause visits that are impossible to due to travel time to "bounce off" each other
/// in such a way that they (hopefully) get enought space to become travel time feasible.
pub struct Bounce {
    rng: rand::rngs::StdRng,
    passes: usize,
    mode: BounceMode,
}

impl Bounce {
    pub fn new(passes: usize, mode: BounceMode) -> Bounce {
        Bounce {
            rng: rand::rngs::StdRng::from_entropy(),
            passes,
            mode,
        }
    }

    pub fn apply_bounce_pass(
        problem: &Problem,
        vessel: VesselIndex,
        solution: &mut RoutingSolution,
    ) {
        let first = problem.origin_visit(vessel);
        let last = solution.artificial_end(vessel);
        let boat = &problem.vessels()[vessel];
        let max_t = problem.timesteps() - 1;

        let mut mutator = solution.mutate();
        let mut plan = mutator[vessel].mutate();

        for i in 0..plan.len() {
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

            match w1.cmp(&w2) {
                // If there is more warp to the future than the past, we'll move `current` one timestep towards
                // the start if possible (this is effectively time = max(time - 1, 0) for usize)
                std::cmp::Ordering::Less => current.time = current.time - 1.min(current.time),
                // If they are equal we will not move any of them
                std::cmp::Ordering::Equal => (),
                // If there is more warp towards the past than the future we'll try to push the `current` visit
                // one step forward.
                std::cmp::Ordering::Greater => current.time = (current.time + 1).min(max_t),
            }
        }
    }
}

impl Mutation for Bounce {
    fn apply(
        &mut self,
        problem: &Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
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
