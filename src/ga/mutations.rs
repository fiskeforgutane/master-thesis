use log::warn;
use rand::prelude::*;

use crate::{ga::Mutation, problem::Problem, solution::Visit};

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
    pub fn those_hips<R: rand::Rng>(mut rng: &mut R, problem: &Problem, plan: &mut [Visit]) {
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
