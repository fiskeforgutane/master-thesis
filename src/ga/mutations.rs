use grb::attr;

use log::warn;
use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::Problem,
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

pub enum RedCostMode {
    /// Performs only one iteration where it updates the upper bounds of a random subset of visits
    /// that look promising to expand
    Mutate,
    /// Several iterations where it iteratively seeks to improve the soution by expanding visits
    LocalSerach,
}

/// This mutation exploits the dual solution of the quantities LP to direct the search towards a hopefulle better solution.
pub struct RedCost {
    mode: RedCostMode,
    max_visits: usize,
}

impl RedCost {
    /// Returns a RedCost with mode set to mutation
    pub fn red_cost_mutation(max_visits: usize) -> Self {
        let mode = RedCostMode::Mutate;

        RedCost { mode, max_visits }
    }
    /// Returns a RedCost with mode set to local search
    pub fn red_cost_local_search(max_visits: usize) -> Self {
        let mode = RedCostMode::LocalSerach;

        RedCost { mode, max_visits }
    }

    /// Returns an iterator with all the x-variable indices that can have the upper bound increased
    pub fn mutable_indices<'a>(
        v: usize,
        solution: &'a RoutingSolution,
    ) -> impl Iterator<Item = (usize, usize, usize)> + 'a {
        let problem = solution.problem();

        solution[v].windows(2).map(move |visits| {
            let (curr, next) = (visits[0], visits[1]);
            let (_, t2) = (curr.time, next.time);

            // vessel must leave at the beginning of this time period, i.e. this time period can be opened for laoding/unloading if next is pushed
            let must_leave =
                t2.max(t2 - problem.travel_time(curr.node, next.node, &problem.vessels()[v]));
            (must_leave, curr.node, v)
        })
    }

    /// Returns the visit indices for the given vessel that should be mutated
    fn get_visit_indices(
        n_visits: usize,
        vessel: usize,
        problem: &Problem,
        solution: &RoutingSolution,
    ) -> Vec<usize> {
        let quant_lp = solution.quantities();
        let vars = solution.variables();
        let model = &quant_lp.model;

        // the visits indeccorresponding to the ones with high reduced cost
        let mut visit_indices: Vec<usize> = (0..n_visits).collect();
        // the reduced costs
        let mut reduced_costs = vec![f64::NEG_INFINITY; n_visits];

        for (visit_idx, (t, n, v)) in Self::mutable_indices(vessel, solution).enumerate() {
            // sum the reduced cost over all products
            let reduced = (0..problem.products())
                .map(|p| model.get_obj_attr(attr::RC, &vars.x[t][n][v][p]).unwrap())
                .sum::<f64>();

            // get the index of the lowest reduced cost found so far
            let index = reduced_costs
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|x| x.0)
                .unwrap();

            // if the new reduced cost is larger than the lowest found so far that has been kept, keep the new instead
            if reduced_costs[index] < reduced {
                reduced_costs[index] = reduced;
                visit_indices[index] = visit_idx;
            }
        }
        visit_indices
    }

    pub fn iterate(
        max_visits: usize,
        rand: &mut ThreadRng,
        problem: &Problem,
        solution: &mut RoutingSolution,
    ) {
        // select random vessel to search for a index where the visit can be extended
        let v = rand.gen_range(0..problem.vessels().len());
        // number of visits to alter
        let n_visits = max_visits.min(rand.gen_range(0..solution[v].len()));

        // indices of visits to alter
        let visit_indices = Self::get_visit_indices(n_visits, v, problem, solution);

        // get a mutator
        let mutator = &mut solution.mutate();
        let mut plan = mutator[v].mutate();

        for i in visit_indices {
            let visit = &mut plan[i];
            // move visit one back or one forward with a 50/50 probability
            if rand.gen::<f64>() < 0.5 {
                // move back, if possible
                visit.time = 0.max(visit.time - 1);
            } else {
                // move forward
                visit.time = (problem.timesteps() - 1).min(visit.time + 1);
            }
        }
    }
}

impl Mutation for RedCost {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        let rand = &mut rand::thread_rng();
        match self.mode {
            RedCostMode::Mutate => Self::iterate(self.max_visits, rand, problem, solution),
            RedCostMode::LocalSerach => todo!(),
        }
    }
}
