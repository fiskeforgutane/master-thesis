use std::iter::once;

use itertools::Itertools;
use log::trace;

use rand::prelude::*;

use crate::{
    ga::{
        mutations::{Bounce, BounceMode},
        Mutation,
    },
    problem::Problem,
    solution::{routing::RoutingSolution, Visit},
};

pub trait RemoveSeveral<T> {
    fn split(&mut self, idx: usize);
}

impl<T> RemoveSeveral<T> for Vec<T> {
    fn split(&mut self, idx: usize) {
        let mut i = idx;
        while i < self.len() {
            self.swap_remove(i);
            i += 1;
        }
    }
}

/* impl<T> Dropout<T> for Vec<T>
where
    T: Copy,
{
    fn dropout<F: Fn(T) -> bool>(&mut self, indices: &Vec<usize>) {
        let mut i = 0;
        while i < self.len() {
            if eligible(self[i]) && rand::thread_rng().gen_bool(removal_rate) {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
} */

#[derive(Debug, Clone)]
pub struct Split;

impl Mutation for Split {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying Split to {:?}", solution);
        let mut rand = rand::thread_rng();
        // get random plan to get a node from
        let v = rand.gen_range(0..problem.vessels().len());
        let plan = &solution[v];
        //let mut mutator = solution.mutate();
        //let plan = &mut mutator[v].mutate();

        // if the plan does not contain any visits other than origin, return
        if plan.len() <= 1 {
            return;
        }
        let vessel = &problem.vessels()[v];

        let cost = |v1: &Option<&Visit>, v2: &Option<&Visit>, v3: &Option<&Visit>| {
            let v2 = v2.unwrap();
            let v1 = v1.unwrap_or(v2);
            let v3 = v3.unwrap_or(v2);
            let a = problem.travel_time(v1.node, v2.node, vessel);
            let b = problem.travel_time(v2.node, v3.node, vessel);
            let c = problem.travel_time(v1.node, v3.node, vessel);
            a + b - c
        };

        let insert_cost = |v1: &Option<&Visit>, v2: &Option<&[Visit]>, v3: &Option<&Visit>| {
            let v2 = v2.unwrap();
            let v1 = v1.unwrap_or(&v2[0]);
            let v3 = v3.unwrap_or(&v2[v2.len() - 1]);
            let a = problem.travel_time(v1.node, v2[0].node, vessel);
            let b = problem.travel_time(v2[v2.len()-1].node, v3.node, vessel);
            let c = problem.travel_time(v1.node, v3.node, vessel);
            a + b - c
        };

        // find most expensive edge
        let most_expensive = once(None)
            .chain(plan.iter().map(|v| Some(v)))
            .chain(once(None))
            .tuple_windows()
            .enumerate()
            //.map(|(v1, v2, v3)| (v1, v2, v3))
            .max_by_key(|(_, (v1, v2, v3))| cost(v1, v2, v3));

        let (i, _) = match most_expensive {
            Some((i, (_, v2, _))) => (i, v2.unwrap()),
            None => return,
        };

        let mut visits =plan[i..].iter().cloned().collect::<Vec<_>>();

        
        
        {
            let mut mutator = solution.mutate();
            // remove most the most expensive nodes
            let mut plan = mutator[v].mutate();
            plan.split(i);
        }

        // find the vessel where it is cheapest to insert the node
        let a = solution
            .iter()
            .enumerate()
            .filter_map(|(plan_idx, p)| {
                p.iter()
                    .map(|v| Some(v))
                    .chain(once(None))
                    .tuple_windows()
                    .enumerate()
                    .map(|(i, (v1, v2))| (plan_idx, i + 1, insert_cost(&v1, &Some(visits.as_slice()), &v2)))
                    .min_by_key(|(_, _, c)| *c)
            })
            .min_by_key(|(_, _, c)| *c);

        let (plan_idx, idx) = match a {
            Some((plan_idx, i, _)) => (plan_idx, i),
            None => return,
        };

        
        let mut mutator = solution.mutate();
        // get the plan of the vessel where the removed vists will be inserted
        let mut plan = mutator[plan_idx].mutate();

        visits[0].time = usize::min(plan[idx - 1].time + 1, problem.timesteps() - 1);
        // modify the visits to insert
        for j in 1..visits.len() {
            let i = j - 1;
            let mut visit = visits[j];
            visit.time = visits[i].time + 1;
        }

        // insert the visits
        let mut i = idx;
        for v in visits {
            plan.insert(i, v);
            i += 1;
        }
        // #help - use fix
        plan.fix();

        // call bounce to hopefully fix timing
        let mut bounce = Bounce::new(25, BounceMode::All);
        drop(plan);
        drop(mutator);
        bounce.apply(problem, solution);
    }
}
