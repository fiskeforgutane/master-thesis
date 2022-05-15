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

#[derive(Debug, Clone)]
pub struct Relocate;

impl Mutation for Relocate {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying Relocate to {:?}", solution);
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

        // find most expensive edge
        let most_expensive = once(None)
            .chain(plan.iter().map(|v| Some(v)))
            .chain(once(None))
            .tuple_windows()
            .enumerate()
            //.map(|(v1, v2, v3)| (v1, v2, v3))
            .max_by_key(|(_, (v1, v2, v3))| cost(v1, v2, v3));

        let (i, visit) = match most_expensive {
            Some((i, (_, v2, _))) => (i + 1, v2.unwrap()),
            None => return,
        };

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
                    .map(|(i, (v1, v2))| (plan_idx, i + 1, cost(&v1, &Some(visit), &v2)))
                    .min_by_key(|(_, _, c)| *c)
            })
            .min_by_key(|(_, _, c)| *c);

        let (plan_idx, idx) = match a {
            Some((plan_idx, i, _)) => (plan_idx, i),
            None => return,
        };

        let mut mutator = solution.mutate();

        // remove most the most expensive node
        let visit = mutator[v].mutate().swap_remove(i);

        // get the plan of the vessel where the removed vist will be inserted
        let mut plan = mutator[plan_idx].mutate();

        // if there is only one time period between prev and next, push next one later and its next and so on until there is room
        for i in (idx - 1)..(plan.len() - 1) {
            let visit1 = plan[i];
            let visit2 = &mut plan[i + 1];

            if visit1.time != visit2.time {
                break;
            }
            visit2.time += 1;
        }

        // create the visit to insert
        let to_insert = Visit {
            node: visit.node,
            time: plan[idx - 1].time + 1,
        };
        plan.insert(idx, to_insert);

        // call bounce to hopefully fix timing
        let mut bounce = Bounce::new(25, BounceMode::All);
        drop(plan);
        drop(mutator);
        bounce.apply(problem, solution);
    }
}
