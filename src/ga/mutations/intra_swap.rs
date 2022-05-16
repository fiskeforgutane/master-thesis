use log::trace;

use rand::prelude::*;

use crate::{
    ga::{Mutation, Fitness},
    problem::Problem,
    solution::routing::RoutingSolution,
    utils::{GetPairMut, SwapNodes},
};

#[derive(Debug, Clone)]
pub struct IntraSwap;

impl Mutation for IntraSwap {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        trace!("Applying IntraSwap to {:?}", solution);
        let mut rand = rand::thread_rng();
        // get random plan where a swap should be performed
        let v = rand.gen_range(0..problem.vessels().len());
        let mut mutator = solution.mutate();
        let plan = &mut mutator[v].mutate();

        // if the plan does not contain any visits other than origin, return
        if plan.len() <= 1 {
            return;
        }

        // select two random visits to swap - exclude the origin
        let v1 = rand.gen_range(1..plan.len());
        let v2 = rand.gen_range(1..plan.len());

        // if v1 and v2 are equal, we don't do anything
        if v1 == v2 {
            return;
        }

        // get the visits
        let (v1, v2) = plan.get_pair_mut(v1, v2);
        // perform the swap
        v1.swap_nodes(v2);
    }
}
