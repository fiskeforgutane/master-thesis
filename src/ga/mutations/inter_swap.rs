use log::trace;

use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::Problem,
    solution::routing::RoutingSolution,
    utils::{GetPairMut, SwapNodes},
};

// swaps one random visit from one route with a visit from another route
#[derive(Debug, Clone)]
pub struct InterSwap;

impl Mutation for InterSwap {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying InterSwap to {:?}", solution);
        let mut rand = rand::thread_rng();
        // select two random vessels participate in the swap
        let vessel1 = rand.gen_range(0..solution.len());
        let vessel2 = rand.gen_range(0..solution.len());

        if vessel1 == vessel2 {
            return;
        }
        // if any of the vessels do not have a visit other than origin, return
        if solution[vessel1].len() <= 1 || solution[vessel2].len() <= 1 {
            return;
        }

        // select a random visit from each vessel - excluding the origin
        let v1 = rand.gen_range(1..solution[vessel1].len());
        let v2 = rand.gen_range(1..solution[vessel2].len());

        let mutator = &mut solution.mutate();

        // perform the swap
        let (p1, p2) = &mut mutator.get_pair_mut(vessel1, vessel2);
        let visit1 = &mut p1.mutate()[v1];
        let visit2 = &mut p2.mutate()[v2];

        visit1.swap_nodes(visit2);
    }
}
