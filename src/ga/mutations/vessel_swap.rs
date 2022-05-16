use log::trace;
use rand::{prelude::ThreadRng, Rng};

use crate::{
    ga::{Fitness, Mutation},
    problem::Problem,
    solution::routing::RoutingSolution,
};

pub struct VesselSwap {
    rand: ThreadRng,
}

impl VesselSwap {
    pub fn new() -> Self {
        Self {
            rand: rand::thread_rng(),
        }
    }
}

impl Mutation for VesselSwap {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        trace!("Applying VesselSwap to {:?}", solution);
        // Select two random vessels for swapping
        let vessel1 = self.rand.gen_range(0..solution.len());
        let mut vessel2 = self.rand.gen_range(0..solution.len());

        // Ensure that the two vessels are not the same
        while vessel1 == vessel2 {
            vessel2 = self.rand.gen_range(0..solution.len());
        }

        let mut mutator = solution.mutate();

        mutator.swap(vessel1, vessel2);
    }
}
