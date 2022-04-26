use float_ord::FloatOrd;

use log::trace;
use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::Problem,
    solution::routing::{PlanMut, RoutingSolution},
};

/// Takes the node associated with the highest cost in a random route and reinserts it at the best
/// position in the same route.
pub struct BestMove {
    rand: ThreadRng,
}

impl Mutation for BestMove {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying BestMove to {:?}", solution);
        // Select a random vessel
        let vessel = self.rand.gen_range(0..solution.len());

        let mut mutator = solution.mutate();
        let plan = &mut mutator[vessel].mutate();
        let plan_len = plan.len();

        // Finds the index in the route of the most expensive node
        let key1 = |x: &usize| FloatOrd(self.decreased_distance(*x, plan, problem));
        let v1 = (0..(plan_len - 1)).max_by_key(key1).unwrap();

        // Finds the cheapest position to insert the most expensive node
        let key2 = |x: &usize| FloatOrd(self.increased_distance(v1, *x, problem, plan));
        let v2 = (0..(plan_len - 1)).min_by_key(key2).unwrap();

        // The new visit time for the selected node
        let new_time = plan[v2].time;

        // Move all visits between the new and the old position in time
        for node_index in v2..v1 {
            if v2 > v1 {
                plan[node_index].time = plan[node_index + 1].time;
            } else {
                plan[node_index].time = plan[node_index - 1].time;
            }
        }

        // Set the correct time for the selected node
        plan[v1].time = new_time;
    }
}

impl BestMove {
    pub fn new() -> Self {
        BestMove {
            rand: rand::thread_rng(),
        }
    }
    /// Calculates the distance removed from the plan if a visit is removed
    fn decreased_distance(
        &self,
        visit: usize,
        vessel_plan: &mut PlanMut,
        problem: &Problem,
    ) -> f64 {
        let prev = vessel_plan[visit - 1].node;
        let cur = vessel_plan[visit].node;
        let next = vessel_plan[visit + 1].node;

        problem.distance(prev, cur) + problem.distance(cur, next) - problem.distance(prev, next)
    }

    /// Calculates the increased distance by inserting a node at a particular position
    fn increased_distance(
        &self,
        node_index: usize,
        position: usize,
        problem: &Problem,
        vessel_plan: &mut PlanMut,
    ) -> f64 {
        let prev = vessel_plan[position - 1].node;
        let cur = vessel_plan[node_index].node;
        let next = vessel_plan[position].node;

        problem.distance(prev, cur) + problem.distance(cur, next) - problem.distance(prev, next)
    }
}
