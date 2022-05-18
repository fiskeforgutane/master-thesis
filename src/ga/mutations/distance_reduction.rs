use float_ord::FloatOrd;

use log::trace;
use pyo3::pyclass;
use rand::prelude::*;

use crate::{
    ga::{Mutation, Fitness},
    problem::Problem,
    solution::routing::{PlanMut, RoutingSolution},
};

#[pyclass]
#[derive(Clone, Debug)]
pub enum DistanceReductionMode {
    All,
    Random,
}

/// A mutation operator that moves the node that leads to the maximum reduction in total travel distance for one vessel
pub struct DistanceReduction {
    mode: DistanceReductionMode,
}

impl DistanceReduction {
    pub fn new(mode: DistanceReductionMode) -> DistanceReduction {
        DistanceReduction { mode }
    }

    pub fn distance_reduction(
        &mut self,
        problem: &Problem,
        solution: &mut RoutingSolution,
        vessel_index: usize,
    ) {
        // Initialize values
        let mut mutator = solution.mutate();
        let plan = &mut mutator[vessel_index].mutate();
        let plan_len = plan.len();

        // Holders for the best move (from, to) and the largest reduction in distance
        let mut best_move: (usize, usize) = (0, 0);
        let mut largest_reduction: f64 = -1.0;

        // Have to check all node moves
        for from in 0..(plan_len - 1) {
            // For each (from, to)-combination we calculate the distance reduction
            let key = |to: &usize| FloatOrd(self.distance_reduction_calc(problem, plan, from, *to));
            let to = (0..(plan_len - 1))
                .filter(|v| *v != from)
                .max_by_key(key)
                .unwrap_or_else(|| from);

            // If the new distance reduction is higher than the previous max, update the move and the
            // largest reduction
            if self.distance_reduction_calc(problem, plan, from, to) > largest_reduction {
                best_move = (from, to);
                largest_reduction = self.distance_reduction_calc(problem, plan, from, to);
            }
        }

        let (start, end) = best_move;

        let new_time = plan[end].time;

        // Move all other visits accordingly to the best move
        if end > start {
            for node_index in (start..(end + 1)).rev() {
                plan[node_index].time = plan[node_index - 1].time;
            }
        } else {
            for node_index in end..(start + 1) {
                plan[node_index].time = plan[node_index + 1].time;
            }
        }

        plan[start].time = new_time;
    }

    fn distance_reduction_calc(
        &self,
        problem: &Problem,
        plan: &mut PlanMut,
        from: usize,
        to: usize,
    ) -> f64 {
        let old_1 = (plan[from].node, plan[from + 1].node);
        let old_2 = (plan[to].node, plan[to + 1].node);
        let new_1 = (plan[to].node, plan[from].node);
        let new_2 = (plan[from].node, plan[to + 1].node);

        if (new_1.0 == new_1.1) || (new_2.0 == new_2.1) {
            return -1.0;
        }
        problem.distance(old_1.0, old_1.1) + problem.distance(old_2.0, old_2.1)
            - problem.distance(new_1.0, new_1.1)
            - problem.distance(new_2.0, new_2.1)
    }
}

impl Mutation for DistanceReduction {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        trace!("DistanceReduction({:?}): {:?}", self.mode, solution);
        match self.mode {
            DistanceReductionMode::All => {
                for vessel_index in 0..solution.len() {
                    self.distance_reduction(problem, solution, vessel_index);
                }
            }
            DistanceReductionMode::Random => {
                let mut rand = rand::prelude::thread_rng();
                let vessel_index = rand.gen_range(0..solution.len());
                self.distance_reduction(problem, solution, vessel_index);
            }
        }
    }
}
