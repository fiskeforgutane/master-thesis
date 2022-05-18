use std::time::Instant;

use log::trace;

use rand::prelude::*;

use crate::{
    ga::{Fitness, Mutation},
    problem::{NodeType, Problem},
    solution::routing::{Plan, RoutingSolution},
    utils::{GetPairMut, SwapNodes},
};

#[derive(Debug, Clone, Copy)]
pub enum TwoOptMode {
    /// Performs a 2-opt local search on every voyage for every vessel. The the u64 is the time limit for the local search per voyage. The f64 is the epsilon for accepting new solutions.
    LocalSerach(u64, f64),
    /// Performs a random 2-opt mutation in a random vessel's route
    IntraRandom,
}

#[derive(Debug, Clone, Copy)]
pub struct TwoOpt {
    mode: TwoOptMode,
}

impl TwoOpt {
    pub fn new(mode: TwoOptMode) -> TwoOpt {
        TwoOpt { mode }
    }

    /// Performs a two-opt swap in the given route for the given visit indices
    ///
    /// ## Arguments
    ///
    /// * `plan` - The current plan in scope
    /// * `v1` - The index of the first visit to swap
    /// * `v2` - The index of the second visit to swap
    pub fn update(plan: &mut Plan, v1: usize, v2: usize) {
        let plan = &mut plan.mutate();

        // reverse the plan from v1+1 to v2
        let v1 = v1 + 1;

        // if the visit indices are equal, we do not do anything
        if v1 == v2 {
            return;
        }

        // switch the order of nodes visited in the inclusive range [v1..v2]
        for i in v1..v2 {
            let k = v2 - (i - v1);

            // break when we are at the midpoint
            if k <= i {
                break;
            }

            // get the visits
            let (visit1, visit2) = plan.get_pair_mut(i, k);

            // perform the swap
            visit1.swap_nodes(visit2);
        }
    }

    /// Evaluates the 2-opt swap and returns the change in travel distance. A decreased distance yields a negative output.
    ///
    /// The change in distance can be evaluated in constant time by only comparing the edges that will be swapped
    ///
    /// ## Arguments
    ///
    /// * `plan` - The plan that a change should be evaluated
    /// * `v1` - The index of the first visit
    /// * `v2` - The index of the second visit, cannot be a visit to a production node
    pub fn evaluate(plan: &Plan, v1: usize, v2: usize, problem: &Problem) -> f64 {
        // node indices corresponding to visit v1 and v2
        let (n1, n2) = (plan[v1].node, plan[v2].node);

        // assert that n2 is not a production visit
        assert!(matches!(
            problem.nodes()[n2].r#type(),
            NodeType::Consumption
        ));

        // node indices corresponding to the next visit from v1 and v2
        let (n1next, n2next) = (plan[v1 + 1].node, plan[v2 + 1].node);
        // current distance
        let current_dist = problem.distance(n1, n1next) + problem.distance(n2, n2next);
        // new distance if the 2opt operation were to be performed
        let new_dist = problem.distance(n1, n2) + problem.distance(n1next, n2next);
        // return the relative change in distance
        new_dist - current_dist
    }

    /// Performs a 2-opt local search on the given voyage
    ///
    /// ## Arguments
    ///
    /// * `plan` - The plan in scope
    /// * `start` - The index of the production node at the beginning of the voyage in scope
    /// * `end` - The index of the production node at the end of the voyage in scope
    /// * `problem` - The underlying problem
    /// * `improvement_threshold` - The relative improvement threshold in solutoion quality to consider a new solution as "better"
    /// * `iterations_without_improvement` - The number of consecutive iterations without improvements below threshold that is required before i breaks.
    ///     Note that if no improving solutions are found in one iteration, it breaks anyway.
    pub fn local_search(
        plan: &mut Plan,
        start: usize,
        end: usize,
        problem: &Problem,
        time_limit: u64,
        epsilon: f64,
    ) {
        // check that the voyage consists of at least four visits, including start and end
        if end - start < 3 {
            return;
        }
        // count of number of iterations with improvemen less than threshold
        let mut count = 0;
        let mut aggregated_improvement = 0.0;

        let now = Instant::now();

        // keep track of wheter an improving solution was found
        let mut found_improving = true;

        while now.elapsed().as_secs() < time_limit && found_improving {
            count += 1;

            // bool to say if we found an improving solution above threshold
            found_improving = false;

            for swap_first in start..(end - 2) {
                //trace!("here");
                for swap_last in (swap_first + 2)..end {
                    let change = Self::evaluate(plan, swap_first, swap_last, problem);

                    if change < epsilon {
                        found_improving = true;
                        aggregated_improvement += f64::abs(change);

                        // move to next solution
                        Self::update(plan, swap_first, swap_last);
                    }
                }
            }
        }
        trace!("Ran local search for {} iterations ({} seconds) from start: {} to end: {}, and reduced the total travel distance by {}", count, now.elapsed().as_secs(), start, end, aggregated_improvement);
    }

    /// Returns the indicies of the production visits in the given plan, and the last visit, regardless of type
    pub fn production_visits(plan: &Plan, problem: &Problem) -> Vec<usize> {
        let mut indices = (0..plan.len())
            .filter(|i| {
                let visit = plan[*i];
                let kind = problem.nodes()[visit.node].r#type();
                match kind {
                    crate::problem::NodeType::Consumption => false,
                    crate::problem::NodeType::Production => true,
                }
            })
            .collect::<Vec<_>>();
        // add the last visit regardless of type, if not included already
        let last = plan.iter().last();
        if let Some(last) = last {
            match problem.nodes()[last.node].r#type() {
                NodeType::Consumption => indices.push(plan.len() - 1),
                _ => (),
            }
        }
        indices
    }
}

impl Mutation for TwoOpt {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        match self.mode {
            TwoOptMode::LocalSerach(time_limit, epsilon) => {
                let mutator = &mut solution.mutate();
                for plan in mutator.iter_mut() {
                    for interval in Self::production_visits(plan, problem).windows(2) {
                        let (start, end) = (interval[0], interval[1]);
                        Self::local_search(plan, start, end, problem, time_limit, epsilon)
                    }
                }
            }
            TwoOptMode::IntraRandom => {
                let mut rand = rand::thread_rng();
                // get random plan where a swap should be performed
                let v = rand.gen_range(0..problem.vessels().len());
                let mut mutator = solution.mutate();
                let plan = &mut mutator[v];

                // check that there are at least four visits in the plan, including start and end
                if plan.len() < 4 {
                    return;
                }
                // select two random visits to swap
                let v1 = rand.gen_range(0..plan.len() - 2);
                let v2 = rand.gen_range((v1 + 2)..plan.len());

                Self::update(plan, v1, v2);
            }
        }
    }
}
