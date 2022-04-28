use float_ord::FloatOrd;
use itertools::Itertools;
use log::warn;

use crate::{
    ga::Mutation,
    models::quantity::{sparse, QuantityLp},
    problem::{NodeIndex, Problem, ProductIndex, TimeIndex, VesselIndex},
    solution::{routing::RoutingSolution, Visit},
};

/// Mutation that exploits the output for the quantity model to eliminate shortages. This is done in two steps.
/// 1. Identify the node experiencing the single largest shortage.
/// 2. Try to eliminate this shortage by adding a visit to this node prior to the shortage. To choose the vessel to perform the visit we select among those not visiting the node in the relevant
/// time interval, and pick the one that is closest to the node.
pub struct AddSmart;

impl AddSmart {
    /// Returns the node experiencing the most significant violation throughout the time period, as well as the time period prior to when this violation started
    ///
    /// # Arguments
    /// * `solution` - The routing solution that should be investigated
    pub fn most_significant_violation(solution: &RoutingSolution) -> (NodeIndex, TimeIndex) {
        let lp = solution.quantities();
        let lp: &QuantityLp = todo!("must be changed after introduction of sparse QuantityLp");
        let w = &lp.vars.w;

        let get = |var: &grb::Var| -> f64 {
            lp.model
                .get_obj_attr(grb::attr::X, var)
                .expect("failed to retrieve variable value")
        };

        let key = |&(t, n, _, p): &(TimeIndex, NodeIndex, VesselIndex, ProductIndex)| {
            FloatOrd(get(&w[t][n][p]).abs())
        };

        let max_violation_key = QuantityLp::active(solution).max_by_key(key).unwrap();

        let (time, node, _, product) = max_violation_key.into();

        // backtrack to find the time period in which the violation started
        let violation_start = (0..time)
            .rev()
            .find_or_last(|t| f64::abs(get(&w[*t][node][product])) <= 1e-5)
            .unwrap();

        (node, violation_start)
    }

    /// Retrieves the vessel that is closest to the given node at the given time.
    /// The returned vessel does not visit the node either in its next or prior visit.
    /// The vessel must also be availble
    /// If all vessels visit this node in the previous or next visit, it returns None
    ///
    /// # Arguments
    /// * `problem` - The underlying problem
    /// * `solution` - The routing solution to investigate
    /// * `node` - The index of the node to visit
    /// * `time` - The time period in which the node should be visited
    pub fn find_closest_vessel(
        problem: &Problem,
        solution: &RoutingSolution,
        node: NodeIndex,
        time: TimeIndex,
    ) -> Option<VesselIndex> {
        let mut vessels = Vec::new();
        for v in 0..problem.vessels().len() {
            // if the vessel it not available, continue
            if time <= problem.vessels()[v].available_from() {
                continue;
            }

            let plan = &solution[v];
            // find the visit in the plan closest in time
            let closest_in_time = plan
                .iter()
                .enumerate()
                .min_by_key(|x| isize::abs(x.1.time as isize - time as isize))
                .unwrap();

            // check that neither the previous, current, nor the next visit is at the given node
            let range = (0.max(closest_in_time.0 as isize - 1) as usize)
                ..(plan.len() - 1).min(closest_in_time.0 + 1);
            let mut flag = false;
            for visit_idx in range {
                if plan[visit_idx].node == node {
                    flag = true
                }
            }
            if flag {
                continue;
            }

            // add the vessel and the distance to the node at the time
            let dist = problem.distance(closest_in_time.1.node, node);
            vessels.push((v, dist));
        }

        let closest = vessels.iter().min_by_key(|x| FloatOrd(x.1));
        closest.map(|(v, _)| *v)
    }
}

impl Mutation for AddSmart {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        let (node, time) = Self::most_significant_violation(solution);
        let closest_vessel = Self::find_closest_vessel(problem, solution, node, time);

        match closest_vessel {
            Some(v) => {
                let mutator = &mut solution.mutate();
                // get the plan of the vessel
                let plan = &mut mutator[v].mutate();
                plan.push(Visit { node, time });
                // fix the plan
                plan.fix();
            }
            None => {
                warn!("AddSmart found no closest vessel");
                return;
            }
        }
    }
}
