use grb::{attr, Status};

use itertools::Itertools;
use log::trace;

use rand::prelude::*;

use crate::{
    ga::Mutation,
    problem::Problem,
    solution::routing::{Plan, RoutingSolution},
};

#[derive(Debug, Clone)]
pub enum RedCostMode {
    /// Performs only one iteration where it updates the upper bounds of a random subset of visits
    /// that look promising to expand
    Mutate,
    /// Several iterations where it iteratively seeks to improve the soution by expanding visits
    LocalSerach(usize),
}

/// This mutation exploits the dual solution of the quantities LP to direct the search towards a hopefulle better solution.
#[derive(Debug, Clone)]
pub struct RedCost {
    pub mode: RedCostMode,
    pub max_visits: usize,
}

#[derive(Debug, Clone)]
pub enum MoveDirection {
    /// move visit to a later time period
    Forward,
    /// move to an earlier time period
    Back,
}

impl RedCost {
    /// Returns a RedCost with mode set to mutation
    pub fn red_cost_mutation(max_visits: usize) -> Self {
        let mode = RedCostMode::Mutate;

        RedCost { mode, max_visits }
    }
    /// Returns a RedCost with mode set to local search
    pub fn red_cost_local_search(max_visits: usize, iterations: usize) -> Self {
        let mode = RedCostMode::LocalSerach(iterations);

        RedCost { mode, max_visits }
    }

    /// Returns an iterator with all the x-variable indices that can have the upper bound increased
    pub fn mutable_indices<'a>(
        v: usize,
        solution: &'a RoutingSolution,
    ) -> impl Iterator<Item = (usize, MoveDirection, usize, usize, usize)> + 'a {
        let problem = solution.problem();

        solution[v].iter().enumerate().tuple_windows().flat_map(
            move |((curr_idx, curr), (next_idx, next))| {
                let (t1, t2) = (curr.time as isize, next.time as isize);

                // check that t2 is acutally at least 1. If not, it should be ensured that the second visit must happen no before time period 1
                assert!(t2 >= 1);
                // time period before arriving at next
                let before_next = t2 - 1;

                // time period when the vesse must leave current visit
                let must_leave = t1.max(
                    t2 - (problem.travel_time(curr.node, next.node, &problem.vessels()[v])
                        as isize),
                );

                // double check that before next and must leave are positive
                assert!(before_next >= 0);
                assert!(must_leave >= 0);

                // vessel must leave at the beginning of this time period, i.e. this time period can be opened for loading/unloading if next is pushed

                [
                    (
                        curr_idx,
                        MoveDirection::Forward,
                        must_leave as usize,
                        curr.node,
                        v,
                    ),
                    (
                        next_idx,
                        MoveDirection::Back,
                        before_next as usize,
                        next.node,
                        v,
                    ),
                ]
                .into_iter()
            },
        )
    }

    /// Returns the visit indices for the given vessel that should be mutated
    /// Starts at the origin so an index = 0 implpies the origin.
    fn get_visit_indices(
        n_visits: usize,
        vessel: usize,
        problem: &Problem,
        solution: &RoutingSolution,
    ) -> Vec<(usize, MoveDirection)> {
        let quant_lp = &mut solution.quantities();

        let model = &quant_lp.model;
        let vars = &quant_lp.vars;

        // the visits indices ccorresponding to the ones with high reduced cost
        let mut visit_indices: Vec<(usize, MoveDirection)> =
            (0..n_visits).map(|i| (i, MoveDirection::Forward)).collect();
        // the reduced costs
        let mut reduced_costs = vec![f64::NEG_INFINITY; n_visits];
        trace!("init reduced costs: {:?}", reduced_costs);

        for (visit_idx, direction, t, n, v) in Self::mutable_indices(vessel, solution) {
            trace!(
                "Trying to retrieve the reduced cost for x_{}_{}_{}",
                t,
                n,
                v
            );

            trace!(
                "reduced cost of x_{}_{}_{}_0: {:?}",
                t,
                n,
                v,
                model.get_obj_attr(attr::RC, &vars.x[t][n][v][0])
            );

            // assert that the value of the x-variable is 0
            let value = model.get_obj_attr(attr::X, &vars.x[t][n][v][0]);
            trace!(
                "variable x_{}_{}_{}_0, and its current value: {:?}",
                t,
                n,
                v,
                value
            );

            // sum the reduced cost over all products
            let reduced = (0..problem.products())
                .map(|p| {
                    f64::abs(
                        model
                            .get_obj_attr(attr::RC, &vars.x[t][n][v][p])
                            .expect("Failed to retrieve reduced cost"),
                    )
                })
                .sum::<f64>();

            trace!("reduced costs: {:?}", reduced_costs);
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
                visit_indices[index] = (visit_idx, direction);
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

        // If the chosen vessel has no visits return, only origin is considered no visits
        if solution[v].len() <= 1 {
            return;
        }
        // number of visits to alter
        let n_visits = max_visits.min(rand.gen_range(1..solution[v].len()));

        // indices of visits to alter, including origin which is index 0
        let visit_indices = Self::get_visit_indices(n_visits, v, problem, solution);

        trace!(
            "visit indices for vessel plan {} of lenght {}: {:?}",
            v,
            solution[v].len(),
            visit_indices
        );

        // the the plan as mut
        let plan = &mut solution.mutate()[v];

        for (i, direction) in visit_indices {
            match direction {
                MoveDirection::Forward => Self::move_forward(i + 1, plan, problem),
                MoveDirection::Back => Self::move_back(i, v, plan, problem),
            }
        }
    }

    /// Moves the visit of the given index in the given plan one time period later, if possible, otherwise, nothing happens.
    fn move_forward(visit_index: usize, plan: &mut Plan, problem: &Problem) {
        trace!(
            "in move_forward, plan has lenght: {}, and visit index is: {}",
            plan.len(),
            visit_index
        );
        trace!(
            "plan: {:?}",
            plan.iter().map(|v| (v.node, v.time)).collect::<Vec<_>>()
        );

        // check that the visit after, if any, this one does not happen in the same time period as this one plus 1
        if let Some(next_visit) = plan.get(visit_index + 1) {
            let visit = plan.get(visit_index).unwrap();
            trace!(
                "time of this visit: {}, time of next visit: {}",
                visit.time,
                next_visit.time
            );
            if next_visit.time == visit.time + 1 {
                // not possible to move the visit later
                return;
            }
        }

        let mut_plan = &mut plan.mutate();

        let visit = mut_plan.get_mut(visit_index).unwrap();

        visit.time = (problem.timesteps() - 1).min(visit.time + 1);

        trace!(
            "finished move forward, plan is now: {:?}",
            mut_plan
                .iter()
                .map(|v| (v.node, v.time))
                .collect::<Vec<_>>()
        )
    }
    /// Moves the visit of the given index in the given plan one time period earlier, if possible, otherwise, nothing happens.
    fn move_back(visit_index: usize, vessel: usize, plan: &mut Plan, problem: &Problem) {
        trace!(
            "in move_back, plan has lenght: {}, and visit index is: {}",
            plan.len(),
            visit_index
        );
        trace!(
            "plan: {:?}",
            plan.iter().map(|v| (v.node, v.time)).collect::<Vec<_>>()
        );
        trace!(
            "previous time: {}, current time: {}",
            plan[visit_index - 1].time,
            plan[visit_index].time - 1
        );
        // check that the visit before this one does not happen in the preious time period
        // should be safe to index the previous, as origin should never be called move back on.
        if plan[visit_index - 1].time == plan[visit_index].time - 1 {
            return;
        }

        let mut_plan = &mut plan.mutate();

        let visit = mut_plan.get_mut(visit_index).unwrap();
        assert!(visit.time >= 1);

        visit.time = problem.vessels()[vessel]
            .available_from()
            .max(visit.time - 1);

        trace!(
            "finished move back, plan is now: {:?}",
            mut_plan
                .iter()
                .map(|v| (v.node, v.time))
                .collect::<Vec<_>>()
        )
    }
}

impl Mutation for RedCost {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        // check that status is optimal and do nothing if semi-cont has been enabled
        let status = solution
            .quantities()
            .model
            .status()
            .expect("Could not retrieve the model status");
        let is_mip = solution
            .quantities()
            .model
            .get_attr(attr::IsMIP)
            .expect("Could not retrieve the IsMIP attribuite");

        assert!(matches!(status, Status::Optimal));
        trace!("type of model: {:?}", is_mip);
        // if the model is not an LP, we have used semi-cont which doesn't have a well defined dual so we return
        if is_mip == 1 {
            return;
        }

        trace!("Applying RedCost({:?}) to {:?}", self.mode, solution);
        let rand = &mut rand::thread_rng();
        match self.mode {
            RedCostMode::Mutate => Self::iterate(self.max_visits, rand, problem, solution),
            RedCostMode::LocalSerach(iters) => {
                for _ in 0..iters {
                    Self::iterate(self.max_visits, rand, problem, solution);
                }
            }
        }
        trace!("FINISHED RED COST MUTATION")
    }
}
