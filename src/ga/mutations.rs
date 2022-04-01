use std::{
    cmp::{max, min},
    fmt::Error,
    ops::Deref,
    time::Instant,
};

use grb::{attr, Status};

use float_ord::FloatOrd;
use itertools::Itertools;
use log::{trace, warn};
use pyo3::pyclass;
use rand::prelude::*;

use crate::{
    ga::Mutation,
    models::quantity_cont::QuantityLpCont,
    problem::{Node, NodeType, Problem, Timestep, Vessel, VesselIndex},
    solution::{
        routing::{Plan, PlanMut, RoutingSolution},
        Visit,
    },
    utils::GetPairMut,
};

pub fn choose_proportional_by_key<'a, I, T, F, R>(it: I, f: F, mut rng: R) -> T
where
    I: IntoIterator<Item = T> + 'a,
    for<'b> &'b I: IntoIterator<Item = &'b T>,
    T: 'a,
    F: for<'c> Fn(&'c T) -> f64,
    R: Rng,
{
    let total: f64 = (&it).into_iter().map(&f).sum();
    let threshold = rng.gen_range(0.0..=total);

    let mut sum = 0.0;
    for x in it.into_iter() {
        sum += f(&x);
        if sum >= threshold {
            return x;
        }
    }

    unreachable!()
}

pub struct AddRandom {
    rng: rand::rngs::StdRng,
}

impl AddRandom {
    pub fn new() -> AddRandom {
        AddRandom {
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}

impl Mutation for AddRandom {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying AddRandom to {:?}", solution);
        // Note: there always be at least one vessel in a `Problem`, and
        // 0..=x is always non-empty when x is an unsigned type
        let v = problem.indices::<Vessel>().choose(&mut self.rng).unwrap();
        let node = problem.indices::<Node>().choose(&mut self.rng).unwrap();
        let time = problem.indices::<Timestep>().choose(&mut self.rng).unwrap();

        let mut solution = solution.mutate();
        let mut plan = solution[v].mutate();
        plan.push(Visit { node, time });
    }
}

pub struct RemoveRandom {
    rng: rand::rngs::StdRng,
}

impl RemoveRandom {
    pub fn new() -> Self {
        Self {
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }
}

impl Mutation for RemoveRandom {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying RemoveRandom to {:?}", solution);
        // Note: there always be at least one vessel in a `Problem`, and
        // 0..=x is always non-empty when x is an unsigned type
        let v = problem.indices::<Vessel>().choose(&mut self.rng).unwrap();

        match (0..solution[v].len()).choose(&mut self.rng) {
            Some(x) => {
                solution.mutate()[v].mutate().remove(x);
            }
            None => (),
        }
    }
}

// How we're going to perform the twerking.
#[derive(Debug)]
pub enum TwerkMode {
    Random,
    All,
}
/// "Shake" up the time at which each vessel's visits are performed.
pub struct Twerk {
    rng: rand::rngs::StdRng,
    mode: TwerkMode,
}

impl Twerk {
    pub fn some_random_person() -> Twerk {
        Twerk {
            rng: rand::rngs::StdRng::from_entropy(),
            mode: TwerkMode::Random,
        }
    }

    pub fn everybody() -> Twerk {
        Twerk {
            rng: rand::rngs::StdRng::from_entropy(),
            mode: TwerkMode::All,
        }
    }
}

impl Twerk {
    pub fn those_hips<R: rand::Rng>(rng: &mut R, problem: &Problem, plan: &mut [Visit]) {
        // Note: assumes that the visits are sorted in ascending order by time, which is normally enforced by the mutation guard.
        // However, if this is called after some other mutation that breaks that guarantee we might have to fix it here
        let total_time = plan
            .windows(2)
            .map(|w| w[1].time - w[0].time)
            .sum::<usize>();

        // The "average time" between two visits.
        let avg = total_time / plan.len().max(1);
        // We don't want the visits to cross too often, so we'll try to keep it such that they sheldom cross
        let max_delta = (avg / 3) as isize;
        // The highest allowed timestep
        let t_max = (problem.timesteps() - 1) as isize;

        for visit in plan {
            let delta = rng.gen_range(-max_delta..=max_delta);
            let new = visit.time as isize + delta;
            visit.time = new.clamp(0, t_max) as usize;
        }
    }
}

impl Mutation for Twerk {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        trace!("Applying Twerk({:?}) to {:?}", self.mode, solution);
        let rng = &mut self.rng;

        let mut plans = solution.mutate();

        match self.mode {
            TwerkMode::Random => match plans.choose_mut(rng) {
                Some(plan) => Twerk::those_hips(rng, problem, &mut plan.mutate()),
                None => warn!("unable to twerk"),
            },
            TwerkMode::All => {
                for plan in plans.iter_mut() {
                    Twerk::those_hips(rng, problem, &mut plan.mutate())
                }
            }
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum RedCostMode {
    /// Performs only one iteration where it updates the upper bounds of a random subset of visits
    /// that look promising to expand
    Mutate,
    /// Several iterations where it iteratively seeks to improve the soution by expanding visits
    LocalSerach,
}

/// This mutation exploits the dual solution of the quantities LP to direct the search towards a hopefulle better solution.
pub struct RedCost {
    mode: RedCostMode,
    max_visits: usize,
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
    pub fn red_cost_local_search(max_visits: usize) -> Self {
        let mode = RedCostMode::LocalSerach;

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
        let quant_lp = solution.quantities();
        let vars = solution.variables();
        let model = &quant_lp.model;

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
            let value = value.expect(&format!(
                "Not able to retrieve x-variable x_{}_{}_{}_{}",
                t, n, v, 0
            ));

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

        // If the chosen vessel has no visits return
        if solution[v].len() < 1 {
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
            RedCostMode::LocalSerach => todo!(),
        }
        trace!("FINISHED RED COST MUTATION")
    }
}

/// How to apply the Bounce
#[pyclass]
#[derive(Debug, Clone)]
pub enum BounceMode {
    All,
    Random,
}

/// A mutation that cause visits that are impossible to due to travel time to "bounce off" each other
/// in such a way that they (hopefully) get enought space to become travel time feasible.
pub struct Bounce {
    rng: rand::rngs::StdRng,
    passes: usize,
    mode: BounceMode,
}

impl Bounce {
    pub fn new(passes: usize, mode: BounceMode) -> Bounce {
        Bounce {
            rng: rand::rngs::StdRng::from_entropy(),
            passes,
            mode,
        }
    }

    pub fn apply_bounce_pass(
        problem: &Problem,
        vessel: VesselIndex,
        solution: &mut RoutingSolution,
    ) {
        let first = problem.origin_visit(vessel);
        let last = solution.artificial_end(vessel);
        let boat = &problem.vessels()[vessel];
        let max_t = problem.timesteps() - 1;

        let mut mutator = solution.mutate();
        let mut plan = mutator[vessel].mutate();

        for i in 0..plan.len() {
            // Note: `one.len() == i` and `two.len() == plan.len() - i`, by construction
            let (one, two) = plan.split_at_mut(i);
            let prev = one.last().unwrap_or(&first);
            // `i < plan.len()` => `two.len() > 0`
            let (two, three) = two.split_at_mut(1);
            let current = &mut two[0];
            // Note: `last` if None iff plan.len() == 0, in which case we wouldn't be in this loop
            let next = three.first().unwrap_or(last.as_ref().unwrap());

            // The time required to travel from `prev` to `current`, and from `current` to `next`
            let t1 = problem.travel_time(prev.node, current.node, boat);
            let t2 = problem.travel_time(current.node, next.node, boat);
            // The amount of warp we have in each direction.
            // Note: this is really just max(t1 - (dest.time - src.time), 0) formulated to avoid underflow of usize.
            let w1 = t1 - (current.time - prev.time).min(t1);
            let w2 = t2 - (next.time - current.time).min(t2);

            match w1.cmp(&w2) {
                // If there is more warp to the future than the past, we'll move `current` one timestep towards
                // the start if possible (this is effectively time = max(time - 1, 0) for usize)
                std::cmp::Ordering::Less => current.time = current.time - 1.min(current.time),
                // If they are equal we will not move any of them
                std::cmp::Ordering::Equal => (),
                // If there is more warp towards the past than the future we'll try to push the `current` visit
                // one step forward.
                std::cmp::Ordering::Greater => current.time = (current.time + 1).min(max_t),
            }
        }
    }
}

impl Mutation for Bounce {
    fn apply(
        &mut self,
        problem: &Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        trace!("Applying Bounce({:?}) to {:?}", self.mode, solution);
        for _ in 0..self.passes {
            match self.mode {
                BounceMode::All => {
                    for (v, _) in problem.vessels().iter().enumerate() {
                        Self::apply_bounce_pass(problem, v, solution);
                    }
                }
                BounceMode::Random => {
                    let vessel = problem
                        .vessels()
                        .choose(&mut self.rng)
                        .expect("there must be vessels")
                        .index();
                    Self::apply_bounce_pass(problem, vessel, solution)
                }
            }
        }
    }
}

pub struct IntraSwap;

impl Mutation for IntraSwap {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying IntraSwap to {:?}", solution);
        let mut rand = rand::thread_rng();
        // get random plan where a swap should be performed
        let v = rand.gen_range(0..problem.vessels().len());
        let mut mutator = solution.mutate();
        let plan = &mut mutator[v].mutate();

        // select two random visits to swap
        let v1 = rand.gen_range(0..plan.len());
        let v2 = rand.gen_range(0..plan.len());

        // if v1 and v2 are equal, we don't do anything
        if v1 == v2 {
            return;
        }

        // get the visits
        let (v1, v2) = plan.get_pair_mut(v1, v2);
        let n1 = v1.node;

        // perform the swap
        v1.node = v2.node;
        v2.node = n1;
    }
}

#[derive(Debug)]
pub enum TwoOptMode {
    /// Performs a 2-opt local search on every voyage for every vessel. The the u64 is the time limit for the local search per voyage
    LocalSerach(u64),
    /// Performs a random 2-opt mutation in a random vessel's route
    IntraRandom,
}

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
            let (visit1, visit2) = plan.get_pair_mut(v1, v2);
            let temp = visit1.node;

            // perform the swap
            visit1.node = visit2.node;
            visit2.node = temp;
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
            /* trace!(
                "Starting new iteration, count is {:?}. Start: {:?}, end: {:?}",
                count,
                start,
                end
            ); */
            // bool to say if we found an improving solution above threshold
            found_improving = false;

            for swap_first in start..(end - 2) {
                //trace!("here");
                for swap_last in (swap_first + 2)..end {
                    let change = Self::evaluate(plan, swap_first, swap_last, problem);
                    /* trace!(
                        "checking: {:?}, {:?}, change is {:?}",
                        swap_first,
                        swap_last,
                        change
                    ); */
                    if change < 0.0 {
                        found_improving = true;
                        /* trace!(
                            "found improving with change: {:?} for swap 1:{:?} 2: {:?}",
                            change,
                            swap_first,
                            swap_last
                        ); */
                        aggregated_improvement += f64::abs(change);

                        // move to next solution
                        Self::update(plan, swap_first, swap_last);
                        //trace!("Plan is now: {:?}", plan)
                    }
                }
            }
        }
        trace!("Ran local search for {} iterations from start: {} to end: {}, and reduced the total travel distance by {}", count, start, end, aggregated_improvement);
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
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
        trace!("Applying TwoOpt({:?}) to {:?}", self.mode, solution);
        match self.mode {
            TwoOptMode::LocalSerach(time_limit) => {
                println!("starting local search");
                let mutator = &mut solution.mutate();
                for plan in mutator.iter_mut() {
                    for interval in Self::production_visits(plan, problem).windows(2) {
                        let (start, end) = (interval[0], interval[1]);
                        Self::local_search(plan, start, end, problem, time_limit)
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

// swaps one random visit from one route with a visit from another route
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

        // select a random visit from each vessel
        let v1 = rand.gen_range(0..solution[vessel1].len());
        let v2 = rand.gen_range(0..solution[vessel2].len());

        let mutator = &mut solution.mutate();

        // perform the swap
        let (p1, p2) = &mut mutator.get_pair_mut(vessel1, vessel2);
        let visit1 = &mut p1.mutate()[v1];
        let visit2 = &mut p2.mutate()[v2];

        std::mem::swap(visit1, visit2);
    }
}

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
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution) {
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

pub struct VesselSwap {
    rand: ThreadRng,
}

impl Mutation for VesselSwap {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution) {
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

/// Solves a linear program to decide quantities and arrival times at each node
pub struct TimeSetter {
    quants_lp: QuantityLpCont,
}

impl TimeSetter {
    /// Create a TimeSetter mutation
    ///
    /// ## Arguments
    ///
    /// * `delay` - The mandatory delay that is added between visits for a vessel. A nonzero value will hopefully make the output from the continuous model fit a discrete time representation better.
    pub fn new(delay: f64) -> grb::Result<TimeSetter> {
        let quants_lp = QuantityLpCont::new(delay)?;
        Ok(TimeSetter { quants_lp })
    }
}

impl Mutation for TimeSetter {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution) {
        // solve the lp and retrieve the new time periods
        trace!("Applying TimeSetter to {:?}", solution);

        let new_times = self.quants_lp.get_visit_times(&solution);

        match new_times {
            Ok(times) => {
                let mutator = &mut solution.mutate();
                for vessel_idx in 0..times.len() {
                    for visit_idx in 0..times[vessel_idx].len() {
                        let new_time = times[vessel_idx][visit_idx];
                        let visit = &mut mutator[vessel_idx].mutate()[visit_idx];
                        // change arrival time
                        visit.time = new_time;
                    }
                }
            }
            Err(_) => return,
        }
    }
}
