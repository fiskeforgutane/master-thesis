use std::cell::{Cell, Ref, RefCell, RefMut};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{DerefMut, RangeInclusive};
use std::rc::Rc;
use std::{ops::Deref, sync::Arc};

use itertools::{iproduct, Itertools};
use log::trace;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};
use slice_group_by::GroupBy;

use crate::models::quantity::QuantityLp;
use crate::problem::{
    Cost, FixedInventory, Inventory, NodeIndex, Problem, Product, Quantity, TimeIndex, VesselIndex,
};
use crate::solution::Visit;

/// A plan is a series of visits over a planning period, often attributed to a single vessel.
/// There are no restrictions on the ordering of the visits.
#[pyclass]
#[derive(Debug, Serialize, Deserialize)]
pub struct Plan {
    /// The origin visit of this plan
    origin: Visit,
    /// The set of visits sorted by ascending time
    sorted: Vec<Visit>,
}

/// We implement `Clone` manually to override `clone_from`.
impl Clone for Plan {
    fn clone(&self) -> Self {
        Self {
            sorted: self.sorted.clone(),
            origin: self.origin.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.sorted.clear();
        self.sorted.extend_from_slice(&source.sorted);
    }
}

impl Plan {
    fn new(origin: Visit, mut raw: Vec<Visit>) -> Self {
        // Sort by time, and then ensure that the origin visit occurs first.
        raw.sort_unstable_by_key(|&visit| (visit.time, visit != origin));

        let first = raw.first();

        if first.map(|v| v.time < origin.time).unwrap_or(false) {
            panic!("Visit occurs before origin");
        }

        match raw.first().map(|&v| v == origin).unwrap_or(false) {
            true => (),
            false => raw.insert(0, origin),
        }

        let plan = Self {
            sorted: raw,
            origin,
        };

        plan.validate();
        plan
    }

    pub fn mutate(&mut self) -> PlanMut<'_> {
        PlanMut(self)
    }

    pub fn origin(&self) -> Visit {
        self.origin
    }

    // Iterates over the plan, including its origin. Do not tamper with origin!
    pub fn iter_with_origin(
        &self,
        v: usize,
        problem: &Problem,
    ) -> impl Iterator<Item = Visit> + '_ {
        let node = problem.vessels()[v].origin();
        let time = problem.vessels()[v].available_from();
        let origin_visit = std::iter::once(Visit { node, time });
        origin_visit.chain(self.iter().cloned())
    }

    /// Check invariants. Assumes that `self.sorted` is sorted.
    fn validate(&self) {
        let origin = self.origin;
        let sorted = &self.sorted;

        // Enforce that the first visit is equal to the origin visit
        let first = sorted.first();
        trace!(
            "origin: {:?}, is now: {:?}",
            origin.node,
            first.unwrap().node
        );
        assert!(first.map(|&v| v == origin).unwrap_or(false));

        // Enforce that there is at least one time step between consecutive visits
        assert!(sorted
            .iter()
            .tuple_windows()
            .all(|(x, y)| y.time - x.time >= 1));
    }
}

impl Deref for Plan {
    type Target = [Visit];

    fn deref(&self) -> &Self::Target {
        &self.sorted
    }
}

impl<'s> IntoIterator for &'s Plan {
    type Item = &'s Visit;

    type IntoIter = <&'s [Visit] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        let slice: &[Visit] = self;
        slice.iter()
    }
}

#[derive(Debug)]
/// A mutable reference to a plan that enforces that enforces that
/// invariants are upheld after this goes out of scope.
pub struct PlanMut<'a>(&'a mut Plan);

impl<'a> PlanMut<'a> {
    /// Fixes a plan that has been mutated in order to satisfy the "no simultaneous visits"-requirement by:
    /// 1. Attempting to move one of the simultaneous visits forward/backward until it encounters a free time slot.
    ///    The preferred movement way is the one that maintains the ordering of the visits
    /// 2. If (1) fails, the visit is simply removed
    pub fn fix(&mut self) {
        // Ensure that we're sorted by time.
        self.sort_unstable_by_key(|v| v.time);
        // Attempt to move `current` in order to avoid collisions
        // Note 1: this only does one pass, so it cannot handle all types of collisions.
        // Note 2: the last visit is not ever changed
        for i in 1..self.len() - 1 {
            let prev = self[i - 1];
            let next = self[i + 1];
            let current = &mut self[i];
            if next.time == current.time && current.time - prev.time > 1 {
                current.time -= 1;
            }

            if prev.time == current.time && next.time - current.time > 1 {
                current.time += 1;
            }
        }

        let mut remove = Vec::new();
        for i in 1..self.len() {
            if self[i].time == self[i - 1].time {
                remove.push(i);
            }
        }

        // Note: by going over this in reverse order, we prevent any "shuffling" of the indices
        // (since `remove` is constructed sorted ascending)
        for &i in remove.iter().rev() {
            self.swap_remove(i);
        }
    }
}

impl PlanMut<'_> {
    pub fn origin(&self) -> Visit {
        self.0.origin()
    }
}

impl Deref for PlanMut<'_> {
    type Target = Vec<Visit>;

    fn deref(&self) -> &Self::Target {
        &self.0.sorted
    }
}

impl DerefMut for PlanMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.sorted
    }
}

impl Drop for PlanMut<'_> {
    fn drop(&mut self) {
        let origin = self.0.origin;
        let sorted = &mut self.0.sorted;
        sorted.sort_unstable_by_key(|&visit| (visit.time, visit != origin));

        self.0.validate();
    }
}

/// A cache for use within `RoutingSolution`
pub struct Cache {
    /// The amount of time warp in this plan i.e. the sum of the number of time periods we would need to "missing timesteps"
    /// between visits where it is physically impossible to travel between two nodes (only considering travel time; not loading time)
    warp: Cell<Option<usize>>,
    /// The quantity LP associated with this plan. This is the thing that
    /// will determine (optimal) quantities given the current set of routes.
    quantity: Rc<RefCell<QuantityLp>>,
    /// The total inventory violations (excess/shortage)
    violation: Cell<Option<Quantity>>,
    /// The total cost of the solution.
    cost: Cell<Option<Cost>>,
    /// The total revenue of the solution.
    revenue: Cell<Option<Cost>>,
    /// The timing punishment of the solution.
    timing: Cell<Option<Cost>>,
    /// A hint to the amount of sum of berth violations in the solution
    /// The `hint` is simply sum(max(visits at node `n` at time `t` - berth capacity, 0) for all (n, t))
    approx_berth_violation: Cell<Option<usize>>,
}

/// A solution of the routing within a `Problem`, i.e. where and when each vessel arrives at different nodes throughout the planning period.
/// It does not quarantee feasibility in any manner, it is possible to write a routing solution that e.g. requires a vessel to be present at two different nodes
/// at the exact same time. Avoiding this is usually the responsibility of whoever creates the solution (e.g. the GA)
pub struct RoutingSolution {
    /// The problem this routing solution `belongs` to.
    problem: Arc<Problem>,
    /// The routes of each vessel.
    routes: Vec<Plan>,
    /// Cache for the evaluation of the solution
    cache: Cache,
}

impl Debug for RoutingSolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoutingSolution")
            .field("problem", &self.problem)
            .field("routes", &self.routes)
            .finish_non_exhaustive()
    }
}

impl Clone for RoutingSolution {
    fn clone(&self) -> Self {
        Self {
            problem: self.problem.clone(),
            routes: self.routes.clone(),
            cache: Cache {
                warp: self.cache.warp.clone(),
                quantity: self.cache.quantity.clone(),
                violation: Cell::new(None),
                cost: Cell::new(None),
                revenue: Cell::new(None),
                timing: Cell::new(None),
                approx_berth_violation: Cell::new(None),
            },
        }
    }

    fn clone_from(&mut self, source: &Self) {
        // In "all" cases, we expect `self.routes.len()` to equal `source.routes.len()`
        for (one, two) in self.routes.iter_mut().zip(&source.routes) {
            one.clone_from(two);
        }

        // However: we should still handle the case where the lengths are different from each other
        let l = self.routes.len();
        let r = source.routes.len();
        match l.cmp(&r) {
            Ordering::Less => self.routes.extend_from_slice(&source.routes[l..]),
            Ordering::Equal => (),
            Ordering::Greater => self.routes.truncate(r),
        }

        // Copy over the problem
        self.problem = source.problem.clone();

        // Invalidate the caches, and then pass over any data that would still be valid to use for the clone,
        // which is basically everything except the LP itself
        self.invalidate_caches();
        self.cache.warp = source.cache.warp.clone();
        self.cache.violation = source.cache.violation.clone();
        self.cache.cost = source.cache.cost.clone();
        self.cache.revenue = source.cache.revenue.clone();
    }
}

impl Deref for RoutingSolution {
    type Target = [Plan];

    fn deref(&self) -> &Self::Target {
        &self.routes
    }
}

impl RoutingSolution {
    pub fn empty(problem: Arc<Problem>) -> Self {
        let routes = vec![Vec::new(); problem.vessels().len()];
        RoutingSolution::new(problem, routes)
    }

    /// Construct a new RoutingSolution with a RefCell to the given QuantityLp
    pub fn new_with_model(
        problem: Arc<Problem>,
        routes: Vec<Vec<Visit>>,
        model: Rc<RefCell<QuantityLp>>,
    ) -> Self {
        // We won't bother returning a result from this, since it'll probably just be .unwrapped() anyways
        if routes.len() != problem.vessels().len() {
            panic!("#r = {} != V = {}", routes.len(), problem.vessels().len());
        }

        let cache = Cache {
            warp: Cell::default(),
            quantity: model,
            violation: Cell::new(None),
            cost: Cell::new(None),
            revenue: Cell::new(None),
            timing: Cell::new(None),
            approx_berth_violation: Cell::new(None),
        };

        Self {
            routes: routes
                .into_iter()
                .enumerate()
                .map(|(v, route)| Plan::new(problem.origin_visit(v), route))
                .collect(),
            problem,
            cache,
        }
    }

    pub fn new(problem: Arc<Problem>, routes: Vec<Vec<Visit>>) -> Self {
        let model = Rc::new(RefCell::new(QuantityLp::new(&problem).unwrap()));
        Self::new_with_model(problem, routes, model)
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// An artificial "terminal visit" for a plan at the same node as the last actual visit at the timestep
    /// just after the end of the planning horizon
    pub fn artificial_end(&self, vessel: VesselIndex) -> Option<Visit> {
        self[vessel].last().map(|v| Visit {
            node: v.node,
            time: self.problem().timesteps(),
        })
    }

    /// Loop over each plan in order, and include the origin visit of the vessel as the first visit
    pub fn iter_with_origin(&self) -> impl Iterator<Item = impl Iterator<Item = Visit> + '_> + '_ {
        self.iter().enumerate().map(|(v, plan)| {
            let vessel = &self.problem.vessels()[v];
            let node = vessel.origin();
            let time = vessel.available_from();
            let first = Visit { node, time };

            assert!(plan.first() == Some(&first));

            plan.iter().cloned()
        })
    }

    /// Loop over each plan in order, including the origin visit at the start,
    /// and an artificial terminal visit at the very end.
    pub fn iter_with_terminals(
        &self,
    ) -> impl Iterator<Item = impl Iterator<Item = Visit> + '_> + '_ {
        self.iter().enumerate().map(|(v, plan)| {
            let vessel = &self.problem.vessels()[v];
            let node = vessel.origin();
            let time = vessel.available_from();
            let first = Visit { node, time };
            let end = plan.last().map(|v| Visit {
                node: v.node,
                time: self.problem.timesteps() - 1,
            });

            assert!(plan.first() == Some(&first));

            plan.iter().cloned().chain(end)
        })
    }

    /// Access a mutator for this RoutingSolution. This ensures that any caches are always updated
    /// as needed, but no more.
    pub fn mutate(&mut self) -> RoutingSolutionMut<'_> {
        RoutingSolutionMut(self)
    }

    /// Retrieve a reference to the quantity assignment LP.
    pub fn quantities(&self) -> Ref<'_, QuantityLp> {
        // If the LP hasn't been solved for the current state, we'll do so
        let cache = &self.cache;
        let mut lp = self.cache.quantity.borrow_mut();
        lp.configure(self, false, false).expect("configure failed");
        lp.solve().expect("solve failed");
        std::mem::drop(lp);

        cache.quantity.borrow()
    }

    /// Force an exact solution for the quantities delivered.
    /// This will use semicont variables for the amount delivered, turning the quantity
    /// assignment from an LP to a MILP. This can take a considerable amount of time to solve
    pub fn exact_mut(&self) -> RefMut<'_, QuantityLp> {
        // This will trigger a (possibly) different quantity assignment, so
        // we will need to invalidate the caches.
        self.invalidate_caches();

        let mut lp = self.cache.quantity.borrow_mut();
        lp.configure(self, true, true).expect("configure failed");
        lp.solve().expect("solve failed");

        lp
    }

    /// Force an exact solution for the quantities delivered.
    /// This will use semicont variables for the amount delivered, turning the quantity
    /// assignment from an LP to a MILP. This can take a considerable amount of time to solve
    pub fn exact(&self) -> Ref<'_, QuantityLp> {
        let _ = self.exact_mut();
        self.cache.quantity.borrow()
    }

    /// Force an evaluation for the quantites delivered using the inexact model.

    /// Retrieve the amount of time warp in this solution. Time warp occurs when two visits at different nodes are
    /// too close apart in time, such that it is impossible to go from one of them to the other in time.
    pub fn warp(&self) -> usize {
        let cached = &self.cache.warp;
        cached.get().unwrap_or_else(|| self.update_warp())
    }

    /// Retrieve the `approx_berth_violation` of this solution
    pub fn approx_berth_violation(&self) -> usize {
        let cached = &self.cache.approx_berth_violation;

        cached
            .get()
            .unwrap_or_else(|| self.update_approx_berth_violation())
    }

    /// The total inventory violation (excess + shortage) for the entire planning period
    pub fn violation(&self) -> Quantity {
        let cached = &self.cache.violation;
        if cached.get().is_none() {
            self.update();
        }

        cached.get().unwrap()
    }

    /// Retrieves the total cost for the deliveries done by the solution.
    /// Note: if there are several consecutive `Visit`s to the same node, we will only incur
    /// a port fee on the first visit.
    pub fn cost(&self) -> Cost {
        let cached = &self.cache.cost;
        if cached.get().is_none() {
            self.update();
        }

        cached.get().unwrap()
    }

    /// Retrieve the total revenue for the amount delivered by in the solution
    pub fn revenue(&self) -> Cost {
        let cached = &self.cache.revenue;
        if cached.get().is_none() {
            self.update();
        }

        cached.get().unwrap()
    }

    /// Recycle this solution into a new fresh one with no content.
    /// This allows us to reuse the inner heap-allocated structures.
    /// The result is an empty solution that is unlikely to need allocations when pushing visits,
    /// and which will reuse the quantity LP.
    pub fn recycle(mut self) -> Self {
        // We clear any old data. Since caches are invalidated, the new owner of our data
        // will not be able to see our (now deleted) data or our cached data.
        for plan in self.mutate().iter_mut() {
            plan.mutate().clear();
        }

        Self {
            problem: self.problem,
            routes: self.routes,
            cache: self.cache,
        }
    }

    fn update_warp(&self) -> usize {
        let mut warp = 0;
        for (v, route) in self.routes.iter().enumerate() {
            let vessel = &self.problem.vessels()[v];
            for visits in route.windows(2) {
                let (v1, v2) = (visits[0], visits[1]);

                let needed = self.problem.travel_time(v1.node, v2.node, vessel);
                let available = v2.time - v1.time;
                // Note: equivalent to max(needed - available, 0) for signed variables.
                // we need to use this instead to avoid underflow.
                warp += needed.max(available) - available;
            }
        }
        self.cache.warp.set(Some(warp));
        warp
    }

    fn update_approx_berth_violation(&self) -> usize {
        let sorted = self
            .iter()
            .flat_map(|x| x.iter().cloned())
            .sorted_unstable_by_key(|v| (v.time, v.node))
            .collect::<Vec<_>>();

        let nodes = self.problem().nodes();

        let violation = sorted
            .linear_group()
            .map(|group| {
                // The `group` is guaranteed to be non-empty, by construction
                // additionally, all visits are equivalent (also by construction)
                let visit = group[0];
                let used = group.len();

                let capacity = nodes[visit.node].port_capacity()[visit.time];
                // An unsigned-equivalent version of max(used - capacity, 0)
                used.max(capacity) - capacity
            })
            .sum();

        self.cache.approx_berth_violation.set(Some(violation));

        violation
    }

    fn update_violation(&self, quantities: &QuantityLp) -> f64 {
        let lp = quantities;
        let violation = lp
            .model
            .get_obj_attr(grb::attr::X, &lp.vars.violation)
            .expect("failed to retrieve variables");
        self.cache.violation.set(Some(violation));
        violation
    }

    fn update_timing(&self, quantities: &QuantityLp) -> f64 {
        let lp = quantities;
        let timing = lp
            .model
            .get_obj_attr(grb::attr::X, &lp.vars.timing)
            .expect("failed to retrieve variables");
        self.cache.timing.set(Some(timing));
        timing
    }

    fn update_cost(&self, quantities: &QuantityLp) -> f64 {
        let problem = self.problem();
        let lp = quantities;
        let load = &lp.vars.l;
        let p = problem.count::<Product>();
        let mut inventory = Inventory::zeroed(p).unwrap();
        let cost = self
            .iter()
            .enumerate()
            .map(|(v, plan)| {
                problem.nodes()[problem.vessels()[v].origin()].port_fee()
                    + plan
                        .iter()
                        .tuple_windows()
                        .map(|(v1, v2)| {
                            // NOTE: the time might be an off-by-one error
                            // NOTE2: might consider batching this Gurobi call.
                            let t = v2.time;
                            for p in 0..p {
                                inventory[p] =
                                    lp.model.get_obj_attr(grb::attr::X, &load[t][v][p]).unwrap();
                            }

                            let travel = problem.travel_cost(v1.node, v2.node, v, &inventory);
                            let port_fee = match v1.node == v2.node {
                                true => 0.0,
                                false => problem.nodes()[v2.node].port_fee(),
                            };

                            travel + port_fee
                        })
                        .sum::<f64>()
            })
            .sum::<f64>();

        self.cache.cost.set(Some(cost));

        cost
    }

    fn update_revenue(&self, quantities: &QuantityLp) -> Cost {
        // We need to retrieve the solution variables, and loop over the ones
        // that are non-zero to determine the revenue
        let lp = quantities;
        let revenue = lp
            .model
            .get_obj_attr(grb::attr::X, &lp.vars.revenue)
            .expect("retrieving variable values failed");

        self.cache.revenue.set(Some(revenue));
        revenue
    }

    /// Recalculate all the cached values
    fn update(&self) {
        let quantities = self.quantities();
        self.update_cost(&quantities);
        self.update_revenue(&quantities);
        self.update_violation(&quantities);
        self.update_timing(&quantities);
    }

    /// Invalidate all the cached values on this object (objective function values, etc.)
    fn invalidate_caches(&self) {
        self.cache.warp.set(None);
        self.cache.violation.set(None);
        self.cache.cost.set(None);
        self.cache.revenue.set(None);
        self.cache.approx_berth_violation.set(None);
        self.cache.timing.set(None);
    }

    /// Convert to a double array of visits
    pub fn to_vec(&self) -> Vec<Vec<Visit>> {
        self.iter()
            .map(|plan| plan.iter().cloned().collect())
            .collect()
    }

    /// An iterator that enumerates all the possible visits that can be inserted into the solution
    /// without incurring any additional time warp. Consecutive visits to the same node will not be included.
    pub fn available(
        &self,
    ) -> impl Iterator<
        Item = (
            VesselIndex,
            impl Iterator<Item = (NodeIndex, RangeInclusive<usize>)> + '_,
        ),
    > + '_ {
        let problem = self.problem();
        self.iter_with_terminals()
            .enumerate()
            .map(move |(v, plan)| {
                let vessel = &problem.vessels()[v];
                (
                    v,
                    plan.tuple_windows().flat_map(move |(v1, v2)| {
                        problem
                            .nodes()
                            .iter()
                            .filter(move |n| n.index() != v1.node && n.index() != v2.node)
                            .filter_map(move |n| {
                                let t1 = problem.travel_time(v1.node, n.index(), vessel);
                                let t2 = problem.travel_time(n.index(), v2.node, vessel);

                                // TODO: check off by one (?)
                                let earliest_arrival = v1.time + t1;
                                let latest_departure = v2.time.max(t2) - t2;
                                let available = earliest_arrival..=latest_departure;

                                match available.is_empty() {
                                    true => None,
                                    false => Some((n.index(), available)),
                                }
                            })
                    }),
                )
            })
    }

    pub fn duration(&self, plan_idx: usize, visit_idx: usize) -> usize {
        let lp = &self.quantities();
        let visit = &self[plan_idx][visit_idx];

        let mut count = 0;
        for t in visit.time..self.problem().timesteps() {
            if !lp
                .model
                .get_obj_attr_batch(
                    grb::attr::X,
                    (0..self.problem.products()).map(|p| lp.vars.x[t][visit.node][plan_idx][p]),
                )
                .expect("failed to retrieve variables")
                .into_iter()
                .any(|v| v > 1e-5)
            {
                break;
            }
            count += 1;
        }

        count
    }

    pub fn candidates<'a>(
        &'a self,
        visit_idx: usize,
        plan_idx: usize,
        c: usize,
    ) -> impl Iterator<Item = (VesselIndex, Visit)> + 'a {
        let current_visit = &self[plan_idx][visit_idx];

        let vessel = &self.problem().vessels()[plan_idx];
        // the time period of the next visit, if none, the length of the planning period
        let time_bound = match self[plan_idx].get(visit_idx + 1) {
            Some(v) => v.time,
            None => self.problem().timesteps(),
        };
        self.problem()
            .nodes()
            .into_iter()
            .filter(|n| n.index() != current_visit.node)
            .filter_map(move |n| {
                let travel_time = self
                    .problem()
                    .travel_time(current_visit.node, n.index(), vessel);

                let arrival = current_visit.time + travel_time;
                match arrival < time_bound {
                    true => Some(
                        (arrival..(arrival + c).min(self.problem.timesteps() - 1)).map(move |t| {
                            (
                                plan_idx,
                                Visit {
                                    node: n.index(),
                                    time: t,
                                },
                            )
                        }),
                    ),
                    false => None,
                }
            })
            .flatten()
    }

    /// The load of the given vessel at the **beginning** of the given time period
    pub fn load_at(&self, vessel: VesselIndex, time: TimeIndex) -> FixedInventory {
        let lp = self.quantities();

        let vars = iproduct!(0..self.problem().products())
            .map(|p| lp.vars.l[time][vessel][p])
            .collect::<Vec<_>>();

        let load = lp
            .model
            .get_obj_attr_batch(grb::attr::X, vars)
            .expect("failed to retrieve variables");

        FixedInventory::from(Inventory::new(load.as_slice()).unwrap())
    }

    /// The inventory at the given node at the **beginning** of the given time period
    pub fn inventory_at(&self, node: NodeIndex, time: TimeIndex) -> FixedInventory {
        let lp = self.quantities();

        let vars = iproduct!(0..self.problem().products())
            .map(|p| lp.vars.s[time][node][p])
            .collect::<Vec<_>>();

        let inventory = lp
            .model
            .get_obj_attr_batch(grb::attr::X, vars)
            .expect("failed to retrieve variables");

        FixedInventory::from(Inventory::new(inventory.as_slice()).unwrap())
    }

    /// get first position of the given vessel from the given time period
    pub fn next_position(&self, vessel: VesselIndex, time: TimeIndex) -> (NodeIndex, TimeIndex) {
        let visit = self.routes[vessel]
            .into_iter()
            .find_or_last(|v| v.time >= time)
            .unwrap();
        (visit.node, visit.time)
    }
}

impl<'a> IntoIterator for &'a RoutingSolution {
    type Item = &'a Plan;

    type IntoIter = <&'a [Plan] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

pub struct RoutingSolutionMut<'a>(&'a mut RoutingSolution);

impl Deref for RoutingSolutionMut<'_> {
    type Target = [Plan];

    fn deref(&self) -> &Self::Target {
        &self.0.routes
    }
}

impl DerefMut for RoutingSolutionMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.routes
    }
}

impl Drop for RoutingSolutionMut<'_> {
    fn drop(&mut self) {
        self.0.invalidate_caches();

        let timesteps = self.0.problem.timesteps();

        // Check that the visit times are correct
        for (v, plan) in self.0.routes.iter().enumerate() {
            // Ensure that the last timestep is within the planning period.
            assert!(
                match plan.last() {
                    Some(visit) => visit.time < timesteps,
                    None => true,
                },
                "Some(visit) => visit.time < timesteps, None => true, visit:{:?}",
                plan.last()
            );
            // Assert that the first visit of each vessel's plan corresponds to its origin visit.
            assert!(plan
                .first()
                .map(|&visit| visit == self.0.problem.origin_visit(v))
                .unwrap_or(false));
        }
    }
}
