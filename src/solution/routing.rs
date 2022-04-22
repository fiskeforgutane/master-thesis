use std::cell::{Cell, Ref, RefCell};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::DerefMut;
use std::{ops::Deref, sync::Arc};

use itertools::Itertools;
use log::trace;
use pyo3::pyclass;
use serde::{Deserialize, Serialize};

use crate::models::quantity::{QuantityLp, Variables};
use crate::problem::{Cost, Inventory, Problem, Product, Quantity, VesselIndex};
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
    quantity: RefCell<QuantityLp>,
    /// Whether the LP model has been solved for the current solution
    solved: Cell<bool>,
    /// The total inventory violations (excess/shortage)
    violation: Cell<Option<Quantity>>,
    /// The total cost of the solution.
    cost: Cell<Option<Cost>>,
    /// The total revenue of the solution.
    revenue: Cell<Option<Cost>>,
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
                quantity: RefCell::new(
                    QuantityLp::new(self.problem()).expect("cloning failed for routing solution"),
                ),
                solved: Cell::new(false),
                violation: Cell::new(None),
                cost: Cell::new(None),
                revenue: Cell::new(None),
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
        self.cache.solved = Cell::new(false);
        self.cache.violation = source.cache.violation.clone();
        self.cache.cost = source.cache.cost.clone();
    }
}

impl Deref for RoutingSolution {
    type Target = [Plan];

    fn deref(&self) -> &Self::Target {
        &self.routes
    }
}

impl RoutingSolution {
    pub fn new(problem: Arc<Problem>, routes: Vec<Vec<Visit>>) -> Self {
        // We won't bother returning a result from this, since it'll probably just be .unwrapped() anyways
        if routes.len() != problem.vessels().len() {
            panic!("#r = {} != V = {}", routes.len(), problem.vessels().len());
        }

        let cache = Cache {
            warp: Cell::default(),
            quantity: RefCell::new(QuantityLp::new(&problem).unwrap()),
            solved: Cell::new(false),
            violation: Cell::new(None),
            cost: Cell::new(None),
            revenue: Cell::new(None),
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
                time: self.problem.timesteps(),
            });

            assert!(plan.first() == Some(&first));

            plan.iter().cloned().chain(end)
        })
    }

    /// Retrieve the amount of time warp in this solution. Time warp occurs when two visits at different nodes are
    /// too close apart in time, such that it is impossible to go from one of them to the other in time.
    pub fn warp(&self) -> usize {
        self.cache.warp.get().unwrap_or_else(|| self.update_warp())
    }

    /// Access a mutator for this RoutingSolution. This ensures that any caches are always updated
    /// as needed, but no more.
    pub fn mutate(&mut self) -> RoutingSolutionMut<'_> {
        RoutingSolutionMut(self)
    }

    /// Retrieve a reference to the quantity assignment LP.
    /// The model is re-solved if needed.
    pub fn quantities(&self) -> Ref<'_, QuantityLp> {
        // If the LP hasn't been solved for the current state, we'll do so
        let cache = &self.cache;
        if !cache.solved.get() {
            let mut lp = self.cache.quantity.borrow_mut();
            lp.configure(self).expect("configure failed");
            lp.solve().expect("solve failed");
            self.cache.solved.set(true);
        }

        cache.quantity.borrow()
    }

    /// Force an exact solution for the quantities delivered.
    /// This will use semicont variables for the amount delivered, turning the quantity
    /// assignment from an LP to a MILP. This can take a considerable amount of time to solve
    pub fn exact(&mut self) {
        // This will trigger a (possibly) different quantity assignment, so
        // we will need to invalidate the caches.
        self.invalidate_caches();

        let old = self.cache.quantity.borrow().semicont;
        // Set the QuantityLp to use semicont for the x variables
        self.cache.quantity.get_mut().semicont = true;
        // Force evaluation (i.e. solve MILP)
        let _ = self.quantities();
        // Restore the old preference w.r.t semicont or not
        self.cache.quantity.get_mut().semicont = old;
    }

    /// Retrieve a reference to the variables of the quantity assignment LP.
    pub fn variables(&self) -> Ref<'_, Variables> {
        Ref::map(self.quantities(), |lp| &lp.vars)
    }

    /// The total inventory violation (excess + shortage) for the entire planning period
    pub fn violation(&self) -> Quantity {
        let cache = &self.cache;

        match cache.violation.get() {
            Some(violation) => violation,
            // Solve the LP (if needed) and retrieve the objective function value.
            None => {
                let obj = self.quantities().model.get_attr(grb::attr::ObjVal).unwrap();
                cache.violation.set(Some(obj));
                obj
            }
        }
    }

    /// Retrieves the total cost for the deliveries done by the solution.
    /// Note: if there are several consecutive `Visit`s to the same node, we will only incur
    /// a port fee on the first visit.
    pub fn cost(&self) -> Cost {
        if let Some(cost) = self.cache.cost.get() {
            return cost;
        }

        let problem = self.problem();
        let lp = self.quantities();
        let load = &lp.vars.l;
        let p = problem.count::<Product>();
        let mut inventory = Inventory::zeroed(p).unwrap();
        let cost = self
            .iter_with_origin()
            .enumerate()
            .map(|(v, plan)| {
                plan.tuple_windows()
                    .map(|(v1, v2)| {
                        // NOTE: the time might be an off-by-one error
                        // NOTE2: might consider batching this Gurobi call.
                        let t = v2.time;
                        for p in 0..p {
                            inventory[p] =
                                lp.model.get_obj_attr(grb::attr::X, &load[t][v][p]).unwrap();
                        }

                        let travel = problem.travel_cost(v1.node, v2.node, v, &inventory);
                        let port_fee = problem.nodes()[v2.node].port_fee();

                        travel + port_fee
                    })
                    .sum::<f64>()
            })
            .sum::<f64>();

        self.cache.cost.set(Some(cost));

        cost
    }

    /// Retrieve the total revenue for the amount delivered by in the solution
    pub fn revenue(&self) -> Cost {
        let cache = &self.cache;

        match cache.revenue.get() {
            Some(revenue) => revenue,
            None => {
                // We need to retrieve the solution variables, and loop over the ones
                // that are non-zero to determine the revenue
                let lp = self.quantities();
                let x = &lp.vars.x;
                let nodes = self.problem().nodes();
                let variables = lp
                    .model
                    .get_obj_attr_batch(
                        grb::attr::X,
                        QuantityLp::active(self).map(|(t, n, v, p)| x[t][n][v][p]),
                    )
                    .expect("retrieving variable values failed");

                let revenue = QuantityLp::active(self)
                    .zip_eq(&variables)
                    .map(|((_, n, _, _), quantity)| nodes[n].revenue() * quantity)
                    .sum();

                cache.revenue.set(Some(revenue));
                revenue
            }
        }
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

    /// Invalidate all the cached values on this object (objective function values, etc.)
    fn invalidate_caches(&self) {
        self.cache.warp.set(None);
        self.cache.solved.set(false);
        self.cache.violation.set(None);
        self.cache.cost.set(None);
        self.cache.revenue.set(None);
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
            assert!(match plan.last() {
                Some(visit) => visit.time < timesteps,
                None => true,
            });
            // Assert that the first visit of each vessel's plan corresponds to its origin visit.
            assert!(plan
                .first()
                .map(|&visit| visit == self.0.problem.origin_visit(v))
                .unwrap_or(false));
        }
    }
}
