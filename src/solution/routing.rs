use std::borrow::{Borrow, BorrowMut};
use std::cell::{Cell, Ref, RefCell};
use std::fmt::Debug;
use std::ops::DerefMut;
use std::{ops::Deref, sync::Arc};

use pyo3::pyclass;

use crate::models::quantity::{QuantityLp, Variables};
use crate::problem::{Cost, Problem, Quantity, VesselIndex};
use crate::solution::Visit;

/// A plan is a series of visits over a planning period, often attributed to a single vessel.
/// There are no restrictions on the ordering of the visits.
#[pyclass]
#[derive(Debug, Clone)]
pub struct Plan {
    /// The set of visits sorted by ascending time
    sorted: Vec<Visit>,
}

impl Plan {
    fn new(mut raw: Vec<Visit>) -> Self {
        raw.sort_unstable_by_key(|visit| visit.time);
        Self { sorted: raw }
    }

    pub fn mutate(&mut self) -> PlanMut<'_> {
        PlanMut(self)
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

/// A mutable reference to a plan that enforces that enforces that
/// invariants are upheld after this goes out of scope.
pub struct PlanMut<'a>(&'a mut Plan);

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
        self.0.sorted.sort_unstable_by_key(|visit| visit.time);
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
            },
        }
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
        };

        Self {
            problem,
            routes: routes.into_iter().map(|route| Plan::new(route)).collect(),
            cache,
        }
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    /// Loop over each plan in order, and include the origin visit of the vessel as the first visit
    pub fn iter_with_origin(&self) -> impl Iterator<Item = impl Iterator<Item = Visit> + '_> + '_ {
        self.iter().enumerate().map(|(v, plan)| {
            let vessel = &self.problem.vessels()[v];
            let node = vessel.origin();
            let time = vessel.available_from();
            let first = std::iter::once(Visit { node, time });

            first.chain(plan.iter().cloned())
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
            let first = std::iter::once(Visit { node, time });
            let end = plan.last().map(|v| Visit {
                node: v.node,
                time: self.problem.timesteps(),
            });

            first.chain(plan.iter().cloned()).chain(end)
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

impl<'a> RoutingSolutionMut<'a> {
    /// Get mutable references for two separate vessels.
    pub fn get_pair_mut(&mut self, v1: VesselIndex, v2: VesselIndex) -> (&mut Plan, &mut Plan) {
        assert!(v1 != v2);
        let min = v1.min(v2);
        let max = v2.max(v1);

        let (one, rest) = self[min..].split_first_mut().unwrap();
        let two = &mut rest[max - min - 1];

        (one, two)
    }
}

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
        for plan in &self.0.routes {
            assert!(match plan.last() {
                Some(visit) => visit.time < timesteps,
                None => true,
            });
        }
    }
}
