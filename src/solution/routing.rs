use std::cell::Cell;
use std::ops::DerefMut;
use std::{ops::Deref, sync::Arc};

use pyo3::pyclass;

use crate::problem::{Problem, VesselIndex};
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

    pub fn visits(&self) -> Vec<Visit> {
        self.sorted.clone()
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

/// A solution of the routing within a `Problem`, i.e. where and when each vessel arrives at different nodes throughout the planning period.
/// It does not quarantee feasibility in any manner, it is possible to write a routing solution that e.g. requires a vessel to be present at two different nodes
/// at the exact same time. Avoiding this is usually the responsibility of whoever creates the solution (e.g. the GA)
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingSolution {
    /// The problem this routing solution `belongs` to.
    problem: Arc<Problem>,
    /// The routes of each vessel.
    routes: Vec<Plan>,
    /// The amount of time warp in this plan i.e. the sum of the number of time periods we would need to "missing timesteps"
    /// between visits where it is physically impossible to travel between two nodes (only considering travel time; not loading time)
    warp: Cell<Option<usize>>,
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

        Self {
            problem,
            routes: routes.into_iter().map(|route| Plan::new(route)).collect(),
            warp: Cell::default(),
        }
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn routes(&self) -> &Vec<Plan> {
        &self.routes
    }

    pub fn warp(&self) -> usize {
        self.warp.get().unwrap_or_else(|| self.update_warp())
    }

    pub fn mutate(&mut self) -> RoutingSolutionMut<'_> {
        RoutingSolutionMut(self)
    }

    fn update_warp(&self) -> usize {
        let mut warp = 0;
        for (v, route) in self.routes.iter().enumerate() {
            let vessel = &self.problem.vessels()[v];
            for visits in route.windows(2) {
                let (v1, v2) = (visits[0], visits[1]);

                let needed = self.problem.travel_time(v1.node, v2.node, vessel);
                let available = v2.time - v1.time;

                warp += std::cmp::max(needed - available, 0);
            }
        }

        self.warp.set(Some(warp));
        warp
    }

    fn invalidate_caches(&self) {
        self.warp.set(None);
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
