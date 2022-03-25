use std::cell::Cell;
use std::{ops::Deref, sync::Arc};

use pyo3::pyclass;

use crate::problem::Problem;
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
        raw.sort_by_key(|v| v.time);
        Self { sorted: raw }
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

    pub fn warp(&self) -> usize {
        self.warp.get().unwrap_or_else(|| self.update_warp())
    }
}
