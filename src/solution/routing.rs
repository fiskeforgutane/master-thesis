use std::{ops::Deref, sync::Arc};

use pyo3::pyclass;

use crate::problem::Problem;
use crate::solution::Visit;

/// A plan is a series of visits over a planning period, often attributed to a single vessel.
#[pyclass]
#[derive(Debug, Clone)]
pub struct Plan {
    /// The raw set of visits, setting no constraints on the ordering of them.
    raw: Vec<Visit>,
}

impl Plan {
    fn new(raw: Vec<Visit>) -> Self {
        Self { raw }
    }
}

impl Deref for Plan {
    type Target = [Visit];

    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

pub enum RoutingError {
    IncorrectVesselCount,
}

/// A solution of the routing within a `Problem`, i.e. where and when each vessel arrives at different nodes throughout the planning period.
#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingSolution {
    /// The problem this routing solution `belongs` to.
    problem: Arc<Problem>,
    /// The routes of each vessel.
    routes: Vec<Plan>,
}

impl Deref for RoutingSolution {
    type Target = [Plan];

    fn deref(&self) -> &Self::Target {
        &self.routes
    }
}

impl RoutingSolution {
    pub fn new(problem: Arc<Problem>, routes: Vec<Vec<Visit>>) -> Result<Self, RoutingError> {
        if routes.len() != problem.vessels().len() {
            return Err(RoutingError::IncorrectVesselCount);
        }

        Ok(Self {
            problem,
            routes: routes.into_iter().map(|route| Plan::new(route)).collect(),
        })
    }
}
