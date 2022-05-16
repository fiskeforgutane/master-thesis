use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

use pyo3::pyclass;

use crate::{problem::Problem, solution::routing::RoutingSolution};

use super::Fitness;

#[pyclass]
#[derive(Clone, Copy)]
pub struct Weighted {
    pub warp: f64,
    pub violation: f64,
    pub revenue: f64,
    pub cost: f64,
    pub approx_berth_violation: f64,
    pub spot: f64,
    pub offset: f64,
}

impl Fitness for Weighted {
    fn of(&self, _: &Problem, solution: &RoutingSolution) -> f64 {
        let warp = solution.warp() as f64;
        let violation = solution.violation();
        let revenue = solution.revenue();
        let cost = solution.cost();
        let berth = solution.approx_berth_violation() as f64;
        let spot = solution.spot_cost();

        (warp * self.warp
            + violation * self.violation
            + cost * self.cost
            + revenue * self.revenue
            + spot * self.spot
            + self.approx_berth_violation * berth
            + self.offset)
            .ln()
    }
}

/// An AtomicF64 based on wrapping a AtomicU64.
/// Taken from https://github.com/rust-lang/rust/issues/72353
pub struct AtomicF64 {
    storage: AtomicU64,
}

impl AtomicF64 {
    pub fn new(value: f64) -> Self {
        let as_u64 = value.to_bits();
        Self {
            storage: AtomicU64::new(as_u64),
        }
    }
    pub fn store(&self, value: f64, ordering: Ordering) {
        let as_u64 = value.to_bits();
        self.storage.store(as_u64, ordering)
    }
    pub fn load(&self, ordering: Ordering) -> f64 {
        let as_u64 = self.storage.load(ordering);
        f64::from_bits(as_u64)
    }
}

/// A version of `Weighted` with atomic weights. Meant to be used as a shared fitness objective
/// between multiple islands in an islanding GA.
pub struct AtomicWeighted {
    pub warp: AtomicF64,
    pub violation: AtomicF64,
    pub revenue: AtomicF64,
    pub cost: AtomicF64,
    pub approx_berth_violation: AtomicF64,
    pub spot: AtomicF64,
    pub offset: AtomicF64,
}

impl Fitness for AtomicWeighted {
    fn of(&self, problem: &Problem, solution: &RoutingSolution) -> f64 {
        Weighted {
            warp: self.warp.load(Ordering::Relaxed),
            violation: self.violation.load(Ordering::Relaxed),
            revenue: self.revenue.load(Ordering::Relaxed),
            cost: self.cost.load(Ordering::Relaxed),
            approx_berth_violation: self.approx_berth_violation.load(Ordering::Relaxed),
            spot: self.spot.load(Ordering::Relaxed),
            offset: self.offset.load(Ordering::Relaxed),
        }
        .of(problem, solution)
    }
}

impl<F: Fitness> Fitness for Arc<F> {
    fn of(&self, problem: &Problem, solution: &RoutingSolution) -> f64 {
        let inner: &F = &(*self);
        inner.of(problem, solution)
    }
}
