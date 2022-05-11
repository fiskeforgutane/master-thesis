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
