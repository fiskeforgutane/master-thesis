use crate::{problem::Problem, solution::routing::RoutingSolution};

use super::Fitness;

pub struct Weighted {
    warp: f64,
    violation: f64,
    revenue: f64,
    cost: f64,
}

impl Fitness for Weighted {
    fn of(&self, _: &Problem, solution: &RoutingSolution) -> f64 {
        let warp = solution.warp() as f64;
        let violation = solution.violation();
        let revenue = solution.revenue();
        let cost = solution.cost();

        warp * self.warp + violation * self.violation + cost * self.cost + revenue * self.revenue
    }
}
