use crate::{problem::Problem, solution::routing::RoutingSolution};

use super::Fitness;

pub struct Weighted {
    warp: f64,
    violation: f64,
    income: f64,
}

impl Fitness for Weighted {
    fn of(&self, _: &Problem, solution: &RoutingSolution) -> f64 {
        let warp = solution.warp() as f64;
        let violation = solution.violation();
        let income = 0.0;

        warp * self.warp + violation * self.violation + income * self.income
    }
}
