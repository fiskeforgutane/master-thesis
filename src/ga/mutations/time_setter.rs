use log::trace;

use crate::{
    ga::Mutation, models::quantity_cont::QuantityLpCont, problem::Problem,
    solution::routing::RoutingSolution,
};

/// Solves a linear program to decide quantities and arrival times at each node
pub struct TimeSetter {
    quants_lp: QuantityLpCont,
}

impl TimeSetter {
    /// Create a TimeSetter mutation
    ///
    /// ## Arguments
    ///
    /// * `delay` - The mandatory delay that is added between visits for a vessel. A nonzero value will hopefully make the output from the continuous model fit a discrete time representation better.
    pub fn new(delay: f64) -> grb::Result<TimeSetter> {
        trace!("Creating time setter mutation");
        let quants_lp = QuantityLpCont::new(delay)?;
        Ok(TimeSetter { quants_lp })
    }
}

impl Mutation for TimeSetter {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution) {
        // solve the lp and retrieve the new time periods
        trace!("Applying TimeSetter to {:?}", solution);

        let new_times = self.quants_lp.get_visit_times(&solution);
        trace!("new times: {:?}", new_times);

        match new_times {
            Ok(times) => {
                let mutator = &mut solution.mutate();
                for vessel_idx in 0..times.len() {
                    let plan_mut = &mut mutator[vessel_idx].mutate();
                    for visit_idx in 0..times[vessel_idx].len() {
                        let new_time = times[vessel_idx][visit_idx];
                        let mut visit = plan_mut[visit_idx];
                        // change arrival time
                        visit.time = new_time;
                    }
                }
            }
            Err(_) => return,
        }
    }
}
