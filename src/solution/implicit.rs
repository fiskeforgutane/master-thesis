use std::cell::Cell;

use crate::problem::Problem;

use super::{explicit, InsertionError, Visit};

/// A solution that only considers visits to consumption nodes, forming a "giant tour" for each vessel.
/// We then attempt to split this by inserting factory visits at suitable places such that inventory levels are maintained.
pub struct Solution<'p> {
    /// The problem this solution belongs to
    pub problem: &'p Problem,
    /// The set of consumption visits for each vehicle. Sorted ascending by time.
    consumption_visits: Vec<Vec<Visit>>,
    /// The set of visits after inserting production visits.
    routes: Cell<Vec<Vec<Visit>>>,
}

impl<'p> Solution<'p> {
    /// Construct a new implicit solution for `problem` with the consumption visits given in `consumption_visits`
    pub fn new(
        problem: &'p Problem,
        consumption_visits: Vec<Vec<Visit>>,
    ) -> Result<Self, InsertionError> {
        // Check that none of the visits are to production nodes.
        for route in &consumption_visits {
            for visit in route {
                match problem.nodes()[visit.node].r#type() {
                    crate::problem::NodeType::Consumption => (),
                    crate::problem::NodeType::Production => {
                        return Err(InsertionError::InvalidVisit)
                    }
                }
            }
        }

        let explicit = explicit::Solution::new(problem, consumption_visits)?;
        let (_, consumption_visits) = explicit.dissolve();

        // TODO: Check that time constraints are satisfied

        Ok(Self {
            problem,
            consumption_visits,
            routes: Cell::default(),
        })
    }

    /// Invalidate the cached computation of full routes. Must be
    /// done in the event that `self.consumption_visits` is modified.
    fn invalidate_caches(&self) {
        // Note: if we want to save on allocations, we can take `self.routes`
        // and clear it, rather than replacing it with a new Vec.
        self.routes.set(Vec::new())
    }
}
