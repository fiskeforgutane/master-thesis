use std::{ops::Range, sync::Arc};

use crate::problem::{
    FixedInventory, Node, NodeIndex, Problem, ProblemConstructionError, TimeIndex, Vessel,
};

pub struct RollingHorizon {
    problem: Arc<Problem>,
}

impl RollingHorizon {
    pub fn new(problem: Arc<Problem>) -> Self {
        RollingHorizon { problem }
    }

    pub fn create_subproblem(
        &self,
        initial_loads: Vec<FixedInventory>,
        origins: Vec<NodeIndex>,
        available_from: Vec<TimeIndex>,
        initial_inventory: Vec<FixedInventory>,
        period: Range<TimeIndex>,
    ) -> Result<Problem, ProblemConstructionError> {
        let vessels = self
            .problem
            .vessels()
            .into_iter()
            .map(|v| {
                Vessel::new(
                    v.compartments().to_vec(),
                    v.speed(),
                    v.travel_unit_cost(),
                    v.empty_travel_unit_cost(),
                    v.time_unit_cost(),
                    available_from[v.index()],
                    (initial_loads[v.index()]).clone(),
                    origins[v.index()],
                    v.class().to_string(),
                    v.index(),
                )
            })
            .collect();

        let nodes = self
            .problem
            .nodes()
            .into_iter()
            .map(|n| {
                Node::new(
                    n.name().to_string(),
                    n.r#type(),
                    n.index(),
                    n.port_capacity().to_vec(),
                    n.min_unloading_amount(),
                    n.max_loading_amount(),
                    n.port_fee(),
                    n.capacity().clone(),
                    n.inventory_changes()[period.start..period.end].to_vec(),
                    n.revenue(),
                    (initial_inventory[n.index()]).clone(),
                    n.spot_market_limit_per_time(),
                    n.spot_market_limit(),
                    n.coordinates(),
                )
            })
            .collect();

        Problem::new(
            vessels,
            nodes,
            period.end - period.start,
            self.problem.products(),
            self.problem.distances().clone(),
        )
    }
}
