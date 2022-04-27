use crate::{problem::Problem, quants::Quantities};

use itertools::iproduct;

type PortIndex = usize;
type VesselIndex = usize;
type ProductIndex = usize;
type CompartmentIndex = usize;
type TimeIndex = usize;

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct Sets {
    // Set of ports
    pub I: Vec<PortIndex>,
    // Set of production ports
    pub Ip: Vec<PortIndex>,
    // Set of consumption ports
    pub Ic: Vec<PortIndex>,
    // Set of vessels
    pub V: Vec<VesselIndex>,
    // Set of time periods
    pub T: Vec<TimeIndex>,
    // Set of products
    pub P: Vec<ProductIndex>,
    // Set of compartments on the vessels
    pub S: Vec<Vec<CompartmentIndex>>,
    // Set of nodes (time-port pairs)
    pub N: Vec<(PortIndex, TimeIndex)>,
    // Set of nodes including source and sink
    pub Nst: Vec<(PortIndex, TimeIndex)>,
    // Set of arcs associated with each vessel
    pub A: Vec<Vec<((PortIndex, TimeIndex), (PortIndex, TimeIndex))>>,
    // Set of all outgoing arcs associated with a node
    pub Fs: Vec<Vec<((PortIndex, TimeIndex), (PortIndex, TimeIndex))>>,
    // Set of all incoming arcs associated with a node
    pub Rs: Vec<Vec<((PortIndex, TimeIndex), (PortIndex, TimeIndex))>>,
}

#[allow(non_snake_case)]
pub struct Parameters {
    // Capacity of silo c on vessel v
    pub silo_capacity: Vec<Vec<f64>>,
    // Initial inventory of product p in silo c in vessel v
    pub initial_silo_inventory: Vec<Vec<Vec<f64>>>,
    // The cost of traversing an arc with a particular vessel
    pub travel_cost: Vec<Vec<f64>>,
    // The unit cost of buying a product from the spot marked in a time period
    pub spot_market_cost: Vec<Vec<f64>>,
    // Berth capacity of a port
    pub berth_capacity: Vec<Vec<usize>>,
    // Total capacity of a product at a port
    pub port_capacity: Vec<Vec<f64>>,
    // The lower inventory limit of a product at a port
    pub min_inventory_port: Vec<Vec<f64>>,
    // Initial inventory of a product at a port
    pub initial_port_inventory: Vec<Vec<f64>>,
    // Consumption/production rate of a product at a port in a time period
    pub consumption: Vec<Vec<Vec<f64>>>,
    // Indicator indicating if a port is a consumption or production port
    pub port_type: isize,
    // Nonnegative cost parameter used in objective for favoring early deliveries
    pub epsilon: f64,
    // The maximum that can be bought from the spot market of a product to a port
    // in a time period
    pub max_spot_period: Vec<Vec<Vec<f64>>>,
    // Total max delivery across all time periods
    pub max_sport_horizon: Vec<Vec<f64>>,
    // Minimum amount that can be loaded/unloaded to a port in a time period if such
    // action finds place
    pub min_loading_rate: Vec<Vec<f64>>,
    // Maximum amount that can be loaded/unloaded to a port in a time period if such
    // action finds place
    pub max_loading_rate: Vec<Vec<f64>>,
}

#[allow(non_snake_case)]
impl Sets {
    pub fn new(problem: &Problem) -> Sets {
        let I = problem.nodes().iter().map(|i| i.index()).collect();
        let Ic = problem.consumption_nodes().iter().map(|i| i.index()).collect();
        let Ip = problem.production_nodes().iter().map(|i| i.index()).collect();
        let V = problem.vessels().iter().map(|v| v.index()).collect();
        let T = (0..problem.timesteps()).collect();
        let P = (0..problem.products()).collect();
        let S = problem
            .vessels()
            .iter()
            .map(|v| (0..v.compartments().len()).collect())
            .collect();
        let N = iproduct!(&I, &T).collect();
    }
}