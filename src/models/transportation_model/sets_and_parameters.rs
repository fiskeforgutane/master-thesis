use crate::{problem::Problem, quants::Quantities};
use log::trace;
use std::collections::HashMap;

type NodeIndex = usize;
type VesselIndex = usize;
type ProductIndex = usize;
type OrderIndex = usize;
type CompartmentIndex = usize;

/// sets for the transportation model
#[derive(Debug)]
#[allow(non_snake_case)]
pub struct Sets {
    /// Set of nodes
    pub N: Vec<NodeIndex>,
    /// Set of vessels
    pub V: Vec<VesselIndex>,
    /// Set of products
    pub P: Vec<ProductIndex>,
    /// Compartments of vessels
    pub O: Vec<Vec<CompartmentIndex>>,
    /// set of orders to serve each node. Each node has orders ranging from 0..n
    pub H: Vec<OrderIndex>,
}

/// parameters for the transportation model
#[allow(non_snake_case)]
pub struct Parameters {
    /// 1 if node i is a producer, -1 if node i is a consumer
    pub J: Vec<isize>,
    /// Capacity of each compartment of every vessel
    //pub Capacity: Vec<Vec<f64>>,
    /// lower limit on the quantity of product *p* that may be transported for node *i* to node *j* in one shipment
    pub lower_Q: Vec<Vec<Vec<f64>>>,
    /// upper limit on the quantity of product *p* that may be transported for node *i* to node *j* in one shipment
    pub upper_Q: Vec<Vec<Vec<f64>>>,
    /// the quantity either delivered or picked up of product type p at node i
    pub Q: Vec<Vec<f64>>,
    /// minimum transportation cost from node i to j (C_ij = min(C_ijv, i,j in N, for v in Vessels))
    pub C: Vec<Vec<f64>>,

    /// fixed port costs
    pub C_fixed: Vec<f64>,
    /// pertubations of the objective coefficients for cargoes from node i to node j
    pub epsilon: Vec<Vec<f64>>,
    /// pertubations of the objective coefficients for cargoes from node i to node j
    pub delta: Vec<Vec<f64>>,
}

#[allow(non_snake_case)]
impl Sets {
    pub fn new(problem: &Problem) -> Sets {
        let O = problem
            .vessels()
            .iter()
            .map(|v| (0..v.compartments().len()).collect())
            .collect();

        let H = Sets::get_h(problem);
        Sets {
            N: problem.nodes().iter().map(|n| n.index()).collect(),
            V: problem.vessels().iter().map(|v| v.index()).collect(),
            P: (0..problem.products()).collect(),
            O,
            H: H,
        }
    }

    fn get_h(problem: &Problem) -> Vec<usize> {
        trace!("\n--------------Getting H-------------");
        let mut res: HashMap<NodeIndex, usize> = HashMap::new();
        let slowest_vessel = problem
            .vessels()
            .iter()
            .min_by(|a, b| a.speed().partial_cmp(&b.speed()).unwrap())
            .unwrap();

        trace!("Speed of slowest vessel: {:?}", slowest_vessel.speed());
        for n in problem.consumption_nodes() {
            let node = &problem.nodes()[*n];
            trace!("\nnode: {:?}", node.index());
            let closest_prod = problem.closest_production_node(node);
            trace!("closest prod: {:?}", closest_prod.index());
            // time to load - sail - unload - sail back to depot
            let round_trip_time =
                2 + 2 * problem.travel_time(closest_prod.index(), node.index(), slowest_vessel);
            trace!("rtt: {:?}", round_trip_time);
            trace!("num timesteps {:?}", problem.timesteps());
            let num_trips = (problem.timesteps() as f64 / round_trip_time as f64).floor() as usize;
            trace!("num_trips: {:?}", num_trips);
            res.insert(node.index(), num_trips);
        }
        trace!("res: {:?}", res);
        let sum = res.iter().map(|(_, v)| v).sum();
        trace!("sum: {:?}", sum);
        let H = (0..sum).collect();
        H
    }
}

#[allow(non_snake_case)]
impl Parameters {
    pub fn new(problem: &Problem, sets: &Sets) -> Parameters {
        let J = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                crate::problem::NodeType::Consumption => -1,
                crate::problem::NodeType::Production => 1,
            })
            .collect();
        let smallest_vessel = problem
            .vessels()
            .iter()
            .map(|v| v.compartments().iter().map(|c| c.0).sum())
            .reduce(f64::min)
            .unwrap();
        let lower_Q = sets
            .P
            .iter()
            .map(|_| {
                problem
                    .nodes()
                    .iter()
                    .map(|i| {
                        problem
                            .nodes()
                            .iter()
                            .map(|j| f64::max(i.min_unloading_amount(), j.min_unloading_amount()))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let upper_Q = sets
            .P
            .iter()
            .map(|p| {
                problem
                    .nodes()
                    .iter()
                    .map(|i| {
                        problem
                            .nodes()
                            .iter()
                            .map(|j| {
                                f64::min(
                                    smallest_vessel,
                                    f64::min(i.capacity()[*p], j.capacity()[*p]),
                                )
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let Q = (0..problem.products())
            .map(|p| {
                let quants = Quantities::quantities(&problem, p);
                problem.nodes().iter().map(|n| quants[&n.index()]).collect()
            })
            .collect();

        let C = {
            let cheapest_vessel = problem
                .vessels()
                .iter()
                .min_by(|a, b| a.time_unit_cost().partial_cmp(&b.time_unit_cost()).unwrap())
                .unwrap();
            let mut res = Vec::new();
            for i in problem.nodes() {
                let mut a = Vec::new();
                for j in problem.nodes() {
                    let cost =
                        cheapest_vessel.time_unit_cost() * problem.distance(i.index(), j.index());
                    a.push(cost)
                }
                res.push(a);
            }
            res
        };

        let C_fixed = problem.nodes().iter().map(|n| n.port_fee()).collect();

        let pertubation = problem
            .nodes()
            .iter()
            .map(|_| problem.nodes().iter().map(|_| 1.0).collect())
            .collect();

        let pertubation2 = problem
            .nodes()
            .iter()
            .map(|_| problem.nodes().iter().map(|_| 1.0).collect())
            .collect();

        Parameters {
            J,
            lower_Q,
            upper_Q,
            Q,
            C,
            C_fixed,
            epsilon: pertubation,
            delta: pertubation2,
        }
    }
}
