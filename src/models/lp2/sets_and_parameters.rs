use std::collections::HashMap;

use derive_more::{Deref, From, Into};
use itertools::Itertools;
use log::trace;
use typed_index_collections::TiVec;

use crate::{
    models::lp::sets_and_parameters::{NodeIndex, ProductIndex, TimeIndex, VesselIndex},
    problem::{NodeType, Problem, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

#[derive(Debug)]
pub enum Error {
    WrongOrderException(String),
    DifferentNodesException(String),
    VisitDoesNotExistException(String),
}

#[allow(non_snake_case)]
pub struct Sets {
    /// Time periods in which at least one visit occurs
    pub T: Vec<TimeIndex>,
    /// Set of nodes
    pub N: Vec<NodeIndex>,
    /// Set of production nodes
    pub N_P: Vec<NodeIndex>,
    /// Set of consumption nodes
    pub N_C: Vec<NodeIndex>,
    /// Set of products
    pub P: Vec<ProductIndex>,
    /// Set of vessels
    pub V: Vec<VesselIndex>,
    /// Nodes being visited (or a visit can be active) in time period t
    pub N_t: HashMap<TimeIndex, Vec<TimeIndex>>,
    /// Vessels performing a visit (or can perform a visit) at node n in time period t
    pub V_nt: HashMap<(NodeIndex, TimeIndex), Vec<VesselIndex>>,
    /// Production nodes in N_t
    pub N_tP: HashMap<TimeIndex, Vec<TimeIndex>>,
    /// Consumption nodes in N_t
    pub N_tC: HashMap<TimeIndex, Vec<TimeIndex>>,
    /// Time periods in which the node n can be visited
    pub T_n: Vec<Vec<TimeIndex>>,
}

#[allow(non_snake_case)]
pub struct Parameters<'a> {
    /// The sets used to create these parameters
    pub sets: Sets,
    /// The problem these parameters "belong" to.
    pub problem: &'a Problem,

    /// Capacity of each vessel v in V
    pub Q: TiVec<VesselIndex, f64>,
    /// initial load of vessel v of product p
    pub L_0: TiVec<VesselIndex, TiVec<ProductIndex, f64>>,
    /// Initial inventory at node n of product p
    pub S_0: TiVec<NodeIndex, TiVec<ProductIndex, f64>>,
    /// Lower limit at the beginning of the given time period at the node and product (n,p,t)
    pub S_min: HashMap<(NodeIndex, ProductIndex, TimeIndex), f64>,
    /// Upper limit at the beginning of the given time period at the node and product (n,p,t)
    pub S_max: HashMap<(NodeIndex, ProductIndex, TimeIndex), f64>,
    /// Lower limit at the given node of product p
    /// Kind of the node n, +1 for production, -1 for consumption
    pub I: TiVec<NodeIndex, f64>,
    /// The loading/unloading rate per time period at visit j
    pub R: HashMap<(NodeIndex, VesselIndex, TimeIndex), f64>,
}

impl<'a> Parameters<'a> {
    pub fn D(n: NodeIndex, i: TimeIndex, j: TimeIndex, p: ProductIndex) {}

    pub fn check_time_periods(&self, i: TimeIndex, j: TimeIndex) -> Result<(), Error> {
        if i > j {
            return Err(Error::WrongOrderException(format!(
                "{:?} is larger than {:?} - given in wrong order",
                i, j
            )));
        }
        Ok(())
    }
}
