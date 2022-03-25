use derive_more::{Deref, From, Into};
use typed_index_collections::TiVec;

use crate::problem::Problem;

pub enum Error {
    WrongOrderException(String),
    DifferentNodesException(String),
    VisitDoesNotExistException(String),
}

#[derive(Deref, Debug, PartialEq, PartialOrd, From, Into, Clone, Copy)]
pub struct NodeIndex(usize);

#[derive(Deref, Debug, PartialEq, PartialOrd, From, Into, Clone, Copy)]
pub struct VesselIndex(usize);

#[derive(Deref, Debug, PartialEq, PartialOrd, From, Into, Clone, Copy)]
pub struct ProductIndex(usize);

#[derive(Deref, Debug, PartialEq, PartialOrd, From, Into, Clone, Copy)]
pub struct VisitIndex(usize);

#[allow(non_snake_case)]
pub struct Sets {
    /// Set of products
    pub P: Vec<ProductIndex>,
    /// Set of vessels
    pub V: Vec<VesselIndex>,
    /// Set of nodes
    pub N: Vec<NodeIndex>,
    /// Set of visits, sorted on the arrival time
    pub J: Vec<VisitIndex>,
    /// Set of visits performed by vessel v, sorted on arrival time
    pub J_n: TiVec<NodeIndex, Vec<VisitIndex>>,
    /// Set of visitis at node n, sorted on arrival time
    pub J_v: TiVec<VesselIndex, Vec<VisitIndex>>,
}

#[allow(non_snake_case)]
pub struct Parameters<'a> {
    /// The sets used to create these parameters
    sets: &'a Sets,
    problem: &'a Problem,
    /// Node of visit j in J
    pub N_j: TiVec<VisitIndex, NodeIndex>,
    /// The vessel performing the given visit j
    pub V_j: TiVec<VisitIndex, VesselIndex>,
    /// Capacity of each vessel v in V
    pub Q: TiVec<VesselIndex, f64>,
    /// initial load of vessel v of product p
    pub L_0: TiVec<VesselIndex, TiVec<ProductIndex, f64>>,
    /// Initial inventory at node n of product p
    pub S_0: TiVec<NodeIndex, TiVec<ProductIndex, f64>>,
    /// Lower limit at the arrival time of j at node N_j[j] of product p indexed (j,p)
    pub S_min: TiVec<VisitIndex, TiVec<ProductIndex, f64>>,
    /// Upper limit at the arrival time of j at node N_j[j] of product p indexed (j,p)
    pub S_max: TiVec<VisitIndex, TiVec<ProductIndex, f64>>,
    /// Kind of the node associated with visit j, +1 for production, -1 for consumption
    pub I: TiVec<VisitIndex, usize>,
    /// Kind of the node, +1 for production, -1 for consumption
    pub K: TiVec<NodeIndex, usize>,
    /// The number of time periods that the vessel can spend on the given vessel v
    pub A: TiVec<VesselIndex, usize>,
    /// The loading/unloading rate per time period at visit j
    pub R: TiVec<VisitIndex, f64>,
    /// Arrival time of visit j
    pub T: TiVec<VisitIndex, usize>,
}

#[allow(non_snake_case)]
impl<'a> Parameters<'a> {
    /// Quantity of product p produced/consumed at the node associated with the two given vistits in the **exclusive** range [arrival i - arrival j)
    /// ## Note
    /// The given visits must have the same associated node
    /// indexed (i,j,p)
    pub fn D(&self, i: VisitIndex, j: VisitIndex, p: ProductIndex) -> Result<f64, Error> {
        // check that i and j have the same node
        self.check_node_visits(i, j)?;
        // arrival of visit i and j
        let (t_i, t_j) = (self.T[i], self.T[j]);
        let n_idx = self.N_j[i];
        let node = &self.problem.nodes()[*n_idx];
        Ok(f64::abs(node.inventory_change(t_i, t_j - 1, *p)))
    }

    /// Total consumption/production during the planning period for the given node n and product p
    pub fn D_tot(&self, node: NodeIndex, p: ProductIndex) -> f64 {
        let node = &self.problem.nodes()[*node];
        f64::abs(node.inventory_change(0, self.problem.timesteps() - 1, *p))
    }

    /// Return (+1) for production node (-1) for consumption node
    pub fn kind(&self, n: NodeIndex) -> isize {
        match &self.problem.nodes()[*n].r#type() {
            crate::problem::NodeType::Consumption => -1,
            crate::problem::NodeType::Production => 1,
        }
    }

    /// Returns the kind of the node associated with the given visit
    pub fn v_kind(&self, visit: VisitIndex) -> isize {
        // get the node index of the node associated with the visit
        let n_idx = self.N_j[visit];
        self.kind(n_idx)
    }

    /// checks that the visit actually exists
    pub fn check_exists(&self, visit: VisitIndex) -> Result<(), Error> {
        if *visit < self.sets.J.len() {
            return Ok(());
        }
        Err(Error::VisitDoesNotExistException(format!(
            "{:?} does not exist. Only have visits from {:?} to {:?}",
            visit,
            0,
            self.sets.J.len()
        )))
    }

    /// Checks that visit `i` is before or the same time period as visit `j`
    /// Checks that visit `i` and `j` have the same node
    pub fn check_node_visits(&self, i: VisitIndex, j: VisitIndex) -> Result<(), Error> {
        self.check_exists(i)?;
        self.check_exists(j)?;
        let i_node = self.N_j[i];
        let j_node = self.N_j[j];

        // check arrival time
        if self.T[i] > self.T[j] {
            return Err(Error::WrongOrderException(format!(
                "{:?} is visited after {:?} - it should be opposite",
                i, j
            )));
        }

        if i_node != j_node {
            return Err(Error::DifferentNodesException(format!(
                "{:?} and {:?} do not have the same neigbor. Had {:?} and {:?}, respectively",
                i, j, i_node, j_node
            )));
        }
        Ok(())
    }
}
