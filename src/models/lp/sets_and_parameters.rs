use derive_more::{Deref, From, Into};
use typed_index_collections::TiVec;

use crate::{
    problem::{Problem, Vessel},
    solution::{routing::RoutingSolution, Visit},
};

#[derive(Debug)]
pub enum Error {
    WrongOrderException(String),
    DifferentNodesException(String),
    VisitDoesNotExistException(String),
}

#[derive(Deref, Debug, PartialEq, Eq, PartialOrd, From, Into, Clone, Copy, Hash)]
pub struct NodeIndex(usize);

#[derive(Deref, Debug, PartialEq, Eq, PartialOrd, From, Into, Clone, Copy, Hash)]
pub struct VesselIndex(usize);

#[derive(Deref, Debug, PartialEq, Eq, PartialOrd, From, Into, Clone, Copy, Hash)]
pub struct ProductIndex(usize);

#[derive(Deref, Debug, PartialEq, Eq, PartialOrd, From, Into, Clone, Copy, Hash)]
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

impl Sets {
    pub fn production_visits(&self) -> Vec<VisitIndex> {
        todo!()
    }

    pub fn consumption_visits(&self) -> Vec<VisitIndex> {
        todo!()
    }

    pub fn production_nodes(&self) -> Vec<NodeIndex> {
        todo!()
    }

    pub fn consumption_nodes(&self) -> Vec<NodeIndex> {
        todo!()
    }
    pub fn new(solution: &RoutingSolution) -> (Sets, Vec<(usize, Visit)>, TiVec<NodeIndex, usize>) {
        macro_rules! set {
            ($type:ident, $n:expr) => {
                (0..$n).map(|i| $type(i)).collect::<Vec<_>>()
            };
        }

        let problem = solution.problem();
        let n = problem.nodes().len();
        let v = problem.vessels().len();
        let t = problem.timesteps();
        let p = problem.products();

        let mut J = solution
            .iter()
            .enumerate()
            .flat_map(|(v, plan)| plan.iter().map(|&visit| (v, visit)))
            .collect::<Vec<_>>();
        J.sort_unstable_by_key(|(v, visit)| visit.time);

        let J_n: TiVec<NodeIndex, Vec<VisitIndex>> = vec![Vec::new(); n].into();
        let J_v: TiVec<VesselIndex, Vec<VisitIndex>> = vec![Vec::new(); v].into();
        // When a node is first visited
        let mut t_0: TiVec<NodeIndex, usize> = vec![t - 1; n].into();

        for (j, &(v, visit)) in J.iter().enumerate() {
            let (v, j, n) = (VesselIndex(v), VisitIndex(j), NodeIndex(n));
            J_v[v].append(j);
            J_n[n].append(j);
            t_0[n] = t_0[n].min(visit.time);
        }

        (
            Sets {
                P: set!(ProductIndex, p),
                V: set!(VesselIndex, v),
                N: set!(NodeIndex, n),
                J: set!(VisitIndex, J.len()),
                J_n,
                J_v,
            },
            J,
            t_0,
        )
    }
}

#[allow(non_snake_case)]
pub struct Parameters<'a> {
    /// The sets used to create these parameters
    sets: Sets,
    /// The problem these parameters "belong" to.
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
    /// Lower limit at the given node of product p
    pub S_min_n: TiVec<NodeIndex, TiVec<ProductIndex, f64>>,
    /// Upper limit at the given node of product p
    pub S_max_n: TiVec<NodeIndex, TiVec<ProductIndex, f64>>,
    /// Kind of the node associated with visit j, +1 for production, -1 for consumption
    pub I: TiVec<VisitIndex, usize>,
    /// Kind of the node, +1 for production, -1 for consumption
    pub K: TiVec<NodeIndex, usize>,
    /// The number of time periods that the vessel can spend on the given vessel v
    pub A: TiVec<VisitIndex, f64>,
    /// The loading/unloading rate per time period at visit j
    pub R: TiVec<VisitIndex, f64>,
    /// Arrival time of visit j
    pub T: TiVec<VisitIndex, usize>,
}

impl<'a> Parameters<'a> {
    pub fn new(solution: &RoutingSolution) -> Self {
        let (sets, J, t0) = Sets::new(solution);
        let problem = solution.problem();
        let p = problem.products();
        let n = problem.nodes().len();

        let N_j = J.iter().map(|(_, visit)| NodeIndex(visit.node)).collect();
        let V_j = J.iter().map(|(v, _)| VesselIndex(*v)).collect();

        let q = |vessel: &Vessel| vessel.compartments().iter().map(|c| c.0).sum();
        let Q = problem.vessels().iter().map(q).sum();

        let l = |vessel: &Vessel| (0..p).map(|p| vessel.initial_inventory()[p]).collect();
        let L_0: TiVec<VesselIndex, TiVec<ProductIndex, _>> =
            problem.vessels().iter().map(l).collect();

        let S_0: TiVec<NodeIndex, TiVec<ProductIndex, f64>> = (0..n)
            .map(|i| {
                let t = t0[NodeIndex(i)];
                let node = &problem.nodes()[i];

                (0..p)
                    .map(|p| node.inventory_without_deliveries(p)[t])
                    .collect()
            })
            .collect();

        let S_min = vec![vec![0.0; p].into(); J.len()].into();
        let S_max = J
            .iter()
            .map(|(v, visit)| {
                let node = &problem.nodes()[visit.node];
                let capacity = node.capacity();
                (0..p).map(|i| capacity[i]).collect()
            })
            .collect();

        Parameters {
            problem,
            N_j,
            V_j,
            Q,
            L_0,
            S_0,
            S_min,
            S_max,
            I: todo!(),
            K: todo!(),
            A: todo!(),
            R: todo!(),
            T: todo!(),
            sets,
            S_min_n: todo!(),
            S_max_n: todo!(),
        }
    }
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
    pub fn kind(&self, n: NodeIndex) -> f64 {
        match &self.problem.nodes()[*n].r#type() {
            crate::problem::NodeType::Consumption => -1.0,
            crate::problem::NodeType::Production => 1.0,
        }
    }

    /// Returns the kind of the node associated with the given visit
    pub fn v_kind(&self, visit: VisitIndex) -> f64 {
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

    /// returns the remaining consumption/production in the node associated with the given visit and product
    pub fn remaining(&self, visit: VisitIndex, p: ProductIndex) -> f64 {
        let arrival = self.T[visit];
        let node = self.problem.nodes()[*self.N_j[visit]];
        node.inventory_change(arrival, self.problem.timesteps() - 1, *p)
    }
}
