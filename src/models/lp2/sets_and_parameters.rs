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
    pub N_t: HashMap<TimeIndex, Vec<NodeIndex>>,
    /// Vessels performing a visit (or can perform a visit) at node n in time period t
    pub V_nt: HashMap<(NodeIndex, TimeIndex), Vec<VesselIndex>>,
    /// Production nodes in N_t
    pub N_tP: HashMap<TimeIndex, Vec<NodeIndex>>,
    /// Consumption nodes in N_t
    pub N_tC: HashMap<TimeIndex, Vec<NodeIndex>>,
    /// Time periods in which the node n can be visited
    pub T_n: TiVec<NodeIndex, Vec<TimeIndex>>,
}

#[allow(non_snake_case)]
impl Sets {
    pub fn new(solution: &RoutingSolution) -> Sets {
        let problem = solution.problem();

        let mut N_t = HashMap::<TimeIndex, Vec<_>>::new();
        let mut V_nt = HashMap::<(NodeIndex, TimeIndex), Vec<VesselIndex>>::new();
        let mut N_tP = HashMap::<TimeIndex, Vec<NodeIndex>>::new();
        let mut N_tC = HashMap::<TimeIndex, Vec<NodeIndex>>::new();
        let mut T_n: TiVec<NodeIndex, _> = vec![Vec::new(); problem.nodes().len()].into();
        let mut T = Vec::new();

        for (v, plan) in solution.iter().enumerate() {
            let vessel = &problem.vessels()[v];
            // An artificial "last visit" to the same node in the final time step
            let last = plan.last().map(|v| Visit {
                node: v.node,
                time: problem.timesteps(),
            });

            for (v1, v2) in plan.iter().chain(last.as_ref()).tuple_windows() {
                let n = NodeIndex::from(v1.node);
                // We must determine when we need to leave `v1.node` in order to make it to `v2.node` in time.
                // Some additional arithmetic parkour is done to avoid underflow cases (damn usizes).
                let travel_time = problem.travel_time(v1.node, v2.node, vessel);
                let departure_time = v2.time - v2.time.min(travel_time);
                // In addition, we can further restrict the active time periods by looking at the longest possible time
                // the vessel can spend at the node doing constant loading/unloading.
                let rate = problem.nodes()[v1.node].min_unloading_amount();
                let max_loading_time = (vessel.capacity() / rate).ceil() as usize;
                // let time_available = departure_time.max(v1.time) - v1.time;
                for t in v1.time..departure_time.min(v1.time + max_loading_time) {
                    let t = TimeIndex::from(t);
                    N_t.entry(t).or_default().push(n);
                    V_nt.entry((n, t)).or_default().push(VesselIndex::from(v));
                    match problem.nodes()[v1.node].r#type() {
                        NodeType::Consumption => N_tC.entry(t).or_default().push(n),
                        NodeType::Production => N_tP.entry(t).or_default().push(n),
                    }

                    T_n[n].push(t);
                    T.push(t);
                }
            }
        }

        // Sort and dedup
        macro_rules! normalize {
            ($it:expr) => {{
                $it.sort_unstable_by_key(|x| {
                    let y: usize = **x;
                    y
                });
                $it.dedup();
            }};
        }

        N_t.values_mut().for_each(|xs| normalize!(xs));
        V_nt.values_mut().for_each(|xs| normalize!(xs));
        N_tP.values_mut().for_each(|xs| normalize!(xs));
        N_tC.values_mut().for_each(|xs| normalize!(xs));
        T_n.iter_mut().for_each(|xs| normalize!(xs));

        let T = (0..problem.timesteps()).map(TimeIndex::from).collect();
        let P = (0..problem.products()).map(ProductIndex::from).collect();
        let V = (0..problem.vessels().len()).map(From::from).collect();
        let N = (0..problem.nodes().len()).map(NodeIndex::from).collect();
        let mut N_P = Vec::new();
        let mut N_C = Vec::new();

        for (n, node) in problem.nodes().iter().enumerate() {
            match node.r#type() {
                NodeType::Consumption => N_C.push(NodeIndex::from(n)),
                NodeType::Production => N_P.push(NodeIndex::from(n)),
            }
        }

        Sets {
            T,
            N,
            N_P,
            N_C,
            P,
            V,
            N_t,
            V_nt,
            N_tP,
            N_tC,
            T_n,
        }
    }
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

macro_rules! map {
    ($set:expr, $f:expr) => {
        $set.iter().map($f).collect::<TiVec<_, _>>()
    };
}

#[allow(non_snake_case)]
impl<'a> Parameters<'a> {
    pub fn new(solution: &'a RoutingSolution) -> Self {
        let problem = solution.problem();
        let p = problem.products();
        let sets = Sets::new(solution);

        let Q = map!(problem.vessels(), Vessel::capacity);
        let L_0 = map!(problem.vessels(), |vessel| {
            (0..p).map(|i| vessel.initial_inventory()[i]).collect()
        });
        let S_0 = sets
            .T_n
            .iter()
            .enumerate()
            .map(|(n, times)| {
                let time = times.first().cloned().unwrap_or(TimeIndex::from(0));
                (0..p)
                    .map(|i| problem.nodes()[n].inventory_without_deliveries(i)[*time])
                    .collect()
            })
            .collect();
        let S_min = sets
            .T_n
            .iter_enumerated()
            .flat_map(|(n, times)| {
                times
                    .iter()
                    .flat_map(move |&t| (0..p).map(move |p| ((n, ProductIndex::from(p), t), 0.0)))
            })
            .collect();
        let S_max = sets
            .T_n
            .iter_enumerated()
            .flat_map(|(n, times)| {
                let idx: usize = n.into();
                let node = &problem.nodes()[idx];
                times.iter().flat_map(move |&t| {
                    (0..p).map(move |p| ((n, ProductIndex::from(p), t), node.capacity()[p]))
                })
            })
            .collect();

        let I = map!(problem.nodes(), |node| match node.r#type() {
            NodeType::Consumption => -1.0,
            NodeType::Production => 1.0,
        });

        let N_t = &sets.N_t;
        let V_nt = &sets.V_nt;
        let R = sets
            .T
            .iter()
            .flat_map(|&t| {
                N_t[&t].iter().flat_map(move |&n| {
                    V_nt[&(n, t)].iter().map(move |&v| {
                        let idx: usize = n.into();
                        ((n, v, t), problem.nodes()[idx].max_loading_amount())
                    })
                })
            })
            .collect();

        Parameters {
            sets,
            problem,
            Q,
            L_0,
            S_0,
            S_min,
            S_max,
            I,
            R,
        }
    }

    /// Quantity of product p produced/consumed at the node in the **exclusive** range [i,j)
    pub fn D(
        &self,
        n: NodeIndex,
        i: TimeIndex,
        j: TimeIndex,
        p: ProductIndex,
    ) -> Result<f64, Error> {
        self.check_time_periods(i, j)?;
        let node = &self.problem.nodes()[*n];
        Ok(f64::abs(node.inventory_change(*i, *j - 1, *p)))
    }

    /// total consumption of node n of product p
    pub fn D_tot(&self, n: NodeIndex, p: ProductIndex) -> Result<f64, Error> {
        self.D(n, 0.into(), self.problem.timesteps().into(), p)
    }

    /// The remaining consumption of a node n of product p from the given time period
    pub fn D_rem(&self, n: NodeIndex, p: ProductIndex, t: TimeIndex) -> Result<f64, Error> {
        self.D(n, t, self.problem.timesteps().into(), p)
    }

    /// get the lower limit of product p at the given node in the last time period
    pub fn S_min(&self, n: NodeIndex, p: ProductIndex) -> f64 {
        0.0
    }

    /// get the upper limit of product p at the given node in the last time period
    pub fn S_max(&self, n: NodeIndex, p: ProductIndex) -> f64 {
        let node = &self.problem.nodes()[*n];
        node.capacity()[*p]
    }

    pub fn check_time_periods(&self, i: TimeIndex, j: TimeIndex) -> Result<(), Error> {
        if i >= j {
            return Err(Error::WrongOrderException(format!(
                "{:?} is larger than {:?} - given in wrong order",
                i, j
            )));
        }
        Ok(())
    }
}
