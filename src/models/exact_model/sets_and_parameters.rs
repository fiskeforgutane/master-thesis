use std::collections::{HashMap, HashSet};

use crate::problem::{Inventory, Problem, Vessel};

use itertools::iproduct;

type PortIndex = usize;
type VesselIndex = usize;
type ProductIndex = usize;
type CompartmentIndex = usize;
type TimeIndex = usize;
type ArcIndex = usize;
type NodeIndex = usize;

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct Sets {
    /// Set of ports
    pub I: Vec<PortIndex>,
    /// Set of production ports
    pub Ip: Vec<PortIndex>,
    /// Set of consumption ports
    pub Ic: Vec<PortIndex>,
    /// Set of vessels
    pub V: Vec<VesselIndex>,
    /// Set of time periods
    pub T: Vec<TimeIndex>,
    /// Set of products
    pub P: Vec<ProductIndex>,
    /// Set of compartments on the vessels
    pub S: Vec<Vec<CompartmentIndex>>,
    /// Set of nodes excluding source and sink
    pub N: Vec<NetworkNode>,
    /// Set of nodes including source and sink
    pub Nst: Vec<NetworkNode>,
    /// Set of all arcs
    pub A: Vec<Arc>,
    /// Set of all arcs associated with a particular vessel
    pub Av: Vec<Vec<ArcIndex>>,
    /// Set of all arcs associated with "normal" nodes
    pub At: Vec<ArcIndex>,
    /// Set of all outgoing arcs associated with a node
    pub Fs: Vec<Vec<Vec<ArcIndex>>>,
    /// Set of all incoming arcs associated with a node
    pub Rs: Vec<Vec<Vec<ArcIndex>>>,
}

#[allow(non_snake_case)]
pub struct Parameters {
    /// Total capacity of vessel v
    pub vessel_capacity: Vec<f64>,
    /// Capacity of the compartments on vessel v
    pub compartment_capacity: Vec<Vec<f64>>,
    /// Initial inventory of product p in vessel v
    pub initial_inventory: Vec<Vec<f64>>,
    /// The cost of traversing an arc with a particular vessel
    pub travel_cost: Vec<Vec<f64>>,
    /// The unit cost of buying a product from the spot market at node n at time t
    pub spot_market_cost: Vec<Vec<f64>>,
    /// Berth capacity of a port
    pub berth_capacity: Vec<Vec<usize>>,
    /// Total capacity of a product at a port
    pub port_capacity: Vec<Vec<f64>>,
    /// The lower inventory limit of a product at a port
    pub min_inventory_port: Vec<Vec<f64>>,
    /// Initial inventory of a product at a port
    pub initial_port_inventory: Vec<Vec<f64>>,
    /// Consumption/production rate of a product at a port in a time period
    pub consumption: Vec<Vec<Vec<f64>>>,
    /// Indicator indicating if a port is a consumption or production port
    pub port_type: Vec<isize>,
    /// Nonnegative cost parameter used in objective for favoring early deliveries
    pub epsilon: f64,
    /// The maximum that can be bought from the spot market to a port
    /// in a time period
    pub max_spot_period: Vec<Vec<f64>>,
    /// Total max delivery across all time periods
    pub max_spot_horizon: Vec<f64>,
    /// Minimum amount that can be loaded/unloaded to a port in a time period if such
    /// action finds place
    pub min_loading_rate: Vec<f64>,
    /// Maximum amount that can be loaded/unloaded to a port in a time period if such
    /// action finds place
    pub max_loading_rate: Vec<f64>,
    /// The revenue associated with a unit delivery (like in MIRPLib instances)
    pub revenue: Vec<f64>,
}

#[allow(non_snake_case)]
impl Sets {
    pub fn new(problem: &Problem) -> Sets {
        let I: Vec<usize> = problem.nodes().iter().map(|i| i.index()).collect();
        let Ip = problem.production_nodes().clone();
        let Ic = problem.consumption_nodes().clone();
        let V = problem.vessels().iter().map(|v| v.index()).collect();
        let T = (0..problem.timesteps()).collect();
        let P = (0..problem.products()).collect();
        let S = problem
            .vessels()
            .iter()
            .map(|v| (0..v.compartments().len()).collect())
            .collect();
        let N: Vec<NetworkNode> = iproduct!(0..(problem.nodes().len()), 0..problem.timesteps())
            .enumerate()
            .map(|(index, (i, t))| NetworkNode::new(i, t, index, NetworkNodeType::Normal))
            .collect();
        // Add sink and source to the set of all nodes
        let mut Nst = N.clone();
        Nst.push(NetworkNode::new(
            I.len() + 1,
            0,
            Nst.len(),
            NetworkNodeType::Source,
        ));
        Nst.push(NetworkNode::new(
            I.len() + 2,
            problem.timesteps(),
            Nst.len(),
            NetworkNodeType::Sink,
        ));
        let A = Sets::get_all_arcs(&Nst);
        let Av = Sets::get_arcs(problem, &A, &Nst);
        let At = Sets::get_travel_arcs(&A);
        let Fs = problem
            .vessels()
            .iter()
            .map(|v| {
                Nst.iter()
                    .map(|n| Sets::get_forward_star(Av.get(v.index()).unwrap(), &A, &n))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let Rs = problem
            .vessels()
            .iter()
            .map(|v| {
                Nst.iter()
                    .map(|n| Sets::get_reverse_star(Av.get(v.index()).unwrap(), &A, &n))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        Sets {
            I,
            Ip,
            Ic,
            V,
            T,
            P,
            S,
            N,
            Nst,
            A,
            Av,
            At,
            Fs,
            Rs,
        }
    }

    pub fn get_all_arcs(nodes: &Vec<NetworkNode>) -> Vec<Arc> {
        // Set holding a list enumerating all possible arcs moving forward in time in the network
        let mut all_arcs: Vec<Arc> = Vec::new();

        for n1 in nodes {
            match n1.kind() {
                // if n1 is a source node we add arcs to all other nodes
                NetworkNodeType::Source => nodes
                    .iter()
                    .filter(|n2| n2.port() != n1.port())
                    .for_each(|n2| all_arcs.push(Arc::new(n1, n2, all_arcs.len()))),
                // if n1 is a source we do not add any arcs
                NetworkNodeType::Sink => {
                    ();
                }
                // if n1 is a normal node we add arcs to all other nodes that are in the future, as well as the next node that have the same port
                NetworkNodeType::Normal => {
                    // add arcs to other nodes
                    nodes
                        .iter()
                        .filter(|n2| n2.port() != n1.port() && n2.time() > n1.time())
                        .for_each(|n2| all_arcs.push(Arc::new(n1, n2, all_arcs.len())));

                    // add the arc to the next node (iterator will be empty if this is the last node at the current port)
                    nodes
                        .iter()
                        .filter(|n2| n2.port() == n1.port() && n2.time() == n1.time() + 1)
                        .for_each(|n2| all_arcs.push(Arc::new(n1, n2, all_arcs.len())));
                }
            }
        }

        all_arcs
    }

    pub fn get_arcs(problem: &Problem, A: &Vec<Arc>, Nst: &Vec<NetworkNode>) -> Vec<Vec<ArcIndex>> {
        let arc_map = A
            .iter()
            .map(|arc| ((arc.get_from(), arc.get_to()), arc))
            .collect::<HashMap<(&NetworkNode, &NetworkNode), &Arc>>();

        let node_map: HashMap<(isize, TimeIndex), &NetworkNode> = Nst
            .iter()
            .map(|n| match n.kind() {
                NetworkNodeType::Source => ((-1, 0), n),
                NetworkNodeType::Sink => ((-1, problem.timesteps()), n),
                NetworkNodeType::Normal => ((n.port() as isize, n.time()), n),
            })
            .collect();

        // Generate the set of arcs for each of the vessels
        let vessel_arcs = problem
            .vessels()
            .iter()
            .map(|v| {
                Sets::get_vessel_arcs(
                    node_map.get(&(-1, 0)).unwrap(),
                    problem,
                    &arc_map,
                    &node_map,
                    v,
                )
            })
            .collect::<Vec<_>>();

        vessel_arcs
    }

    pub fn get_vessel_arcs(
        start: &NetworkNode,
        problem: &Problem,
        all_arcs: &HashMap<(&NetworkNode, &NetworkNode), &Arc>,
        nodes: &HashMap<(isize, usize), &NetworkNode>,
        vessel: &Vessel,
    ) -> Vec<ArcIndex> {
        let next = |network_node: &NetworkNode| {
            match network_node.kind() {
                // origin or not used arc - directly to sink
                NetworkNodeType::Source => {
                    let mut next_nodes = (vessel.available_from()..problem.timesteps())
                        .map(|t| nodes.get(&(vessel.origin() as isize, t)).unwrap())
                        .collect::<Vec<_>>();
                    next_nodes.push(nodes.get(&(-1, problem.timesteps())).unwrap());
                    Some(next_nodes)
                }
                // no arcs go out from sink
                NetworkNodeType::Sink => None,
                // waiting arc, travel arcs, and arc to sink
                NetworkNodeType::Normal => {
                    let mut next_nodes = problem
                        .nodes()
                        .iter()
                        .filter_map(|n| {
                            if n.index() == network_node.port() {
                                // waiting arc
                                nodes.get(&(n.index() as isize, network_node.time() + 1))
                            } else {
                                // travel arc
                                nodes.get(&(
                                    n.index() as isize,
                                    network_node.time()
                                        + problem.travel_time(
                                            network_node.port(),
                                            n.index(),
                                            vessel,
                                        ),
                                ))
                            }
                        })
                        .collect::<Vec<_>>();
                    // add sink node
                    next_nodes.push(nodes.get(&(-1, problem.timesteps())).unwrap());
                    Some(next_nodes)
                }
            }
        };

        let mut unexplored = Vec::new();
        unexplored.push(start);
        let mut explored: HashSet<&NetworkNode> = HashSet::new();
        let mut arcs = Vec::new();

        while !unexplored.is_empty() {
            let current = unexplored.pop().unwrap();
            let next_nodes = next(current);
            explored.insert(current);
            match next_nodes {
                Some(next_nodes) => {
                    next_nodes.into_iter().for_each(|x| {
                        arcs.push(all_arcs.get(&(current, x)).unwrap().get_index());
                        if !explored.contains(*x) {
                            unexplored.push(x);
                        }
                    });
                }
                None => continue,
            }
        }
        arcs
    }

    pub fn get_travel_arcs(all_arcs: &Vec<Arc>) -> Vec<ArcIndex> {
        all_arcs
            .iter()
            .filter(|a| match a.get_kind() {
                ArcType::TravelArc => true,
                ArcType::WaitingArc => true,
                _ => false,
            })
            .map(|a| a.get_index())
            .collect::<Vec<_>>()
    }

    pub fn get_forward_star(
        vessel_arcs: &Vec<ArcIndex>,
        all_arcs: &Vec<Arc>,
        node: &NetworkNode,
    ) -> Vec<ArcIndex> {
        let fs = vessel_arcs
            .iter()
            .filter(|a| all_arcs[**a].get_from().index() == node.index())
            .map(|a| *a)
            .collect::<Vec<_>>();

        fs
    }

    pub fn get_reverse_star(
        vessel_arcs: &Vec<ArcIndex>,
        all_arcs: &Vec<Arc>,
        node: &NetworkNode,
    ) -> Vec<ArcIndex> {
        vessel_arcs
            .iter()
            .filter(|a| *all_arcs[**a].get_to() == *node)
            .map(|a| *a)
            .collect::<Vec<_>>()
    }

    pub fn get_arc_index(
        &self,
        from_node: NodeIndex,
        from_time: TimeIndex,
        to_node: NodeIndex,
        to_time: TimeIndex,
    ) -> ArcIndex {
        match self.A.iter().find(|arc| {
            arc.get_from().index == from_node
                && arc.get_from().time() == from_time
                && arc.get_to().index() == to_node
                && arc.get_to().time() == to_time
        }) {
            Some(a) => a.get_index(),
            None => {
                panic!(
                    "Arc {}_{}_{}_{} is not in arcs",
                    from_node, from_time, to_node, to_time
                );
            }
        }
    }
}

impl Parameters {
    pub fn new(problem: &Problem, sets: &Sets) -> Parameters {
        let vessel_capacity: Vec<f64> = problem.vessels().iter().map(|v| v.capacity()).collect();
        let compartment_capacity: Vec<Vec<f64>> = problem
            .vessels()
            .iter()
            .map(|v| v.compartments().iter().map(|c| c.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let initial_inventory: Vec<Vec<f64>> = problem
            .vessels()
            .iter()
            .map(|v| {
                (0..problem.products())
                    .map(|p| v.initial_inventory()[p])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let travel_cost: Vec<Vec<f64>> = sets
            .A
            .iter()
            .map(|arc| {
                problem
                    .vessels()
                    .iter()
                    .map(|v| match arc.get_kind() {
                        ArcType::TravelArc => match problem.nodes()[arc.get_to().port()].r#type() {
                            crate::problem::NodeType::Consumption => problem.travel_cost(
                                arc.get_from().port(),
                                arc.get_to().port(),
                                v.index(),
                                &Inventory::new(&vec![1.0]).unwrap(),
                            ),
                            crate::problem::NodeType::Production => problem.travel_cost(
                                arc.get_from().port(),
                                arc.get_to().port(),
                                v.index(),
                                &Inventory::new(&vec![0.0]).unwrap(),
                            ),
                        },
                        ArcType::WaitingArc => 0.0,
                        ArcType::EnteringArc => problem.nodes()[arc.get_to().port()].port_fee(),
                        ArcType::LeavingArc => {
                            -(problem.timesteps() as f64 - arc.get_from().time() as f64) * 0.01
                        }
                        ArcType::NotUsedArc => 0.0,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let spot_market_cost: Vec<Vec<f64>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.timesteps())
                    .map(|t| {
                        n.spot_market_unit_price() * n.spot_market_discount_factor().powi(t as i32)
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        let berth_capacity: Vec<Vec<usize>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.timesteps())
                    .map(|t| n.port_capacity()[t])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let port_capacity: Vec<Vec<f64>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..n.capacity().as_inv().num_products())
                    .map(|p| n.capacity()[p])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let min_inventory_port: Vec<Vec<f64>> = problem
            .nodes()
            .iter()
            .map(|_| (0..problem.products()).map(|_| 0.0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let initial_port_inventory: Vec<Vec<f64>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.products())
                    .map(|p| n.initial_inventory()[p])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let consumption: Vec<Vec<Vec<f64>>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.timesteps())
                    .map(|t| {
                        (0..problem.products())
                            .map(|p| f64::abs(n.inventory_changes()[t][p]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let port_type: Vec<isize> = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                crate::problem::NodeType::Consumption => -1,
                crate::problem::NodeType::Production => 1,
            })
            .collect::<Vec<_>>();
        let epsilon: f64 = 0.01;
        let max_spot_period: Vec<Vec<f64>> = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.timesteps())
                    .map(|_| n.spot_market_limit_per_time())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let max_spot_horizon: Vec<f64> = problem
            .nodes()
            .iter()
            .map(|n| n.spot_market_limit())
            .collect::<Vec<_>>();
        let min_loading_rate: Vec<f64> = problem
            .nodes()
            .iter()
            .map(|n| n.min_unloading_amount())
            .collect::<Vec<_>>();
        let max_loading_rate: Vec<f64> = problem
            .nodes()
            .iter()
            .map(|n| n.max_loading_amount())
            .collect::<Vec<_>>();
        let revenue = problem
            .nodes()
            .iter()
            .map(|n| n.revenue())
            .collect::<Vec<_>>();

        Parameters {
            vessel_capacity,
            compartment_capacity,
            initial_inventory,
            travel_cost,
            spot_market_cost,
            berth_capacity,
            port_capacity,
            min_inventory_port,
            initial_port_inventory,
            consumption,
            port_type,
            epsilon,
            max_spot_period,
            max_spot_horizon,
            min_loading_rate,
            max_loading_rate,
            revenue,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash)]
pub enum NetworkNodeType {
    Source,
    Sink,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NetworkNode {
    // The port
    port: PortIndex,
    // The time
    time: TimeIndex,
    // kind
    kind: NetworkNodeType,
    // index
    index: NodeIndex,
}

impl NetworkNode {
    pub fn new(
        port: PortIndex,
        time: TimeIndex,
        index: NodeIndex,
        kind: NetworkNodeType,
    ) -> NetworkNode {
        NetworkNode {
            port,
            time,
            kind,
            index,
        }
    }

    pub fn port(&self) -> PortIndex {
        self.port
    }

    pub fn time(&self) -> TimeIndex {
        self.time
    }

    pub fn kind(&self) -> NetworkNodeType {
        self.kind
    }

    pub fn index(&self) -> NodeIndex {
        self.index
    }
}

#[derive(Debug)]
pub enum Error {
    LeavingSink(Box<str>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArcType {
    TravelArc,
    WaitingArc,
    EnteringArc,
    LeavingArc,
    NotUsedArc,
}

#[derive(Debug, Clone, Copy)]
pub struct Arc {
    // The node in which an arc begins
    from: NetworkNode,
    // The node in which an arc ends
    to: NetworkNode,
    // The kind of arc
    kind: ArcType,
    // Index of the arc
    index: ArcIndex,
}

impl Arc {
    pub fn new(from: &NetworkNode, to: &NetworkNode, index: ArcIndex) -> Arc {
        // Ensure that the timestep of the node the arc leads to is after the node it comes from
        if let NetworkNodeType::Normal = from.kind() {
            assert!(from.time() < to.time())
        }

        // When the arc is a travel arc, it is either a travel arc, a entering arc, a leaving arc or
        // an arc used by vessel not in use
        let arc_type = if from.port() != to.port() {
            match from.kind() {
                NetworkNodeType::Source => match to.kind() {
                    NetworkNodeType::Source => {
                        println!("Got a source -> source arc");
                        None
                    }
                    NetworkNodeType::Sink => Some(ArcType::NotUsedArc),
                    NetworkNodeType::Normal => Some(ArcType::EnteringArc),
                },
                NetworkNodeType::Sink => None,
                NetworkNodeType::Normal => match to.kind() {
                    NetworkNodeType::Source => {
                        println!("Got a normal -> source arc");
                        None
                    }
                    NetworkNodeType::Sink => Some(ArcType::LeavingArc),
                    NetworkNodeType::Normal => Some(ArcType::TravelArc),
                },
            }
        } else {
            // All waiting arcs should have length 1
            if (to.time() - from.time()) != 1 {
                println!("Got a waiting arc with wrong length");
                println!(
                    "From: {} To: {} Time from: {} Time to: {} From type: {:?}",
                    from.port(),
                    to.port(),
                    from.time(),
                    to.time(),
                    from.kind()
                );
                None
            } else {
                Some(ArcType::WaitingArc)
            }
        };

        assert!(arc_type.is_some());

        Arc {
            from: *from,
            to: *to,
            kind: arc_type.unwrap(),
            index: index,
        }
    }

    pub fn get_from(&self) -> &NetworkNode {
        &self.from
    }

    pub fn get_to(&self) -> &NetworkNode {
        &self.to
    }

    /// Get the arc type
    pub fn get_kind(&self) -> ArcType {
        self.kind
    }

    /// Get the index of the arc
    pub fn get_index(&self) -> ArcIndex {
        self.index
    }

    /// Retrieve the time it takes to travel the arc
    pub fn get_time(&self) -> usize {
        match self.get_kind() {
            ArcType::TravelArc => self.get_to().time() - self.get_from().time(),
            ArcType::WaitingArc => 1,
            ArcType::EnteringArc => 0,
            ArcType::LeavingArc => 0,
            ArcType::NotUsedArc => 0,
        }
    }
}
