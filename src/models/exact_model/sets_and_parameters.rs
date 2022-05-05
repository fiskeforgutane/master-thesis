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
    /// Set of all outgoing arcs associated with a node
    pub Fs: Vec<Vec<ArcIndex>>,
    /// Set of all incoming arcs associated with a node
    pub Rs: Vec<Vec<ArcIndex>>,
}

#[allow(non_snake_case)]
pub struct Parameters {
    /// Capacity of vessel v
    pub vessel_capacity: Vec<f64>,
    /// Initial inventory of product p in vessel v
    pub initial_inventory: Vec<Vec<f64>>,
    /// The cost of traversing an arc with a particular vessel
    pub travel_cost: Vec<Vec<Vec<f64>>>,
    /// The unit cost of buying a product from the spot marked in a time period
    pub spot_market_cost: Vec<f64>,
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
        let Ip = problem
            .production_nodes()
            .iter()
            .map(|i| i.index())
            .collect();
        let Ic = problem
            .consumption_nodes()
            .iter()
            .map(|i| i.index())
            .collect();
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
        let Av = Sets::get_arcs(problem, &A);
        let Fs = iproduct!(problem.vessels(), &Nst)
            .map(|(v, n)| Sets::get_forward_star(Av.get(v.index()).unwrap(), &A, &n))
            .collect::<Vec<_>>();
        let Rs = iproduct!(problem.vessels(), &Nst)
            .map(|(v, n)| Sets::get_reverse_star(Av.get(v.index()).unwrap(), &A, &n))
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
            Fs,
            Rs,
        }
    }

    pub fn get_all_arcs(nodes: &Vec<NetworkNode>) -> Vec<Arc> {
        // Set holding a list enumerating all possible arcs moving forward in time in the network
        let mut all_arcs: Vec<Arc> = Vec::new();

        for n1 in nodes {
            // add arcs to other nodes
            nodes
                .iter()
                .filter(|n2| *n2 != n1 && n2.time() > n1.time())
                .for_each(|n2| all_arcs.push(Arc::new(n1, n2, all_arcs.len())));

            // add the arc to the next node (iterator will be empty if this is the last node at the current port)
            nodes
                .iter()
                .filter(|n2| *n2 == n1 && n2.time() == n1.time() + 1)
                .for_each(|n2| all_arcs.push(Arc::new(n1, n2, all_arcs.len())));

            //
        }

        iproduct!(nodes, nodes).for_each(|(n_1, n_2)| {
            if n_1.time() < n_2.time() {
                if n_1.port() == n_2.port() {
                    if (n_1.time() + 1) == n_2.time() {
                        all_arcs.push(Arc::new(n_1, n_2, all_arcs.len()));
                    }
                } else {
                    all_arcs.push(Arc::new(n_1, n_2, all_arcs.len()));
                }
            }
            // As the source node has time = 0 and the sink node has time = n timesteps, we want to accept these as well
            else {
                if ((n_1.kind() == NetworkNodeType::Source)
                    && (n_2.kind() != NetworkNodeType::Source))
                    || ((n_2.kind() == NetworkNodeType::Sink)
                        && (n_1.kind() != NetworkNodeType::Sink))
                {
                    all_arcs.push(Arc::new(n_1, n_2, all_arcs.len()));
                }
            }
        });

        all_arcs
    }

    pub fn get_arcs(problem: &Problem, A: &Vec<Arc>) -> Vec<Vec<ArcIndex>> {
        // Generate the set of arcs for each of the vessels
        let vessel_arcs = problem
            .vessels()
            .iter()
            .map(|v| Sets::get_vessel_arcs(problem, A, v))
            .collect::<Vec<_>>();

        vessel_arcs
    }

    pub fn get_vessel_arcs(
        problem: &Problem,
        all_arcs: &Vec<Arc>,
        vessel: &Vessel,
    ) -> Vec<ArcIndex> {
        let vessel_arcs: Vec<ArcIndex> = all_arcs
            .into_iter()
            .filter(|a| {
                let available_from = vessel.available_from();
                let travel_time: usize = match a.get_kind() {
                    ArcType::TravelArc => {
                        problem.travel_time(a.get_from().port(), a.get_to().port(), vessel)
                    }
                    ArcType::WaitingArc => 1,
                    _ => a.get_time(),
                };
                let origin = vessel.origin();

                (a.get_from().time() < available_from)
                    && (a.get_to().time() < available_from)
                    && ((a.get_kind() == ArcType::TravelArc)
                        && ((travel_time != a.get_time()) || (travel_time == usize::MAX)))
                    && ((a.get_kind() == ArcType::EnteringArc) && (a.get_from().index() != origin))
            })
            .map(|a| a.get_index())
            .collect::<Vec<_>>();

        vessel_arcs
    }

    pub fn get_forward_star(
        vessel_arcs: &Vec<ArcIndex>,
        all_arcs: &Vec<Arc>,
        node: &NetworkNode,
    ) -> Vec<ArcIndex> {
        vessel_arcs
            .iter()
            .filter(|a| all_arcs[**a].get_from() == *node)
            .map(|a| *a)
            .collect::<Vec<_>>()
    }

    pub fn get_reverse_star(
        vessel_arcs: &Vec<ArcIndex>,
        all_arcs: &Vec<Arc>,
        node: &NetworkNode,
    ) -> Vec<ArcIndex> {
        vessel_arcs
            .iter()
            .filter(|a| all_arcs[**a].get_to() == *node)
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
    pub fn new(problem: &Problem) -> Parameters {
        let vessel_capacity: Vec<f64> = problem.vessels().iter().map(|v| v.capacity()).collect();
        let initial_inventory: Vec<Vec<f64>> = problem
            .vessels()
            .iter()
            .map(|v| {
                (0..v.initial_inventory().as_inv().num_products())
                    .map(|p| v.initial_inventory()[p])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let travel_cost: Vec<Vec<Vec<f64>>> = problem
            .nodes()
            .iter()
            .map(|n_1| {
                problem
                    .nodes()
                    .iter()
                    .map(|n_2| {
                        problem
                            .vessels()
                            .iter()
                            .map(|v| match n_2.r#type() {
                                crate::problem::NodeType::Consumption => problem.travel_cost(
                                    n_1.index(),
                                    n_2.index(),
                                    v.index(),
                                    &Inventory::new(&vec![1.0]).unwrap(),
                                ),
                                crate::problem::NodeType::Production => problem.travel_cost(
                                    n_1.index(),
                                    n_2.index(),
                                    v.index(),
                                    &Inventory::new(&vec![0.0]).unwrap(),
                                ),
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let spot_market_cost: Vec<f64> = problem.nodes().iter().map(|_| 1.0).collect::<Vec<_>>();
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
                (0..n.initial_inventory().as_inv().num_products())
                    .map(|p| n.capacity()[p])
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
                            .map(|p| {
                                if t == 0 {
                                    0.0
                                } else {
                                    n.inventory_change(t - 1, t, p)
                                }
                            })
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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NetworkNodeType {
    Source,
    Sink,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

    pub fn get_from(&self) -> NetworkNode {
        self.from
    }

    pub fn get_to(&self) -> NetworkNode {
        self.to
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
