use crate::problem::{Cost, NodeIndex, Problem, Vessel};

type RouteIndex = usize;
type VisitIndex = usize;
type RoutePool = Vec<Route>;

pub struct Route {
    visits: Vec<NodeIndex>,
    costs: Vec<Cost>,
}

impl Route {
    pub fn new(visits: Vec<NodeIndex>, problem: &Problem) -> Route {
        let costs = problem
            .vessels()
            .iter()
            .map(|v| Self::cost(&visits, problem, v))
            .collect();
        Route { visits, costs }
    }

    pub fn cost(visits: &Vec<NodeIndex>, problem: &Problem, vessel: &Vessel) -> f64 {
        let mut res = 0.0;
        for n in &visits[0..visits.len() - 1] {
            let t_time = problem.travel_time(*n, visits[n + 1], vessel);
            let t_cost = t_time as f64 * vessel.travel_unit_cost();
            // port fee of the next port, the first node is thus not accounted for as this was accounted for in the route bringing the vessel
            // to the origin of the next route
            let port_fee = vessel.port_fee(visits[n + 1]);
            res += t_cost + port_fee;
        }
        res
    }

    /// Get a reference to the route's costs.
    pub fn costs(&self) -> &[f64] {
        self.costs.as_ref()
    }

    /// Get a reference to the route's visits.
    pub fn visits(&self) -> &[usize] {
        self.visits.as_ref()
    }
}

#[allow(non_snake_case)]
pub struct Sets {
    /// Set of routes
    pub R: usize,
    /// Set of Vessels
    pub V: usize,
    /// Set of Nodes
    pub N: usize,
    /// Set of products
    pub P: usize,
    /// Set of time steps
    pub T: usize,
    /// Set of the last time step that vessel v can be at visit i in route R indexed (rvi)
    pub T_r: Vec<Vec<Vec<usize>>>,
    /// # The number of visits of route r
    pub I_r: Vec<usize>,
    /// set of production nodes
    pub N_P: Vec<usize>,
    /// set of consumption nodes
    pub N_C: Vec<usize>,
}

impl Sets {
    pub fn new(problem: &Problem, routes: &RoutePool) -> Sets {
        Sets {
            R: routes.len(),
            V: problem.vessels().len(),
            N: problem.nodes().len(),
            P: problem.products(),
            T: problem.timesteps(),
            T_r: routes.iter().map(|r| Self::get_T_r(problem, r)).collect(),
            I_r: routes.iter().map(|r| r.visits().len()).collect(),
            N_P: problem
                .production_nodes()
                .iter()
                .map(|n| n.index())
                .collect(),
            N_C: problem
                .consumption_nodes()
                .iter()
                .map(|n| n.index())
                .collect(),
        }
    }

    #[allow(non_snake_case)]
    pub fn get_T_r(problem: &Problem, route: &Route) -> Vec<Vec<usize>> {
        // we assume that all production nodes always are feasible destination nodes, i.e. the end of the route is always good
        let mut vessel_res = Vec::new();
        for v in problem.vessels() {
            let mut t = problem.timesteps() - 1;
            println!("orn number {:?}", t);
            let mut next = route.visits().last().unwrap();
            println!("original next {:?}", next);
            let mut times = route
                .visits()
                .iter()
                .rev()
                .map(|n| {
                    let t_time = problem.travel_time(*n, *next, v);
                    let last = t - t_time;
                    t = last;
                    next = n;
                    last
                })
                .collect::<Vec<_>>();
            times.reverse();
            vessel_res.push(times);
        }
        vessel_res
    }

    pub fn add_route(&mut self, problem: &Problem, route: Route) {
        self.R += 1;
        self.T_r.push(Self::get_T_r(problem, &route));
        self.I_r.push(route.visits().len());
    }
}

#[allow(non_snake_case)]
pub struct Parameters {
    /// Cost of route r for vessel v
    /// This includes the travel cost and fixed port fees,
    /// but not fixed cost of operating the vessel as this depends on the time the vessel spends in port
    pub C_r: Vec<Vec<f64>>,
    /// Initial inventory at node i of product p
    pub S_0: Vec<Vec<f64>>,
    /// Consumption (or production) at node i of product p at time period t
    pub D: Vec<Vec<Vec<f64>>>,
    /// The time step in which vessel v becomes available
    pub T_0: Vec<usize>,
    /// The travel time from visit i to visit i+1 for vessel v in route r, indexed (r,i,v)
    pub travel: Vec<Vec<Vec<usize>>>,
    /// The nodeindex of visit i in route r, indexed (r,i)
    pub N_r: Vec<Vec<usize>>,
    /// Orign of vessel v
    pub O: Vec<usize>,
    /// capacity for product p in time period t at node n, indexed(n,p,t)
    pub S_max: Vec<Vec<Vec<f64>>>,
    /// lower limit for product p in time period t at node n, indexed(n,p,t)
    pub S_min: Vec<Vec<Vec<f64>>>,
    /// initial load of vessel v of product p
    pub L_0: Vec<Vec<f64>>,
    /// capacity of vessel v
    pub Q: Vec<f64>,
    // minimum loaded/unloaded in a time period if the action is initiated node n, indexed (n,p)
    pub F_min: Vec<Vec<f64>>,
    // maximum loaded/unloaded in a time period if the action is initiated node n, indexed (n,p)
    pub F_max: Vec<Vec<f64>>,
    // The kind of node, 1 for production, -1 for consumption
    kind: Vec<isize>,
    // routes
    routes: Vec<Vec<NodeIndex>>,
}

#[allow(unused, non_snake_case)]
impl Parameters {
    pub fn new(problem: &Problem, sets: &Sets, routes: &RoutePool) -> Parameters {
        let C_r = routes.iter().map(|r: &Route| r.costs().to_vec()).collect();
        let S_0 = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.products())
                    .map(|p| n.initial_inventory()[p])
                    .collect()
            })
            .collect();
        let D = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.timesteps())
                    .map(|t| {
                        (0..problem.products())
                            .map(|p| n.inventory_changes()[t][p])
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let T_0 = problem
            .vessels()
            .iter()
            .map(|v| v.available_from())
            .collect();

        let travel = routes
            .iter()
            .map(|r: &Route| {
                r.visits()[0..r.visits().len() - 1]
                    .iter()
                    .enumerate()
                    .map(|(i, n)| {
                        problem
                            .vessels()
                            .iter()
                            .map(|v| problem.travel_time(*n, r.visits()[i + 1], v))
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let N_r = routes.iter().map(|r| r.visits().to_vec()).collect();
        let O = problem.vessels().iter().map(|v| v.origin()).collect();
        let S_max = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.products())
                    .map(|p| vec![n.capacity()[p]; problem.timesteps()])
                    .collect()
            })
            .collect();
        let S_min = problem
            .nodes()
            .iter()
            .map(|n| {
                (0..problem.products())
                    .map(|p| vec![0.0; problem.timesteps()])
                    .collect()
            })
            .collect();

        let L_0 = problem
            .vessels()
            .iter()
            .map(|v| {
                (0..problem.products())
                    .map(|p| v.initial_inventory()[p])
                    .collect()
            })
            .collect();

        let Q = problem
            .vessels()
            .iter()
            .map(|v| v.compartments().iter().map(|c| c.0).sum())
            .collect();

        let F_min = problem
            .nodes()
            .iter()
            .map(|n| vec![n.min_unloading_amount(); problem.products()])
            .collect();
        let F_max = problem
            .nodes()
            .iter()
            .map(|n| vec![n.max_loading_amount(); problem.products()])
            .collect();

        let kind = problem
            .nodes()
            .iter()
            .map(|n| match n.r#type() {
                crate::problem::NodeType::Consumption => -1,
                crate::problem::NodeType::Production => 1,
            })
            .collect();

        let routes = routes
            .into_iter()
            .map(|r| r.visits().clone().to_vec())
            .collect();
        Parameters {
            C_r,
            S_0,
            D,
            T_0,
            travel,
            N_r,
            O,
            S_max,
            S_min,
            L_0,
            Q,
            F_min,
            F_max,
            kind,
            routes,
        }
    }

    pub fn kind(&self, n: NodeIndex) -> isize {
        self.kind[n]
    }
    pub fn node(&self, r: RouteIndex, i: VisitIndex) -> Option<NodeIndex> {
        Some(*self.routes.get(r)?.get(i)?)
    }
    pub fn visit_kind(&self, r: RouteIndex, i: VisitIndex) -> Option<isize> {
        Some(self.kind(self.node(r, i)?))
    }

    pub fn add_route(&mut self, route: &Route, problem: &Problem) {
        self.C_r.push(route.costs().to_vec());
        self.travel.push(
            route.visits()[0..route.visits().len() - 1]
                .iter()
                .enumerate()
                .map(|(i, n)| {
                    problem
                        .vessels()
                        .iter()
                        .map(|v| problem.travel_time(*n, route.visits()[i + 1], v))
                        .collect()
                })
                .collect(),
        );
        self.N_r.push(route.visits().to_vec());
        self.routes.push(route.visits().clone().to_vec());
    }
}
