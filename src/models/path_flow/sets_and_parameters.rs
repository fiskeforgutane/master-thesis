use crate::problem::NodeIndex;

type RouteIndex = usize;
type VisitIndex = usize;
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
    /// Set of last time steps that vessel v can visit visit i in route R indexed (rvi)
    pub T_r: Vec<Vec<Vec<usize>>>,
    /// # The number of visits of route r
    pub I_R: Vec<usize>,
    /// set of production nodes
    pub N_P: Vec<usize>,
    /// set of consumption nodes
    pub N_C: Vec<usize>,
}

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
    pub N_R: Vec<Vec<usize>>,
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
    // minimum loaded/unloaded in a time period if the action is initiated node n
    pub F_min: Vec<Vec<f64>>,
    // maximum loaded/unloaded in a time period if the action is initiated node n
    pub F_max: Vec<Vec<f64>>,
}

impl Parameters {
    pub fn kind(&self, n: NodeIndex) -> isize {
        todo!();
    }
    pub fn node(&self, r: RouteIndex, i: VisitIndex) -> NodeIndex {
        todo!();
    }
    pub fn visit_kind(&self, r: RouteIndex, i: VisitIndex) -> isize {
        self.kind(self.node(r, i))
    }
}
