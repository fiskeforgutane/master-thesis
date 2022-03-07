type NodeIndex = usize;
type VesselIndex = usize;
type ProductIndex = usize;
type OrderIndex = usize;
type CompartmentIndex = usize;

/// sets for the transportation model
pub struct Sets {
    /// Set of nodes
    pub N: Vec<NodeIndex>,
    /// Set of vessels
    pub V: Vec<VesselIndex>,
    /// Set of products
    pub P: Vec<ProductIndex>,
    /// Set of products p that vessel v can transport
    pub K: Vec<Vec<ProductIndex>>,
    /// Compartments of vessels
    pub O: Vec<Vec<CompartmentIndex>>,
    /// set of orders to serve each node. Each node has orders ranging from 0..n
    pub H: Vec<OrderIndex>,
}

/// parameters for the transportation model
pub struct Parameters {
    /// 1 if node i is a producer, -1 if node i is a consumer
    pub J: Vec<isize>,
    /// Capacity of each compartment of every vessel
    pub Capacity: Vec<Vec<f64>>,
    /// lower limit on the quantity of product *p* that may be transported for node *i* to node *j* in one shipment
    pub lower_Q: Vec<Vec<Vec<usize>>>,
    /// upper limit on the quantity of product *p* that may be transported for node *i* to node *j* in one shipment
    pub upper_Q: Vec<Vec<Vec<usize>>>,
    /// the quantity either delivered or picked up at node i of product type p
    pub Q: Vec<Vec<usize>>,
    /// production of consumption rate of product p at node i
    pub R: Vec<Vec<f64>>,
    /// minimum stock levels of product p at node i
    pub S_lower: Vec<Vec<f64>>,
    /// maximum stock levels of product p at node i
    pub S_upper: Vec<Vec<f64>>,
    /// minimum transportation cost from node i to j (C_ij = min(C_ijv, i,j in N, for v in Vessels))
    pub C: Vec<Vec<usize>>,
    /// pertubations of the objective coefficients for cargoes from node i to node j
    pub epsilon: Vec<Vec<f64>>,
    /// pertubations of the objective coefficients for cargoes from node i to node j
    pub delta: Vec<Vec<f64>>,
}
