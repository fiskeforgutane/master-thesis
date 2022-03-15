use std::{
    iter::Sum,
    ops::{AddAssign, Index, IndexMut},
};

use derive_more::Constructor;

/// A point in Euclidean 2d-space.
pub struct Point(f64, f64);

/// The type used for inventory quantity
pub type Quantity = f64;
/// The type used for distance
pub type Distance = f64;
/// The typs used for cost.
pub type Cost = f64;

pub type NodeIndex = usize;
pub type VesselIndex = usize;
pub type TimeIndex = usize;
pub type ProductIndex = usize;

#[derive(Debug, Clone, Constructor)]
pub struct Problem {
    /// The vessels available for use in the problem. Assumed to be ordered by index
    vessels: Vec<Vessel>,
    /// The nodes of this problem. This contains both consumption and production nodes.
    nodes: Vec<Node>,
    /// The number of time steps in the problem
    timesteps: usize,
    /// The number of different products
    products: usize,
    /// A distance matrix between the different nodes.
    distances: Vec<Vec<Distance>>,
}

impl Problem {
    /// The vessels available for use in the problem. Ordered by index (continuous, starting at 0)
    pub fn vessels(&self) -> &[Vessel] {
        &self.vessels
    }

    /// The nodes of this problem. This contains both consumption and production nodes.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The number of time steps in the problem
    pub fn timesteps(&self) -> usize {
        self.timesteps
    }

    /// The products in the problem
    pub fn products(&self) -> usize {
        self.products
    }

    /// The distance between two nodes
    pub fn distance(&self, from: NodeIndex, to: NodeIndex) -> Distance {
        self.distances[from][to]
    }

    /*
    /// The time required for `vessel` to travel from `from` to `to`.
    pub fn travel_time(&self, from: NodeIndex, to: NodeIndex, vessel: VesselIndex) -> TimeIndex {
        let speed = self.vessels[vessel].speed();
        (self.distance(from, to) / speed).ceil() as TimeIndex
    }*/

    /// The minimum amount of time we need to spend at `node` in order to load/unload `quantity`.
    pub fn min_loading_time(&self, node: NodeIndex, quantity: Quantity) -> TimeIndex {
        let rate = self.nodes[node].max_loading_amount();
        (rate / quantity).ceil() as TimeIndex
    }
    /// Returns the consumption nodes of the problem
    /// **VERY BAD** should be done once in the constructor
    pub fn consumption_nodes(&self) -> Vec<&Node> {
        self.nodes()
            .iter()
            .filter_map(|n: &Node| match n.r#type() {
                NodeType::Consumption => Some(n),
                NodeType::Production => None,
            })
            .collect()
    }

    /// Returns the production nodes of the problem
    /// **VERY BAD** should be done once in the constructor
    pub fn production_nodes(&self) -> Vec<&Node> {
        self.nodes()
            .iter()
            .filter_map(|n: &Node| match n.r#type() {
                NodeType::Consumption => None,
                NodeType::Production => Some(n),
            })
            .collect()
    }

    /// Returns the closes production node for the given node
    pub fn closest_production_node(&self, node: &Node) -> &Node {
        let prod_nodes = self.production_nodes();
        prod_nodes
            .iter()
            .min_by(|a, b| {
                self.distance(a.index(), node.index())
                    .partial_cmp(&self.distance(b.index(), node.index()))
                    .unwrap()
            })
            .unwrap()
    }

    pub fn travel_time(&self, from: NodeIndex, to: NodeIndex, vessel: &Vessel) -> usize {
        let distance = self.distance(from, to);
        let speed = vessel.speed();
        (distance / speed).ceil() as usize
    }
}

#[derive(Debug)]
pub enum ProblemConstructionError {
    /// The size of the distance matrix is not as expected,
    DistanceSizeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
    /// The number of time steps must be strictly positive
    NoTimeSteps,
    /// There must be at least one product
    NoProducts,
    /// This node has zero compartments (must be at least one).
    NoCompartments(Node),
    /// Incorrect number of inventory changes. Should match timesteps (-1 ?)
    InventoryChangeSizeMismatch {
        node: Node,
        expected: usize,
        actual: usize,
    },
    /// Node `node` has negative inventory capacity for feed type `feed_type`
    NegativeInventoryCapacity { node: Node, feed_type: usize },
    /// Node `node` has wrong dimension for the inventory
    NodeInventorySizeMismatch {
        node: Node,
        expected: usize,
        actual: usize,
    },
    /// Vessel `vessel` has the wrong dimension for the inventory
    VesselInventorySizeMismatch {
        node: Node,
        expected: usize,
        actual: usize,
    },
    /// Speed of vessel is zero.
    SpeedIsZero { vessel: Vessel },
    /// Origin is not a valid node index
    OriginDoesNotExist { vessel: Vessel },
}

impl Problem {
    /* pub fn new(
        _vessels: Vec<Vessel>,
        _nodes: Vec<Node>,
        _timesteps: usize,
        _products: usize,
        _distances: Vec<Vec<Distance>>,
    ) -> Result<Problem, ProblemConstructionError> {
        todo!()
    } */
}

// A compartment is used to hold fed during transport
#[derive(Debug, Clone, Copy)]
pub struct Compartment(pub Quantity);

#[derive(Debug, Clone, Constructor)]
pub struct Vessel {
    /// The compartments available on the vessel.
    compartments: Vec<Compartment>,
    /// The cruising speed of this vessel, in distance units per time step
    speed: f64,
    /// The cost per time step of travel
    travel_unit_cost: Cost,
    /// The cost when travelling without a load
    empty_travel_unit_cost: Cost,
    /// The cost per time unit
    time_unit_cost: Cost,
    /// The port fee associated with docking at each port
    port_fee: Vec<Cost>,
    /// The time step from which the vessel becomes available
    available_from: usize,
    /// The initial inventory available for this vessel
    initial_inventory: FixedInventory,
    /// The origin node of the vessel
    origin: usize,
    /// The vessel class this belongs to
    class: String,
    /// The index of the vessel
    index: usize,
}

impl Vessel {
    /// The compartments available on the vessel.
    pub fn compartments(&self) -> &[Compartment] {
        &self.compartments
    }
    /// The cruising speed of this vessel, in distance units per time step
    pub fn speed(&self) -> f64 {
        self.speed
    }

    /// The cost per time step of travel
    pub fn travel_unit_cost(&self) -> Cost {
        self.travel_unit_cost
    }

    /// The cost when travelling without a load
    pub fn empty_travel_unit_cost(&self) -> Cost {
        self.empty_travel_unit_cost
    }

    /// The cost per time unit
    pub fn time_unit_cost(&self) -> Cost {
        self.time_unit_cost
    }

    /// The cost per time step while docked at a port
    pub fn port_fee(&self, node: NodeIndex) -> Cost {
        self.port_fee[node]
    }
    /// The time step from which the vessel becomes available
    pub fn available_from(&self) -> TimeIndex {
        self.available_from
    }
    /// The initial inventory available for this vessel
    pub fn initial_inventory(&self) -> &FixedInventory {
        &self.initial_inventory
    }
    /// The origin node of the vessel
    pub fn origin(&self) -> NodeIndex {
        self.origin
    }
    /// The vessel class this belongs to
    pub fn class(&self) -> &str {
        self.class.as_str()
    }

    /// The index of the vessel
    pub fn index(&self) -> usize {
        self.index
    }
}

#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Consumption,
    Production,
}

#[derive(Debug, Clone, Constructor)]
pub struct Node {
    /// The name of the node
    name: String,
    /// The type of node
    kind: NodeType,
    /// The index of the node
    index: usize,
    /// The maximum number of vehicles that can be present at the node at any time step
    port_capacity: Vec<usize>,
    /// The minimum amount that can be unloaded in a single time step
    min_unloading_amount: Quantity,
    /// The maximum amount that can be loaded in a single time step
    max_loading_amount: Quantity,
    /// The fixed fee associated with visiting the port
    port_fee: Cost,
    /// The maximum inventory capacity of the farm
    capacity: FixedInventory,
    /// The change in inventory during each time step.
    inventory_changes: Vec<InventoryChange>,
    /// The revenue associated with a unit sale at a farm
    /// Note: the MIRPLIB instances can "in theory" support varying revenue per time step. However, in practice,
    /// all instances uses a constant value across the entire planning period.
    revenue: Cost,
    /// The cumulative inventory at the node at the **END** of all timesteps if no loading/unloading is done. Used to allow efficient lookup
    /// of cumulative consumption between two time periods etc.
    cumulative_inventory: Vec<Vec<Quantity>>,
    /// The initial inventory of the node
    initial_inventory: FixedInventory,
}

impl Node {
    /// The name of the node
    pub fn name(&self) -> &str {
        self.name.as_str()
    }
    /// The type of node
    pub fn r#type(&self) -> NodeType {
        self.kind
    }
    /// The index of the node
    pub fn index(&self) -> NodeIndex {
        self.index
    }
    /// The maximum number of vehicles that can be present at the node at any time step
    pub fn port_capacity(&self) -> &[usize] {
        &self.port_capacity
    }
    /// The minimum amount that can be unloaded in a single time step
    pub fn min_unloading_amount(&self) -> Quantity {
        self.min_unloading_amount
    }
    /// The maximum amount that can be loaded in a single time step
    pub fn max_loading_amount(&self) -> Quantity {
        self.max_loading_amount
    }
    /// The fixed fee associated with visiting the port
    pub fn port_fee(&self) -> Cost {
        self.port_fee
    }
    /// The maximum inventory capacity of the farm
    pub fn capacity(&self) -> &FixedInventory {
        &self.capacity
    }
    /// The change in inventory during each time step.
    pub fn inventory_changes(&self) -> &[InventoryChange] {
        &self.inventory_changes
    }

    /// The inventory at a given time step for a given product, assuming no deliveries.
    pub fn inventory_without_deliveries(&self, product: ProductIndex) -> &[Quantity] {
        self.cumulative_inventory[product].as_slice()
    }

    /// Returns the change in inventory in the **inclusive range** [from - to] for the given product.
    /// I.e. it will return the quantity produced or consumed of the given product from the beginning of the from-period to the end of the to-period
    /// ## Note
    /// It is assumed that there are no deliveries/pickups at the node
    ///
    /// If the node is a conumption node, the result will be a negative number, and positive in the case of production ondes
    pub fn inventory_change(&self, from: TimeIndex, to: TimeIndex, product: ProductIndex) -> f64 {
        self.inventory_without_deliveries(product)[to]
            - (self.inventory_without_deliveries(product)[from]
                - self.inventory_changes()[from][product]) // subtract the inventory produced or consumed in the from-period
    }

    /// The revenue associated with a unit sale at a farm
    /// Note: the MIRPLIB instances can "in theory" support varying revenue per time step. However, in practice,
    /// all instances uses a constant value across the entire planning period.
    pub fn revenue(&self) -> Cost {
        self.revenue
    }
    /// The initial inventory of the node
    pub fn initial_inventory(&self) -> &FixedInventory {
        &self.initial_inventory
    }

    /// The timestep before the node has consumed/produced at least the given amount of the given product
    pub fn inventory_change_at_least(&self, product: ProductIndex, amount: Quantity) -> TimeIndex {
        let initial_inv = self.initial_inventory()[product];
        match self
            .inventory_without_deliveries(product)
            .binary_search_by(|k| k.partial_cmp(&(amount + initial_inv)).unwrap())
        {
            Ok(x) => x,
            Err(x) => x,
        }
    }
}

/// Inventory at either a node or a vessel.
#[derive(Debug, Clone)]
enum RawInventory {
    /// Inventory when we only have a single product type.
    Single(Quantity),
    /// Inventory for the case of multiple products.
    Multiple(Vec<Quantity>),
}

#[derive(Debug, Clone)]
pub struct Inventory(RawInventory);

impl Inventory {
    pub fn new(value: &[Quantity]) -> Option<Self> {
        match value.len() {
            0 => None,
            1 => Some(Inventory(RawInventory::Single(value[0]))),
            _ => Some(Inventory(RawInventory::Multiple(value.to_vec()))),
        }
    }

    pub fn single(value: Quantity) -> Self {
        Inventory(RawInventory::Single(value))
    }

    pub fn num_products(&self) -> usize {
        match &self.0 {
            RawInventory::Single(_) => 1,
            RawInventory::Multiple(xs) => xs.len(),
        }
    }

    pub fn capacity_for(&self, product: ProductIndex, compartments: &[Compartment]) -> Quantity {
        if self.num_products() == 1 {
            let capacity = compartments.iter().map(|c| c.0).sum::<f64>();
            return capacity - self[0];
        }

        todo!("capacity not implemented for multiple product types yet.");
    }

    pub fn fixed(self) -> FixedInventory {
        FixedInventory::from(self)
    }

    pub fn is_empty(&self) -> bool {
        let epsilon = 1.0e-6;
        match &self.0 {
            RawInventory::Single(x) => x.abs() <= epsilon,
            RawInventory::Multiple(xs) => xs.iter().sum::<f64>().abs() <= epsilon,
        }
    }
}
impl From<FixedInventory> for Inventory {
    fn from(inventory: FixedInventory) -> Self {
        inventory.0
    }
}

impl Index<usize> for Inventory {
    type Output = Quantity;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Inventory(RawInventory::Single(inventory)) => inventory,
            Inventory(RawInventory::Multiple(inventory)) => &inventory[index],
        }
    }
}

impl IndexMut<usize> for Inventory {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Inventory(RawInventory::Single(inventory)) => inventory,
            Inventory(RawInventory::Multiple(inventory)) => &mut inventory[index],
        }
    }
}

/// An inventory that can not be changed.
#[derive(Debug, Clone)]
pub struct FixedInventory(Inventory);

impl FixedInventory {
    pub fn as_inv(&self) -> &Inventory {
        &self.0
    }

    pub fn unfixed(self) -> Inventory {
        self.0
    }
}

impl From<Inventory> for FixedInventory {
    fn from(inventory: Inventory) -> Self {
        Self(inventory)
    }
}

impl Index<usize> for FixedInventory {
    type Output = <Inventory as Index<usize>>::Output;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// A change in inventory can also be represented as an inventory
type InventoryChange = FixedInventory;

impl<'a> AddAssign<&'a Inventory> for Inventory {
    fn add_assign(&mut self, rhs: &'a Inventory) {
        match (self, rhs) {
            (Inventory(RawInventory::Single(lhs)), Inventory(RawInventory::Single(rhs))) => {
                *lhs += *rhs
            }
            (Inventory(RawInventory::Multiple(lhs)), Inventory(RawInventory::Single(rhs))) => {
                lhs[0] += *rhs
            }
            (Inventory(RawInventory::Multiple(lhs)), Inventory(RawInventory::Multiple(rhs))) => {
                // Element-wise addition of each feed type
                for (x, y) in lhs.iter_mut().zip(rhs) {
                    *x += y;
                }

                // If the RHS has more feed types than the LHS, we need to extend the LHS.
                if lhs.len() < rhs.len() {
                    lhs.extend_from_slice(&rhs[lhs.len()..]);
                }
            }
            // This case is (single, multiple)
            (inventory, Inventory(RawInventory::Multiple(rhs))) => {
                let mut inner = rhs.clone();
                inner[0] += inventory[0];
                *inventory = Inventory(RawInventory::Multiple(inner));
            }
        }
        todo!()
    }
}

impl<'a> AddAssign<Inventory> for Inventory {
    fn add_assign(&mut self, rhs: Inventory) {
        match (self, rhs) {
            (Inventory(RawInventory::Single(lhs)), Inventory(RawInventory::Single(rhs))) => {
                *lhs += rhs
            }
            (Inventory(RawInventory::Multiple(lhs)), Inventory(RawInventory::Single(rhs))) => {
                lhs[0] += rhs
            }
            (
                Inventory(RawInventory::Multiple(lhs)),
                Inventory(RawInventory::Multiple(mut rhs)),
            ) => {
                if lhs.len() < rhs.len() {
                    std::mem::swap(lhs, &mut rhs);
                }

                // Element-wise addition of each feed type.
                // The swap above guarantees that lhs.len() >= rhs.len(), which ensures that this is value
                for (x, y) in lhs.iter_mut().zip(rhs) {
                    *x += y;
                }
            }
            // This case is (single, multiple)
            (lhs, rhs) => {
                let value = lhs[0];
                let _ = std::mem::replace(lhs, rhs);
                lhs[0] += value;
            }
        }
    }
}

impl<'a> Sum<&'a Inventory> for Inventory {
    fn sum<I: Iterator<Item = &'a Inventory>>(iter: I) -> Self {
        let mut sum = Inventory(RawInventory::Single(0.0));

        for i in iter {
            sum += i;
        }

        sum
    }
}

impl<'a> Sum<&'a FixedInventory> for Inventory {
    fn sum<I: Iterator<Item = &'a FixedInventory>>(iter: I) -> Self {
        let mut sum = Inventory(RawInventory::Single(0.0));

        for i in iter {
            sum += i.as_inv();
        }

        sum
    }
}
