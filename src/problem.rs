use std::{
    iter::Sum,
    ops::{AddAssign, Deref, Index, IndexMut},
};

use derive_more::Constructor;
use pyo3::pyclass;

use crate::solution::Delivery;

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

#[pyclass]
#[derive(Debug, Clone)]
pub struct Problem {
    #[pyo3(get)]
    /// The vessels available for use in the problem. Assumed to be ordered by index
    vessels: Vec<Vessel>,
    #[pyo3(get)]
    /// The nodes of this problem. This contains both consumption and production nodes.
    nodes: Vec<Node>,
    #[pyo3(get)]
    /// The number of time steps in the problem
    timesteps: usize,
    #[pyo3(get)]
    /// The number of different products
    products: usize,
    #[pyo3(get)]
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
        (quantity.abs() / rate.abs()).ceil() as TimeIndex
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

    /// The cost for a vessel to move from `from` to `to`
    pub fn travel_cost(
        &self,
        from: NodeIndex,
        to: NodeIndex,
        vessel: VesselIndex,
        inventory: &Inventory,
    ) -> f64 {
        let vessel = &self.vessels[vessel];
        let unit_cost = match inventory.is_empty() {
            true => vessel.empty_travel_unit_cost(),
            false => vessel.travel_unit_cost(),
        };

        self.travel_time(from, to, vessel) as f64 * unit_cost
    }

    /// Return the `origin` visit of a `vessel`
    pub fn origin_visit(&self, vessel: VesselIndex) -> Delivery {
        let vessel = &self.vessels[vessel];
        Delivery {
            node: vessel.origin(),
            product: 0,
            time: vessel.available_from(),
            quantity: 0.0,
        }
    }
}

#[derive(Debug)]
pub enum ProblemConstructionError {
    /// The number of rows is not as expected
    DistanceWrongRowCount { expected: usize, actual: usize },
    /// One of the rows has an incorrect length
    DistanceWrongRowLength {
        row: usize,
        expected: usize,
        actual: usize,
    },
    /// A set of distances are non-negative
    NegativeDistance {
        from: usize,
        to: usize,
        distance: Distance,
    },
    /// The number of time steps must be strictly positive
    NoTimeSteps,
    /// There must be at least one product
    NoProducts,
    /// This vessel has zero compartments (must be at least one).
    NoCompartments { vessel: usize },
    /// Incorrect number of inventory changes. Should match timesteps (-1 ?)
    InventoryChangeSizeMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    /// Node `node` has negative inventory capacity for feed type `feed_type`
    NegativeInventoryCapacity { node: usize, feed_type: usize },
    /// Node `node` has wrong dimension for the inventory
    NodeInventorySizeMismatch {
        node: usize,
        expected: usize,
        actual: usize,
    },
    /// Vessel `vessel` has the wrong dimension for the inventory
    VesselInventorySizeMismatch {
        vessel: usize,
        expected: usize,
        actual: usize,
    },
    /// A vessel has a negative capacity for a compartment
    VesseNegativeCompartmentCapacity { vessel: usize, compartment: usize },
    /// Speed of vessel is zero.
    SpeedIsZero { vessel: Vessel },
    /// Origin is not a valid node index
    OriginDoesNotExist { vessel: Vessel },
}

impl Problem {
    pub fn general_checks(
        nodes: &[Node],
        distances: &[Vec<Distance>],
        products: usize,
        timesteps: usize,
    ) -> Result<(), ProblemConstructionError> {
        use ProblemConstructionError::*;
        let n = nodes.len();

        if distances.len() != n {
            return Err(DistanceWrongRowCount {
                expected: n,
                actual: distances.len(),
            });
        }

        for (i, row) in distances.iter().enumerate() {
            if row.len() != n {
                return Err(DistanceWrongRowLength {
                    row: i,
                    expected: n,
                    actual: row.len(),
                });
            }

            for (j, &x) in row.iter().enumerate() {
                if x < 0.0 {
                    return Err(NegativeDistance {
                        from: i,
                        to: j,
                        distance: x,
                    });
                }
            }
        }

        if timesteps == 0 {
            return Err(NoTimeSteps);
        }

        if products == 0 {
            return Err(NoProducts);
        }

        Ok(())
    }

    pub fn check_node(
        i: usize,
        node: &Node,
        t: usize,
        p: usize,
    ) -> Result<(), ProblemConstructionError> {
        use ProblemConstructionError::*;

        if node.initial_inventory().num_products() != p {
            return Err(NodeInventorySizeMismatch {
                node: i,
                expected: p,
                actual: node.initial_inventory().num_products(),
            });
        }

        if node.inventory_changes.len() != t {
            return Err(InventoryChangeSizeMismatch {
                node: i,
                expected: t,
                actual: node.inventory_changes.len(),
            });
        };

        for product in 0..p {
            if node.capacity[product] < 0.0 {
                return Err(ProblemConstructionError::NegativeInventoryCapacity {
                    node: i,
                    feed_type: product,
                });
            }
        }

        Ok(())
    }

    pub fn check_vessel(
        v: usize,
        vessel: &Vessel,
        _n: usize,
        _t: usize,
        _p: usize,
    ) -> Result<(), ProblemConstructionError> {
        if vessel.compartments.len() == 0 {
            return Err(ProblemConstructionError::NoCompartments { vessel: v });
        }

        for (c, compartment) in vessel.compartments.iter().enumerate() {
            if compartment.0 < 1e-5 {
                return Err(ProblemConstructionError::VesseNegativeCompartmentCapacity {
                    vessel: v,
                    compartment: c,
                });
            }
        }

        Ok(())
    }

    pub fn new(
        vessels: Vec<Vessel>,
        nodes: Vec<Node>,
        timesteps: usize,
        products: usize,
        distances: Vec<Vec<Distance>>,
    ) -> Result<Problem, ProblemConstructionError> {
        let n = nodes.len();
        let _v = vessels.len();
        let t = timesteps;
        let p = products;

        // Perform general checks
        Self::general_checks(&nodes, &distances, timesteps, products)?;
        // Check each node
        for (i, node) in nodes.iter().enumerate() {
            Self::check_node(i, node, t, p)?;
        }

        for (v, vessel) in vessels.iter().enumerate() {
            Self::check_vessel(v, vessel, n, t, p)?;
        }

        Ok(Self {
            vessels,
            nodes,
            timesteps,
            products,
            distances,
        })
    }
}

// A compartment is used to hold fed during transport
#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct Compartment(pub Quantity);

#[pyclass]
#[derive(Debug, Clone, Constructor)]
pub struct Vessel {
    #[pyo3(get)]
    /// The compartments available on the vessel.
    compartments: Vec<Compartment>,
    #[pyo3(get)]
    /// The cruising speed of this vessel, in distance units per time step
    speed: f64,
    #[pyo3(get)]
    /// The cost per time step of travel
    travel_unit_cost: Cost,
    #[pyo3(get)]
    /// The cost when travelling without a load
    empty_travel_unit_cost: Cost,
    #[pyo3(get)]
    /// The cost per time unit
    time_unit_cost: Cost,
    #[pyo3(get)]
    /// The time step from which the vessel becomes available
    available_from: usize,
    /// The initial inventory available for this vessel
    initial_inventory: FixedInventory,
    #[pyo3(get)]
    /// The origin node of the vessel
    origin: usize,
    #[pyo3(get)]
    /// The vessel class this belongs to
    class: String,
    #[pyo3(get)]
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

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Consumption,
    Production,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Node {
    #[pyo3(get)]
    /// The name of the node
    name: String,
    #[pyo3(get)]
    /// The type of node
    kind: NodeType,
    #[pyo3(get)]
    /// The index of the node
    index: usize,
    #[pyo3(get)]
    /// The maximum number of vehicles that can be present at the node at any time step
    port_capacity: Vec<usize>,
    #[pyo3(get)]
    /// The minimum amount that can be unloaded in a single time step
    min_unloading_amount: Quantity,
    #[pyo3(get)]
    /// The maximum amount that can be loaded in a single time step
    max_loading_amount: Quantity,
    #[pyo3(get)]
    /// The fixed fee associated with visiting the port
    port_fee: Cost,
    /// The maximum inventory capacity of the farm
    capacity: FixedInventory,
    /// The change in inventory during each time step.
    inventory_changes: Vec<InventoryChange>,
    #[pyo3(get)]
    /// The revenue associated with a unit sale at a farm
    /// Note: the MIRPLIB instances can "in theory" support varying revenue per time step. However, in practice,
    /// all instances uses a constant value across the entire planning period.
    revenue: Cost,
    #[pyo3(get)]
    /// The cumulative inventory at the node at the **END** of all timesteps if no loading/unloading is done. Used to allow efficient lookup
    /// of cumulative consumption between two time periods etc.
    cumulative_inventory: Vec<Vec<Quantity>>,
    /// The initial inventory of the node
    initial_inventory: FixedInventory,
}

impl Node {
    pub fn new(
        name: String,
        kind: NodeType,
        index: usize,
        port_capacity: Vec<usize>,
        min_unloading_amount: Quantity,
        max_loading_amount: Quantity,
        port_fee: Cost,
        capacity: FixedInventory,
        inventory_changes: Vec<InventoryChange>,
        revenue: Cost,
        initial_inventory: FixedInventory,
    ) -> Self {
        let mut cumulative_inventory = vec![Vec::new(); capacity.num_products()];

        for product in 0..capacity.num_products() {
            let mut inventory = initial_inventory[product];
            for delta in &inventory_changes {
                inventory += delta[product];
                cumulative_inventory[product].push(inventory);
            }
        }

        Self {
            name,
            kind,
            index,
            port_capacity,
            min_unloading_amount,
            max_loading_amount,
            port_fee,
            capacity,
            inventory_changes,
            revenue,
            cumulative_inventory,
            initial_inventory,
        }
    }

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
        let arr = self.inventory_without_deliveries(product);
        let initial = self.initial_inventory()[product];

        let index = arr.partition_point(|x| (x - initial).abs() <= amount);

        // Note: index will be zero in cases where amount < abs(delta in first time step).
        // Since we can't realistically have a "timestep before that", we do this
        match index {
            0 => 0,
            n => n - 1,
        }
    }

    pub fn cumulative_inventory(&self) -> &Vec<Vec<Quantity>> {
        &self.cumulative_inventory
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

#[pyclass]
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

    pub fn zeroed(size: usize) -> Option<Self> {
        match size {
            0 => None,
            1 => Some(Inventory(RawInventory::Single(0.0))),
            n => Some(Inventory(RawInventory::Multiple(vec![0.0; n]))),
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

    pub fn capacity_for(&self, _product: ProductIndex, compartments: &[Compartment]) -> Quantity {
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

/// Implementing Deref gives us all of the stuff from `inventory` "for free"
impl Deref for FixedInventory {
    type Target = Inventory;

    fn deref(&self) -> &Self::Target {
        &self.0
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
