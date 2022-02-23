use std::ops::{AddAssign, Index, IndexMut};

/// A point in Euclidean 2d-space.
pub struct Point(f64, f64);

/// The type used for inventory quantity
type InventoryType = f64;
/// The type used for distance
type Distance = f64;
/// The typs used for cost.
type Cost = f64;

#[derive(Debug, Clone)]
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

pub enum ProblemConstructionError {
    DistanceSizeMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
    },
}

impl Problem {
    pub fn new(
        _vessels: Vec<Vessel>,
        _nodes: Vec<Node>,
        _timesteps: usize,
        _products: usize,
        _distances: Vec<Vec<Distance>>,
    ) -> Result<Problem, ProblemConstructionError> {
        todo!()
    }
}

// A compartment is used to hold fed during transport
#[derive(Debug, Clone, Copy)]
pub struct Compartment(pub InventoryType);

#[derive(Debug, Clone)]
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
    /// The cost per time step while docked at a port
    port_unit_cost: Cost,
    /// The time step from which the vessel becomes available
    available_from: usize,
    /// The initial inventory available for this vessel
    initial_inventory: FixedInventory,
    /// The origin node of the vessel
    origin: usize,
    /// The vessel class this belongs to
    class: String,
}

#[derive(Debug, Clone, Copy)]
pub enum NodeType {
    Consumption,
    Production,
}

#[derive(Debug, Clone)]
pub struct Node {
    /// The name of the node
    pub name: String,
    /// The type of node
    r#type: NodeType,
    /// The index of the node
    index: usize,
    /// The maximum number of vehicles that can be present at the node at any time step
    port_capacity: Vec<usize>,
    /// The minimum amount that can be unloaded in a single time step
    min_unloading_amount: Inventory,
    /// The maximum amount that can be loaded in a single time step
    max_loading_amount: Inventory,
    /// The fixed fee associated with visiting the port
    port_fee: Cost,
    /// The maximum inventory capacity of the farm
    capacity: FixedInventory,
    /// The change in inventory during each time step
    inventory_changes: Vec<InventoryChange>,
    /// The revenue associated with a unit sale at a farm
    /// Note: the MIRPLIB instances can "in theory" support varying revenue per time step. However, in practice,
    /// all instances uses a constant value across the entire planning period.
    revenue: Cost,
}

/// Inventory at either a node or a vessel.
#[derive(Debug, Clone)]
enum RawInventory {
    /// Inventory when we only have a single product type.
    Single(InventoryType),
    /// Inventory for the case of multiple products.
    Multiple(Vec<InventoryType>),
}

#[derive(Debug, Clone)]
pub struct Inventory(RawInventory);

impl Inventory {
    pub fn new(value: &[InventoryType]) -> Option<Self> {
        match value.len() {
            0 => None,
            1 => Some(Inventory(RawInventory::Single(value[0]))),
            _ => Some(Inventory(RawInventory::Multiple(value.to_vec()))),
        }
    }
}
impl From<FixedInventory> for Inventory {
    fn from(inventory: FixedInventory) -> Self {
        inventory.0
    }
}

impl Index<usize> for Inventory {
    type Output = InventoryType;

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
