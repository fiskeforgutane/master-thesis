use std::{
    cell::Cell,
    fmt::Debug,
    ops::{Deref, Index, Range, RangeBounds},
    vec::Drain,
};

use float_ord::FloatOrd;
use itertools::Itertools;
use pyo3::{pyclass, pymethods};

use crate::problem::{
    FixedInventory, Inventory, NodeIndex, Problem, ProductIndex, Quantity, TimeIndex, VesselIndex,
};

pub mod explicit;

pub use explicit::{NPTVSlice, Solution, NPTV};

pub trait AnySolution {
    type Inner: Deref<Target = [Delivery]>;

    /// The problem this solution belongs to
    fn problem(&self) -> &Problem;
    /// A list of visits for each vehicle. Each vehicle's list of visits *must* be ordered ascending by time
    fn routes(&self) -> &[Self::Inner];
}

/// A `Visit` is a visit to a `node` at a `time` where unloading/loading of a given `quantity` of `product` is started.
/// Assumption: `quantity` is relative to the node getting services. That is, a positive `quantity` means a delivery to a location,
/// while a negative quantity means a pick-up from a farm. Thus, `node.inventory[product] += quantity` while `vessel.inventory[product] -= quantity`
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Delivery {
    #[pyo3(get, set)]
    /// The node we're visiting.
    pub node: NodeIndex,
    #[pyo3(get, set)]
    /// The product being delivered.
    pub product: ProductIndex,
    #[pyo3(get, set)]
    /// The time at which delivery starts.
    pub time: TimeIndex,
    #[pyo3(get, set)]
    /// The quantity delivered.
    pub quantity: Quantity,
}

#[pymethods]
impl Delivery {
    #[new]
    pub fn new(
        node: NodeIndex,
        product: ProductIndex,
        time: TimeIndex,
        quantity: Quantity,
    ) -> Self {
        Self {
            node,
            product,
            time,
            quantity,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InventoryViolation {
    /// The total amount by which upper inventory levels are exceeded over the planning period.
    pub excess: Quantity,
    /// The total amount by which lower inventory levels are violated over the planning period.
    pub shortage: Quantity,
}

/// Evaluation of a solution's quality
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluation {
    #[pyo3(get)]
    /// The total cost of the solution
    pub cost: f64,
    #[pyo3(get)]
    /// Statistics for the nodes
    pub nodes: InventoryViolation,
    #[pyo3(get)]
    /// Statistics for the vessels
    pub vessels: InventoryViolation,
}

impl Evaluation {
    pub fn inventory_violation(&self) -> f64 {
        self.nodes.excess + self.nodes.shortage + self.vessels.excess + self.vessels.shortage
    }
}

impl Eq for Evaluation {}

impl PartialOrd for Evaluation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // We will weigh violations equally.
        let v1 =
            self.nodes.shortage + self.nodes.excess + self.vessels.shortage + self.vessels.excess;
        let v2 = other.nodes.shortage
            + other.nodes.excess
            + other.vessels.shortage
            + other.vessels.excess;

        FloatOrd(v1).partial_cmp(&FloatOrd(v2)).into()
    }
}

impl Ord for Evaluation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("non-nan")
    }
}

#[derive(Debug)]
pub enum InsertionError {
    /// The number of routes is incorrect
    IncorrectRouteCount,
    /// The vessel index is invalid
    VesselIndexOutOfBounds,
    /// The position we're trying to insert at is out of bounds,
    PositionOufOfBounds,
    /// There is not enough time to reach the node we're trying to insert in time to
    /// serve it at the required time step
    NotEnoughTimeToReach,
    /// There is not enough time to reach the node after the one we're trying to insert in time
    /// to serve it at the required time.
    NotEnoughTimeToReachNext,
    /// A visit is invalid
    InvalidVisit,
}
