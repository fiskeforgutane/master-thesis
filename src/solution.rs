use std::{
    cell::Cell,
    collections::HashSet,
    fmt::Debug,
    ops::{Deref, Index, Range, RangeBounds},
    vec::Drain,
};

use itertools::Itertools;
use pyo3::{pyclass, pymethods};

use crate::{
    models::path_flow::sets_and_parameters::Voyage,
    problem::{Inventory, NodeIndex, Problem, ProductIndex, Quantity, TimeIndex, VesselIndex},
};

/// A `Visit` is a visit to a `node` at a `time` where unloading/loading of a given `quantity` of `product` is started.
/// Assumption: `quantity` is relative to the node getting services. That is, a positive `quantity` means a delivery to a location,
/// while a negative quantity means a pick-up from a farm. Thus, `node.inventory[product] += quantity` while `vessel.inventory[product] -= quantity`
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Visit {
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
impl Visit {
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

/// A solution to `Problem`, i.e. a specification of the routes taken by each vehicle.
/// Note that the solution is in no way guaranteed to be optimal or even capacity-feasible.
/// However, it does guarantee that the solution is possible to actually perform. I.e. that each vessel
/// is able to perform its scheduled visits while conformning to restrictions on travel and (un)loading times.
pub struct Solution<'p> {
    /// The problem this solution belongs to
    pub problem: &'p Problem,
    /// The routes taken by each vehicle
    routes: Vec<Vec<Visit>>,
    /// A cache of the routes stored sorted by (node, product, time, vessel). This allows us to lookup visits to a (node, product)-pair within a time window in log(n)
    npt_cache: Cell<Vec<(VesselIndex, Visit)>>,
    /// A cache of the evaluation of this solution
    evaluation: Cell<Option<Evaluation>>,
}

/// Evaluation of a solution's quality
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Evaluation {
    /// The total cost of the solution
    pub cost: f64,
    /// The total amount of shortage over the planning period.
    /// I.e. shortage = sum(-min(0, inventory) for all inventories at farms)
    pub shortage: Quantity,
    /// The total amount of excess over the planning period. In other words,
    /// the sum of the amounts exceeding nodes' capacity over the planning period
    pub excess: Quantity,
}

impl Eq for Evaluation {}

impl PartialOrd for Evaluation {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // We will weigh violations equally.
        let v1 = self.shortage + self.excess;
        let v2 = other.shortage + other.excess;

        self.cost.partial_cmp(&other.cost).and(v1.partial_cmp(&v2))
    }
}

impl Ord for Evaluation {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).expect("non-nan")
    }
}

impl<'p> Debug for Solution<'p> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solution")
            .field("problem", &self.problem)
            .field("routes", &self.routes)
            .finish()
    }
}

impl<'p> Index<VesselIndex> for Solution<'p> {
    type Output = [Visit];

    fn index(&self, index: VesselIndex) -> &Self::Output {
        &self.routes[index]
    }
}

impl<'p> Clone for Solution<'p> {
    fn clone(&self) -> Self {
        Self {
            problem: self.problem,
            routes: self.routes.clone(),
            npt_cache: {
                let inner = self.npt_cache.take();
                let new = Cell::from(inner.clone());
                self.npt_cache.set(inner);
                new
            },
            evaluation: self.evaluation.clone(),
        }
    }
}

impl<'p> Solution<'p> {
    pub fn new(problem: &'p Problem, mut routes: Vec<Vec<Visit>>) -> Result<Self, InsertionError> {
        // We uphold an invariant that the routes are always sorted ascending by time.
        for route in &mut routes {
            route.sort_unstable_by_key(|x| x.time);
        }

        if routes.len() != problem.vessels().len() {
            return Err(InsertionError::IncorrectRouteCount);
        }

        let mut solution = Self {
            problem,
            routes: routes.iter().map(|r| Vec::with_capacity(r.len())).collect(),
            npt_cache: Cell::default(),
            evaluation: Cell::default(),
        };

        for (v, route) in routes.iter().enumerate() {
            for (i, visit) in route.iter().enumerate() {
                solution.can_insert(v, i, *visit)?;
            }
        }

        Ok(solution)
    }

    pub fn new_unchecked(problem: &'p Problem, routes: Vec<Vec<Visit>>) -> Self {
        Self {
            problem,
            routes,
            npt_cache: Cell::default(),
            evaluation: Cell::default(),
        }
    }

    /// Invalidate caches.
    fn invalidate_caches(&self) {
        self.evaluation.set(None);
        // Note: if we wish to avoid unnecessary allocations, we can instead
        // take the npt_cache, clear the vector, and then set it back.
        self.npt_cache.set(Vec::new());
    }

    /// Update the flat list of visists sorted by (Node, Product, Time) that allows efficient lookup of visits to a node of a certain product within a time window
    fn updated_nptv_cache(&self) -> Vec<(usize, Visit)> {
        self.routes
            .iter()
            .enumerate()
            .flat_map(|(v, visits)| visits.iter().map(move |visit| (v, *visit)))
            .sorted_unstable_by_key(|(vehicle, visit)| {
                (visit.node, visit.product, visit.time, *vehicle)
            })
            .collect()
    }

    /// Access the list of routes for each vehicle
    pub fn routes(&self) -> &[Vec<Visit>] {
        &self.routes
    }

    /// A list of visits sorted in ascending order by (NodeIndex, ProductIndex, TimeIndex, VesselIndex)
    pub fn nptv(&self) -> NPTV<'_> {
        let cache = self.npt_cache.take();

        NPTV {
            cell: &self.npt_cache,
            // We use an empty vec to signify that there is no cached value.
            // This should not occur in any other case, since there will always be a non-zero number of vehicles
            inner: match cache.is_empty() {
                true => self.updated_nptv_cache(),
                false => cache,
            },
        }
    }

    /// Return the `origin` visit of a `vessel`
    pub fn origin_visit(&self, vessel: VesselIndex) -> Visit {
        let vessel = &self.problem.vessels()[vessel];
        Visit {
            node: vessel.origin(),
            product: 0,
            time: vessel.available_from(),
            quantity: 0.0,
        }
    }

    /// Calculates the cost of this solution.
    fn cost(&self) -> f64 {
        let mut cost = 0.0;
        for (vessel, route) in self.routes.iter().enumerate() {
            let boat = &self.problem.vessels()[vessel];
            let mut inventory = boat.initial_inventory().as_inv().clone();
            let origin = self.origin_visit(vessel);
            let tour = std::iter::once(&origin).chain(route);
            for (from, to) in tour.zip(route) {
                // We should keep track of a vessel's inventory
                inventory[from.product] += from.quantity;
                // The travel cost is dependent on the quantity being carried.
                cost += self
                    .problem
                    .travel_cost(from.node, to.node, vessel, &inventory);
                // Note: we have already paid the port fee for `from`
                cost += self.problem.nodes()[to.node].port_fee();
            }
        }

        cost
    }

    /// Calculates the accumulated shortage and excess of this solution
    fn inventory_violations(&self) -> (f64, f64) {
        let mut shortage = 0.0;
        let mut excess = 0.0;

        // It should be possible to do this
        for (n, node) in self.problem.nodes().iter().enumerate() {
            for p in 0..self.problem.products() {
                let accumulative = node.inventory_without_deliveries(p);
                let mut delta = 0.0;
                for t in 0..self.problem.timesteps() {
                    // This should be either one or zero deliveries. `visit.node == n` and `visit.product == p` by construction.
                    for (_, visit) in self.deliveries(n, p, t..t + 1).iter() {
                        delta += visit.quantity;
                    }

                    // The quantity at the farm/factory
                    let inv = accumulative[t] + delta;
                    let capacity = node.capacity()[p];

                    // Calculate shortage and excess at the current time.
                    shortage += f64::min(inv, 0.0).abs();
                    excess += f64::max(0.0, inv - capacity);
                }
            }
        }

        (shortage, excess)
    }

    /// The evaluation of this solution
    pub fn evaluation(&self) -> Evaluation {
        if let Some(evaluation) = self.evaluation.get() {
            return evaluation;
        }

        let cost = self.cost();
        let (shortage, excess) = self.inventory_violations();

        let eval = Evaluation {
            cost,
            shortage,
            excess,
        };

        self.evaluation.set(Some(eval));

        eval
    }

    /// Determine the all visits to a (node, product)-pair within a time window. Includes visits by all vehicles.
    /// The returned slice will be sorted ascending by (time, vessel index)
    pub fn deliveries(
        &self,
        node: NodeIndex,
        product: ProductIndex,
        period: Range<TimeIndex>,
    ) -> NPTVSlice<'_> {
        // `start` is the first element containing relevant deliveries, while `end` is the (exclusive) end.
        // Note that `start` == `end` iff there are not deliveries that are relevant for the order.
        let nptv = self.nptv();

        let open = (node, product, period.start);
        let close = (node, product, period.end);

        let start = nptv.partition_point(|(_, v)| (v.node, v.product, v.time) < open);
        let end = nptv.partition_point(|(_, v)| (v.node, v.product, v.time) <= close);

        nptv.slice(start..end)
    }

    /// Return the inventory of a vessel at a specific point in time
    pub fn vessel_inventory_at(&self, vessel: VesselIndex, time: TimeIndex) -> Inventory {
        let vehicle = &self.problem.vessels()[vessel];
        let visits = self.routes[vessel].iter().take_while(|v| v.time <= time);
        let mut inventory = vehicle.initial_inventory().as_inv().clone();

        for visit in visits {
            inventory[visit.product] -= visit.quantity;
        }

        inventory
    }

    /// Returns the inventory of a node at a specific point in time
    pub fn node_product_inventory_at(
        &self,
        node: NodeIndex,
        product: ProductIndex,
        time: TimeIndex,
    ) -> f64 {
        let n = &self.problem.nodes()[node];
        let base = n.inventory_without_deliveries(product)[time];

        let delta = self
            .deliveries(node, product, 0..time + 1)
            .iter()
            .map(|(_, visit)| visit.quantity)
            .sum::<f64>();

        base + delta
    }

    /// Whether it is allowed to insert `visit` as visit number `position` in `vessel`'s route
    /// Returns `Ok(())` if the insertion is valid, and `Err(InsertionError::*)` otherwise
    pub fn can_insert(
        &mut self,
        vessel: VesselIndex,
        position: usize,
        visit: Visit,
    ) -> Result<(), InsertionError> {
        // To make all of this slightly more concise
        use InsertionError::*;

        let route = self.routes.get_mut(vessel).ok_or(VesselIndexOutOfBounds)?;
        let boat = &self.problem.vessels()[vessel];

        if position > route.len() {
            return Err(PositionOufOfBounds);
        }

        // The node index of the previous node.
        // If there is no previous, we will make a visit for the origin that delivers absolutely,
        // at the time where the vessel becomes available.
        let previous = match position {
            0 => Visit {
                node: boat.origin(),
                product: 0,
                time: boat.available_from(),
                quantity: 0.0,
            },
            _ => route[position - 1],
        };

        // The next visit, if any.
        let next = route.get(position + 1);

        // The earliest time at which we might arrive at the visit we're attempting to insert
        let earliest_arrival = previous.time
            + self
                .problem
                .min_loading_time(previous.node, previous.quantity)
            + self.problem.travel_time(previous.node, visit.node, boat);

        // The latest time at which we can leave the visit we're attempting to insert while still making it to the next one in time
        // (if there is no next, we still require this visit to be done by the end of the planning period)
        let latest_depart = next.map_or(self.problem.timesteps(), |next| {
            let arrival = next.time;
            let loading_time = self.problem.min_loading_time(next.node, next.quantity);
            let travel_time = self.problem.travel_time(visit.node, next.node, boat);
            arrival - loading_time - travel_time
        });

        if visit.time < earliest_arrival {
            return Err(NotEnoughTimeToReach);
        }

        if visit.time + self.problem.min_loading_time(visit.node, visit.quantity) > latest_depart {
            return Err(NotEnoughTimeToReachNext);
        }

        Ok(())
    }

    pub fn insert(
        &mut self,
        vessel: VesselIndex,
        position: usize,
        visit: Visit,
    ) -> Result<(), InsertionError> {
        // Check if we can insert the visit
        self.can_insert(vessel, position, visit)?;
        // Insert the visit
        self.routes[vessel].insert(position, visit);

        // Note: since we must retrieve self.routes directly in order to please the borrow checker,
        // we also need to invalidate the caches. This won't need to be done by users, since they can
        // only mutate solutions through `Solution::routes_mut`.
        self.invalidate_caches();

        Ok(())
    }

    pub fn drain<R: RangeBounds<usize>>(
        &mut self,
        vessel: VesselIndex,
        range: R,
    ) -> Drain<'_, Visit> {
        self.routes[vessel].drain(range)
    }

    /// extracts the unique voyages for every vessel
    pub fn voyages(&self) -> HashSet<Voyage> {
        let mut voyages = HashSet::new();
        let nodes = self.problem.nodes();
        for route in &self.routes {
            // get the indices of the production nodes in the visist
            let split_idxs: Vec<usize> = route
                .iter()
                .enumerate()
                .filter_map(|(i, visit)| match nodes[visit.node].r#type() {
                    crate::problem::NodeType::Consumption => None,
                    crate::problem::NodeType::Production => Some(i),
                })
                .collect();

            // get the slices for every voyage starting and ending at a production node
            for e in split_idxs.windows(2) {
                let (i, j) = (e[0], e[1]);
                let node_idxs: Vec<NodeIndex> =
                    route[i..=j].iter().map(|visit| visit.node).collect();
                // insert the voyage to the hash set (does not insert if the voyage is already there)
                voyages.insert(node_idxs);
            }

            // get the first and last voyage which might not start and end at a production node
            let first_idx = split_idxs.first();
            if let Some(x) = first_idx {
                let node_idxs: Vec<NodeIndex> =
                    route[0..=*x].iter().map(|visit| visit.node).collect();
                if !node_idxs.is_empty() {
                    voyages.insert(node_idxs);
                }
            }

            let last_idx = split_idxs.last();
            if let Some(x) = last_idx {
                let node_idxs: Vec<NodeIndex> =
                    route[*x..].iter().map(|visit| visit.node).collect();
                if !node_idxs.is_empty() {
                    voyages.insert(node_idxs);
                }
            }
        }
        // convert the identified voyages to voyage objects
        voyages
            .into_iter()
            .map(|v| Voyage::new(v, self.problem))
            .collect()
    }
}

pub struct NPTV<'cell> {
    /// The Cell that is supposed to hold the `inner` value when this is dropped
    cell: &'cell Cell<Vec<(VesselIndex, Visit)>>,
    /// The inner Vec
    inner: Vec<(VesselIndex, Visit)>,
}

impl<'cell> NPTV<'cell> {
    pub fn slice(self, range: Range<usize>) -> NPTVSlice<'cell> {
        NPTVSlice {
            inner: self,
            range: range,
        }
    }
}

impl<'cell> Deref for NPTV<'cell> {
    type Target = [(VesselIndex, Visit)];

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'cell> Drop for NPTV<'cell> {
    fn drop(&mut self) {
        let inner = std::mem::take(&mut self.inner);
        self.cell.set(inner);
    }
}

pub struct NPTVSlice<'cell> {
    inner: NPTV<'cell>,
    range: Range<usize>,
}

impl<'cell> Deref for NPTVSlice<'cell> {
    type Target = [(VesselIndex, Visit)];

    fn deref(&self) -> &Self::Target {
        &self.inner[self.range.clone()]
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
}
