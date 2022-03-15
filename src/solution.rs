use std::{
    cell::Cell,
    fmt::Debug,
    ops::{Deref, Index, Range, RangeBounds},
    vec::Drain,
};

use itertools::Itertools;
use pyo3::{pyclass, pymethods};

use crate::problem::{
    Inventory, NodeIndex, Problem, ProductIndex, Quantity, TimeIndex, VesselIndex,
};

/// A `Visit` is a visit to a `node` at a `time` where unloading/loading of a given `quantity` of `product` is started.
/// Assumption: quantity is relative to the vessel's inventory. In other words, the quantity is positive if an amount is loaded onto the
/// vessel and negative is an amount is unloaded.
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
        }
    }
}

impl<'p> Solution<'p> {
    pub fn new(problem: &'p Problem, mut routes: Vec<Vec<Visit>>) -> Result<Self, InsertionError> {
        // We uphold an invariant that the routes are always sorted ascending by time.
        for route in &mut routes {
            route.sort_unstable_by_key(|x| x.time);
        }

        let mut solution = Self {
            problem,
            routes: routes.iter().map(|r| Vec::with_capacity(r.len())).collect(),
            npt_cache: Cell::default(),
        };

        for (v, route) in routes.iter().enumerate() {
            for (i, visit) in route.iter().enumerate() {
                solution.can_insert(v, i, *visit)?;
            }
        }

        Ok(solution)
    }

    /// Invalidate caches.
    fn invalidate_caches(&self) {
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
            inventory[visit.product] += visit.quantity;
        }

        inventory
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
            + self.problem.travel_time(previous.node, visit.node, vessel);

        // The latest time at which we can leave the visit we're attempting to insert while still making it to the next one in time
        // (if there is no next, we still require this visit to be done by the end of the planning period)
        let latest_depart = next.map_or(self.problem.timesteps(), |next| {
            let arrival = next.time;
            let loading_time = self.problem.min_loading_time(next.node, next.quantity);
            let travel_time = self.problem.travel_time(visit.node, next.node, vessel);
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
