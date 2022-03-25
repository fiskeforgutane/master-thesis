use super::{AnySolution, Delivery, Evaluation, InsertionError, InventoryViolation};
use crate::problem::{Inventory, NodeIndex, Problem, ProductIndex, TimeIndex, VesselIndex};
use itertools::Itertools;
use std::{
    cell::Cell,
    ops::{Deref, Index, Range, RangeBounds},
};

/// A solution to `Problem`, i.e. a specification of the routes taken by each vehicle.
/// Note that the solution is in no way guaranteed to be optimal or even capacity-feasible.
/// However, it does guarantee that the solution is possible to actually perform. I.e. that each vessel
/// is able to perform its scheduled visits while conformning to restrictions on travel and (un)loading times.
pub struct FullSolution<'p> {
    /// The problem this solution belongs to
    pub problem: &'p Problem,
    /// The routes taken by each vehicle
    routes: Vec<Vec<Delivery>>,
    /// A cache of the routes stored sorted by (node, product, time, vessel). This allows us to lookup visits to a (node, product)-pair within a time window in log(n)
    npt_cache: Cell<Vec<(VesselIndex, Delivery)>>,
    /// A cache of the evaluation of this solution
    evaluation: Cell<Option<Evaluation>>,
}

impl<'p> AnySolution for FullSolution<'p> {
    type Inner = Vec<Delivery>;

    fn problem(&self) -> &Problem {
        self.problem
    }

    fn routes(&self) -> &[Self::Inner] {
        &self.routes
    }
}

impl<'p> std::fmt::Debug for FullSolution<'p> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solution")
            .field("problem", &self.problem)
            .field("routes", &self.routes)
            .finish()
    }
}

impl<'p> Index<VesselIndex> for FullSolution<'p> {
    type Output = [Delivery];

    fn index(&self, index: VesselIndex) -> &Self::Output {
        &self.routes[index]
    }
}

impl<'p> Clone for FullSolution<'p> {
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

impl<'p> FullSolution<'p> {
    pub fn new(
        problem: &'p Problem,
        mut routes: Vec<Vec<Delivery>>,
    ) -> Result<Self, InsertionError> {
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

    // Dissolve this solution into the problem and routes it consists of.
    pub fn dissolve(self) -> (&'p Problem, Vec<Vec<Delivery>>) {
        (self.problem, self.routes)
    }

    /// Invalidate caches.
    fn invalidate_caches(&self) {
        self.evaluation.set(None);
        // Note: if we wish to avoid unnecessary allocations, we can instead
        // take the npt_cache, clear the vector, and then set it back.
        self.npt_cache.set(Vec::new());
    }

    /// Update the flat list of visists sorted by (Node, Product, Time) that allows efficient lookup of visits to a node of a certain product within a time window
    fn updated_nptv_cache(&self) -> Vec<(usize, Delivery)> {
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
    pub fn routes(&self) -> &[Vec<Delivery>] {
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
    pub fn origin_visit(&self, vessel: VesselIndex) -> Delivery {
        let vessel = &self.problem.vessels()[vessel];
        Delivery {
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

    /// Calculates the vessel shortage and excess of this solution
    fn vessel_violations(&self) -> InventoryViolation {
        let mut shortage = 0.0;
        let mut excess = 0.0;

        for (v, route) in self.routes.iter().enumerate() {
            let vessel = &self.problem.vessels()[v];
            let mut inventory = vessel.initial_inventory().as_inv().clone();

            // Artificial visit for "end of planning horizon"
            // Note: a streaming iterator would have been nice here
            let end = (0..self.problem.products())
                .map(|p| Delivery {
                    node: 0,
                    product: p,
                    time: self.problem.timesteps() - 1,
                    quantity: 0.0,
                })
                .collect::<Vec<_>>();

            // When the inventory for product type `p` was last updated
            let mut last_updated = vec![vessel.available_from(); self.problem.products()];

            // We are always "present" at `visit` (i.e. the vessel has the inventory it has leaving `visit`),
            // and will calculate how large the violation is in the time before we arrive at `visit`
            for visit in route.iter().chain(&end) {
                let time_spanned = (visit.time - last_updated[visit.product]) as f64;
                last_updated[visit.product] = visit.time;
                // This is the amount by which the inventory limit is breached over the time span.
                // The slack (capacity_for) is negative if the inventory exceeds the amount that can be stored in `vessel.compartments()`
                // If the slack is negative, we use that (since -slack is positive in that case)
                // and if it is positive we will use 0 (since -slack will be negative in that case).
                let capacity = inventory.capacity_for(visit.product, vessel.compartments());
                let e = (-capacity).max(0.0);
                // The shortage is the abs of the inventory if it is negative.
                let s = (-inventory[visit.product]).max(0.0);

                excess += time_spanned * e;
                shortage += time_spanned * s;

                inventory[visit.product] += visit.quantity
            }
        }

        InventoryViolation { excess, shortage }
    }

    /// Calculates the accumulated shortage and excess of this solution
    fn node_violations(&self) -> InventoryViolation {
        let mut shortage = 0.0;
        let mut excess = 0.0;

        // It should be possible to do this
        for (n, node) in self.problem.nodes().iter().enumerate() {
            for p in 0..self.problem.products() {
                let accumulative = node.inventory_without_deliveries(p);
                let capacity = node.capacity()[p];
                let mut delta = 0.0;

                for t in 0..self.problem.timesteps() {
                    // This should be either one or zero deliveries. `visit.node == n` and `visit.product == p` by construction.
                    for (_, visit) in self.deliveries(n, p, t..t + 1).iter() {
                        delta += visit.quantity;
                    }

                    // The quantity at the farm/factory
                    let inv = accumulative[t] + delta;

                    // Calculate shortage and excess at the current time.
                    shortage += f64::max(-inv, 0.0);
                    excess += f64::max(0.0, inv - capacity);
                }
            }
        }

        InventoryViolation { excess, shortage }
    }

    /// The evaluation of this solution
    pub fn evaluation(&self) -> Evaluation {
        if let Some(evaluation) = self.evaluation.get() {
            return evaluation;
        }

        let cost = self.cost();
        let nodes = self.node_violations();
        let vessels = self.vessel_violations();

        let eval = Evaluation {
            cost,
            nodes,
            vessels,
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
        visit: Delivery,
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
            0 => Delivery {
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
        visit: Delivery,
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
    ) -> std::vec::Drain<'_, Delivery> {
        self.routes[vessel].drain(range)
    }
}

pub struct NPTV<'cell> {
    /// The Cell that is supposed to hold the `inner` value when this is dropped
    cell: &'cell Cell<Vec<(VesselIndex, Delivery)>>,
    /// The inner Vec
    inner: Vec<(VesselIndex, Delivery)>,
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
    type Target = [(VesselIndex, Delivery)];

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
    type Target = [(VesselIndex, Delivery)];

    fn deref(&self) -> &Self::Target {
        &self.inner[self.range.clone()]
    }
}
