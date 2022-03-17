use std::{
    cmp::Ordering,
    collections::HashSet,
    ops::{Range, RangeInclusive},
};

use itertools::Itertools;
use log::{debug, info, trace, warn};
use rand::{
    self,
    prelude::{Distribution, SliceRandom},
};
use rand::{distributions::Uniform, Rng};

use crate::{
    problem::{
        Compartment, Cost, FixedInventory, Inventory, NodeIndex, Problem, ProductIndex, Quantity,
        TimeIndex, VesselIndex,
    },
    quants::Order,
    solution::{Solution, Visit, NPTV},
};
/// Implements a variant of the SISRs R&R algorithm presented by J. Christiaens and
/// G. V. Berge adapted for use in a VRP variant with MIRP-style time windows.
pub struct SlackInductionByStringRemoval<'p, 'o, 'c> {
    /// The problem we're trying to solve
    problem: &'p Problem,
    /// The orders we're trying to satisfy
    orders: &'o [Order],
    /// The configuration of the SISRs algorithm
    config: &'c Config,
}

/// When recreating a tour, we will sort the set of un-served orders according to one of several `SortingCritera`. This struct
/// defines weights used when randomly choosing a sorting method.
#[derive(Debug, Clone, Copy)]
pub struct SortingWeights {
    /// Sort randomly
    pub random: f64,
    /// Sort by earliest start time
    pub earliest: f64,
    /// Sort by furthest distance from existing tours (within the time window)
    pub furthest: f64,
    /// Sort by closest distance from existing tours (within the time window)
    pub closest: f64,
    /// Sort by demand
    pub demand: f64,
}

impl SortingWeights {
    /// Get the weights of a given sorting criteria
    pub fn weight_of(&self, criteria: &SortingCriteria) -> f64 {
        match criteria {
            SortingCriteria::Random => self.random,
            SortingCriteria::Earliest => self.earliest,
            SortingCriteria::Furthest => self.furthest,
            SortingCriteria::Closest => self.closest,
            SortingCriteria::Demand => self.demand,
        }
    }

    /// Choose a random sorting criteria
    pub fn choose(&self) -> SortingCriteria {
        *[
            SortingCriteria::Random,
            SortingCriteria::Earliest,
            SortingCriteria::Furthest,
            SortingCriteria::Closest,
            SortingCriteria::Demand,
        ]
        .choose_weighted(&mut rand::thread_rng(), |x| self.weight_of(x))
        .unwrap()
    }
}

/// A sorting criteria used for ordering the set of uncovered orders.
#[derive(Debug, Clone, Copy)]
pub enum SortingCriteria {
    /// Sort by random (i.e. shuffle)
    Random,
    /// Sort by earliest start of time window
    Earliest,
    /// Sort such that those furthest from a production facility is routed first
    Furthest,
    /// Sort such that the orders furthest from a production facility is routed first
    Closest,
    /// Sort such that the orders with high demand are routed first
    Demand,
}

impl Default for SortingWeights {
    fn default() -> Self {
        Self {
            random: 4.0,
            demand: 4.0,
            earliest: 4.0,
            furthest: 2.0,
            closest: 1.0,
        }
    }
}

/// Configuration determining the behaviour of the SISRs algorithm.
pub struct Config {
    /// The average number of nodes removed during a string removal
    pub average_removal: usize,
    /// The maximum cardinality of removed strings
    pub max_cardinality: usize,
    /// Impacts the number of removed nodes during the split string procedure
    pub alpha: f64,
    /// The probability of a "blink" during the greedy insertion
    pub blink_rate: f64,
    /// The initial temperature
    pub t0: f64,
    /// The end temperature
    pub tk: f64,
    /// The number of iterations to run
    pub iterations: usize,
    /// The weights used for choosing the various sorting criterias
    pub weights: SortingWeights,
}

/// A possible insertion point
#[derive(Debug)]
struct Candidate {
    /// The vessel who's route we will insert into
    pub vessel: VesselIndex,
    /// The idx of the insertion into the vessel's route
    pub idx: usize,
    /// The quantity delivered
    pub quantity: Quantity,
    /// The increase in cost from adding this candidate
    pub cost: Cost,
    /// The time at which docking happens (i.e. when loading/unloading starts)
    pub time: TimeIndex,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            average_removal: 10,
            max_cardinality: 10,
            alpha: 0.01,
            blink_rate: 0.01,
            t0: 100.0,
            tk: 1.0,
            iterations: 1000,
            weights: Default::default(),
        }
    }
}

impl<'p, 'o, 'c> SlackInductionByStringRemoval<'p, 'o, 'c> {
    /// Create a new instance of SISRs for the given problem and set of orders
    pub fn new(problem: &'p Problem, orders: &'o [Order], config: &'c Config) -> Self {
        Self {
            problem,
            orders,
            config,
        }
    }

    /// Determine the set of orders that are not covered by visits.
    /// Returns the indices of the orders that are not covered by the solution.
    pub fn uncovered(solution: &Solution, orders: &[Order]) -> Vec<(usize, f64)> {
        // Order = { node, product, quantity, time window }
        // Visit = { node, product, quantity, time }
        // Assumptions:
        //   - Each (node, product)-pair of has a disjoint set of time windows in the set of orders.
        //     In other words: we do not have multiple orders with overlapping time windows that relate
        //     to the same (node, product)-pair
        //   - Each vessel solution is sorted in ascending order by (node, product, time, quantity).
        //     This allows us to find all deliveries to each order by a binary search on (node, product, time window low, MAX_NEG), (node, product, time window high, MAX_POS)
        //
        // The above assumption(s) significantly reduce the complexity of assigning visits to orders.
        // Each visit at a (node, product)-pair can only be assigned to the unique order with (node, product) that
        // has a time-window containint visit.time (if one such order exists).

        // A flattened list of the visits sorted ascending by (node, product, time, vessel)
        let visits = solution.nptv();

        // The indices of the orders which each visit can possibly be assigned to
        let mut allowed_assignments = (0..visits.len())
            .map(|i| (i, Vec::new()))
            .collect::<Vec<_>>();

        for (o, order) in orders.iter().enumerate() {
            let open = (order.node(), order.product(), order.open());
            let close = (order.node(), order.product(), order.close());
            // `start` is the first element containing relevant deliveries, while `end` is the (exclusive) end.
            // Note that `start` == `end` iff there are not deliveries that are relevant for the order.
            let start = visits.partition_point(|(_, v)| (v.node, v.product, v.time) < open);
            let end = visits.partition_point(|(_, v)| (v.node, v.product, v.time) <= close);

            // We then add the visits to the order's (node, product) during the time period as possible assignments
            for (_, x) in allowed_assignments[start..end].iter_mut() {
                x.push(o);
            }
        }

        // We will sort by ascending number of alternatives, but will but the ones that can't be assigned to the very end.
        allowed_assignments.sort_by_key(|(_, alternatives)| match alternatives.len() {
            0 => usize::MAX,
            n => n,
        });
        // We will then filter out those that cannot be assigned, as they're irrelevant
        let start =
            allowed_assignments.partition_point(|(_, alternatives)| alternatives.len() != 0);
        drop(allowed_assignments.drain(start..));

        // The amount remaining for the delivery to be completed
        let mut remaining = orders.iter().map(|o| o.quantity()).collect::<Vec<_>>();
        // The indices of the orders that are not yet covered
        let mut uncovered = (0..orders.len()).collect::<HashSet<_>>();

        // Decide on an assignment of the visits
        let assigned = Self::assign(
            &mut remaining,
            &mut uncovered,
            &allowed_assignments,
            &visits,
            0,
        );

        trace!("Assignment successful = {}", assigned);

        uncovered
            .into_iter()
            .map(|o| (o, remaining[o]))
            .collect::<Vec<_>>()
    }

    fn assign(
        remaining: &mut [f64],
        uncovered: &mut HashSet<usize>,
        allowed_assignments: &[(usize, Vec<usize>)],
        visits: &NPTV<'_>,
        idx: usize,
    ) -> bool {
        // If all are covered, then we're done
        if uncovered.is_empty() {
            return true;
        }

        // If we're at the end, and it is not empty, then it is impossible to assign
        if idx == allowed_assignments.len() {
            return false;
        }

        let (visit, alternatives) = &allowed_assignments[idx];

        for &order in alternatives {
            let (_, visit) = &visits[*visit];
            let old = remaining[order];
            let new = old - visit.quantity;
            let remove = old >= 1e-5 && new <= 1e-5;

            // Apply the change
            remaining[order] = new;
            if remove {
                uncovered.remove(&order);
            }

            if Self::assign(remaining, uncovered, allowed_assignments, visits, idx + 1) {
                return true;
            }

            // Undo the change
            remaining[order] = old;
            if remove {
                uncovered.insert(order);
            }
        }

        return false;
    }

    pub fn average_tour_cardinality(solution: &[Vec<Visit>]) -> f64 {
        let total_length = solution.iter().map(|xs| xs.len()).sum::<usize>();
        let tour_count = solution.len();

        (total_length as f64) / (tour_count as f64)
    }

    pub fn select_random_visit(solution: &[Vec<Visit>]) -> (VesselIndex, usize) {
        // Choose a vessel whose solution we will draw from, and then an index from that vessel's solution
        let v = Uniform::new(0, solution.len()).sample(&mut rand::thread_rng());
        let i = Uniform::new(0, solution[v].len()).sample(&mut rand::thread_rng());

        (v, i)
    }

    /// Returns the node that is closest to `node` that is visited by an uncovered vehicle during `time_period`
    pub fn adjacent(
        node: NodeIndex,
        vehicles_used: &HashSet<VesselIndex>,
        time_period: &RangeInclusive<TimeIndex>,
        solution: &Solution,
    ) -> Vec<(VesselIndex, usize)> {
        let mut candidates = solution
            .routes()
            .iter()
            .enumerate()
            .filter(|(v, _)| !vehicles_used.contains(&v))
            .flat_map(|(v, route)| {
                route.iter().enumerate().filter_map(move |(i, visit)| {
                    match time_period.contains(&visit.time) {
                        true => Some((v, i)),
                        false => None,
                    }
                })
            })
            .collect::<Vec<_>>();

        candidates.sort_unstable_by(|&x, &y| {
            let key = |(v, i)| {
                let visit: Visit = solution[v][i];
                let distance = solution.problem.distance(node, visit.node);
                let skew_start = visit.time - time_period.start();
                let skew_end = time_period.end() - visit.time;
                let time_skew = skew_start.max(skew_end) - skew_start.min(skew_end);

                (distance, time_skew)
            };

            key(x).partial_cmp(&key(y)).unwrap_or(Ordering::Equal)
        });

        candidates
    }

    /// The method used to select strings for removal
    pub fn select_strings(
        config: &Config,
        solution: &Solution,
        problem: &Problem,
    ) -> Vec<(VesselIndex, Range<usize>)> {
        let ls_max = (config.max_cardinality as f64).min(
            SlackInductionByStringRemoval::average_tour_cardinality(solution.routes()),
        );
        let ks_max = (4.0 * config.average_removal as f64) / (1.0 + ls_max) - 1.0;
        // The number of strings that will be removed.
        let k_s =
            Uniform::new_inclusive(1.0, ks_max + 1.0).sample(&mut rand::thread_rng()) as usize;

        let (seed_vehicle, seed_index) =
            SlackInductionByStringRemoval::select_random_visit(solution.routes());
        let seed = solution[seed_vehicle][seed_index];

        // The strings we will remove, indexed as (vehicle, index range)
        let mut strings = Vec::with_capacity(k_s);
        // A list of the vehicle's who's tour we have removed.
        let mut vehicles_used = HashSet::with_capacity(k_s);
        // A list of the time periods covered by the strings that will be removed.
        let mut time_periods = {
            let mut v = Vec::with_capacity(k_s);
            v.push(seed.time..=seed.time);
            v
        };

        // The SISRs paper by Christiaens et al. only considers adjacency based on distance, which makes sense when there is no time aspect.
        // However, we need to considers adjacency in both space and time. It makes sense to find a nearby node that is visited in the same
        // time period as the time period of the strings that have been selected for removal so far.
        while strings.len() < k_s {
            let mut adjacents = Vec::new();

            let all = 0..=problem.timesteps() - 1;
            let it = time_periods.iter().chain(std::iter::once(&all));
            for period in it {
                adjacents = Self::adjacent(seed.node, &vehicles_used, &period, solution);
                trace!("Period {:?} has adjacents {:?}", period, &adjacents);
                if !adjacents.is_empty() {
                    break;
                }
            }

            if adjacents.is_empty() {
                warn!("No more adjacents found");
                break;
            }

            for (v, idx) in adjacents {
                // The maximum cardinality we allow for this string.
                let t = solution[v].len();
                let l_max = t.min(ls_max as usize);
                // Draw a random cardinality uniformly form [1, max]
                let l = Uniform::new(1, l_max + 1).sample(&mut rand::thread_rng());
                // We will now select a continuous string containing `idx`
                // Base case: draw the continuous range [idx..idx + l].
                // Using an offset: draw from the continuous range [idx - offset..idx + l - offset].
                // For an offset to be valid, we want it to have the correct length.
                // Let max_offset be the largest offset < l such that idx - offset >= 0,
                // i.e. offset <= l + 1 && offset <= idx.
                // and let min_offset be the smallest offset such that idx + l - offset <= t
                // i.e. offset >= idx + l - t and offset >= 0
                let ub = (l + 1).min(idx);
                let lb = (idx + l - t).max(0);
                // The range of allowed offsets that also gives a slice of size `l`
                let range = lb..ub;

                trace!(
                    "Cardinality = {}, idx = {}, allowed offsets = {:?}",
                    l,
                    idx,
                    range
                );

                let chosen = match range.is_empty() {
                    true => 0..t,
                    false => {
                        let offset = rand::thread_rng().gen_range(range);
                        idx - offset..idx + l - offset
                    }
                };

                // This should hold, unless there's a bug in the above calculations.
                assert!(
                    ((lb..ub).is_empty() && chosen.len() == t)
                        | (!(lb..ub).is_empty() && chosen.len() == l)
                );

                vehicles_used.insert(v);
                time_periods
                    .push(solution[v][chosen.start].time..=solution[v][chosen.end - 1].time);
                strings.push((v, chosen));
            }
        }

        strings
    }

    /// Returns the inventory after each visit
    pub fn inventories(tour: &[Visit], initial: Inventory) -> Vec<FixedInventory> {
        tour.iter()
            .scan(initial, |state, visit| {
                state[visit.product] += visit.quantity;
                Some(state.clone().fixed())
            })
            .collect::<Vec<_>>()
    }

    /// Determine the maximum amount that can be delivered of a given product type between each visit,
    pub fn max_delivery(inventories: &[FixedInventory], product: ProductIndex) -> Vec<Quantity> {
        let mut maximums = inventories
            .iter()
            .rev()
            .scan(f64::INFINITY, |state, x| {
                *state = state.min(x[product]);
                Some(*state)
            })
            .collect::<Vec<_>>();

        maximums.reverse();
        maximums
    }

    /// Determine the maximum amount that can be picked up of a given product type between each visit,
    /// using a vessel with compartments `compartments`
    pub fn max_pickup(
        inventories: &[FixedInventory],
        product: ProductIndex,
        compartments: &[Compartment],
    ) -> Vec<Quantity> {
        let mut maximums = inventories
            .iter()
            .rev()
            .scan(f64::INFINITY, |state, inventory| {
                *state = state.min(inventory.as_inv().capacity_for(product, compartments));
                Some(*state)
            })
            .collect::<Vec<_>>();
        maximums.reverse();
        maximums
    }

    /// Construct a list of candidates for insertion to serve the given `order` using `vessel`
    fn candidates(
        problem: &Problem,
        order: &Order,
        vessel: VesselIndex,
        solution: &Solution,
        amount: Quantity,
    ) -> Vec<Candidate> {
        let tour = &solution[vessel];
        let boat = &solution.problem.vessels()[vessel];

        // We wish to step through all pair of visits, and check whether we can insert a delivery to `order.node`
        // into that time slot, and then whether or not we can fill up to `target_amount`.
        let origin = solution.origin_visit(vessel);
        let visits = || {
            std::iter::once(&origin)
                .chain(tour)
                .map(Option::from)
                .chain(std::iter::once(None))
        };

        let mut candidates = Vec::new();

        // This handles everything except after the last one.
        for (idx, (from, to)) in visits().zip(visits().skip(1)).enumerate() {
            // The earliest time we can arrive at the order node after having completed the visit at `from`
            let from = from.unwrap();
            let earliest = from.time
                + problem.min_loading_time(from.node, from.quantity)
                + problem.travel_time(from.node, order.node(), boat);
            // The latest time at which we can leave `order.node` and still make it to `to` in time.
            let latest = to.map_or(problem.timesteps(), |to| {
                to.time - problem.travel_time(order.node(), to.node, boat)
            }) - problem.min_loading_time(order.node(), amount);

            let intersection = earliest.max(order.open())..latest.min(order.close());

            // Note: the inventory of the vessel should be constant for the duration between the two visits
            let inventory = solution.vessel_inventory_at(vessel, intersection.start);
            let max_quantity = inventory.capacity_for(order.product(), boat.compartments());
            let quantity = max_quantity.min(amount);
            let port_cost = problem.nodes()[order.node()].port_fee();
            // The additional distance that must be travelled
            let distance = problem.distance(from.node, order.node())
                + to.map_or(0.0, |to| {
                    problem.distance(order.node(), to.node) - problem.distance(from.node, to.node)
                });
            // The unit cost per distance travelled
            let unit_cost = match inventory.is_empty() {
                true => boat.empty_travel_unit_cost(),
                false => boat.travel_unit_cost(),
            };
            // The additional cost from making this insertion
            let cost = port_cost + distance * unit_cost;

            for time in intersection {
                candidates.push(Candidate {
                    vessel,
                    idx,
                    quantity,
                    cost,
                    time,
                });
            }
        }

        trace!("Found {} candidates", candidates.len());

        candidates
    }

    /// Attempt to repair a solution
    pub fn repair(&self, solution: &mut Solution, orders: &[Order], uncovered: &[(usize, f64)]) {
        // For each uncovered order, we want to find where we might insert it into the current solution
        for &(o, amount) in uncovered.iter() {
            let order = &orders[o];
            trace!("Trying to cover order {:?} (remaining = {}", order, amount);

            // Construct a set of possible candidates that are valid
            let candidates = solution
                .routes()
                .iter()
                .enumerate()
                .flat_map(|(v, _)| Self::candidates(self.problem, order, v, &solution, amount));

            // Choose the candidate that maximizing quantity while minimizing cost
            let chosen = candidates.max_by(|x, y| {
                (x.quantity, -x.cost)
                    .partial_cmp(&(y.quantity, y.cost))
                    .unwrap_or(Ordering::Equal)
            });

            let candidate = match chosen {
                Some(x) => {
                    trace!("Covering {:?} with {:?}", order, &x);
                    x
                }
                None => {
                    info!("No candidates for order #{}: {:?}", o, orders[o]);
                    continue;
                }
            };

            solution
                .insert(
                    candidate.vessel,
                    candidate.idx,
                    Visit {
                        node: order.node(),
                        product: order.product(),
                        time: candidate.time,
                        quantity: candidate.quantity,
                    },
                )
                .unwrap_or_else(|err| {
                    warn!(
                        "Insertion of candidate {:?} failed with {:?}",
                        candidate, err
                    )
                });
        }
    }

    fn ruin(&self, solution: &mut Solution) {
        // Select strings for removal, and create a new solution without them
        trace!("Ruining solution.");
        let strings =
            SlackInductionByStringRemoval::select_strings(&self.config, solution, self.problem);

        trace!("Dropping strings {:?}", &strings);
        // Note: since there is at most one string drawn from every vessel's tour, this is working as intended.
        // There can not occur any case where one range is "displaced" due to another range being removed from the same Vec.
        for (vessel, range) in strings {
            drop(solution.drain(vessel, range));
        }
    }

    fn recreate(&self, solution: &mut Solution) {
        trace!("Recreating solution.");
        // Determine the orders that are uncovered, and the amount by which they're uncovered
        let mut uncovered = SlackInductionByStringRemoval::uncovered(solution, self.orders);

        trace!("Uncovered orders: {:?}", uncovered);

        // Shuffle the order of the uncovered nodes according
        match self.config.weights.choose() {
            SortingCriteria::Random => uncovered.shuffle(&mut rand::thread_rng()),
            SortingCriteria::Earliest => uncovered.sort_by_key(|(o, _)| self.orders[*o].open()),
            SortingCriteria::Furthest => (),
            SortingCriteria::Closest => (),
            SortingCriteria::Demand => uncovered.sort_by(|(_, a), (_, b)| {
                b.partial_cmp(a).expect("should be non-nan by construction")
            }),
        }

        // Repair the new solution
        self.repair(solution, self.orders, &uncovered);
    }

    /// Run SISRs starting from the given solution
    pub fn run(&mut self, initial: Solution<'p>) -> Solution<'p> {
        info!("Starting SISRs from initial solution {:?}", &initial);
        // The best solution so far.
        let mut best = initial.clone();
        // The current solution
        let mut solution = initial;

        let uniform = Uniform::<f64>::new(0.0, 1.0);
        let c = (self.config.tk / self.config.t0).powf(1.0 / self.config.iterations as f64);
        let mut t = self.config.t0;
        for iteration in 0..self.config.iterations {
            trace!("SISRs iteration {}", iteration);

            // Create a new solution by R&R-ing the current one
            let mut new = solution.clone();
            self.ruin(&mut new);
            self.recreate(&mut new);

            // We'll compare on shortage first, and then on cost
            let eval_new = new.evaluation();
            let eval_old = solution.evaluation();
            let noise = t * uniform.sample(&mut rand::thread_rng()).ln();

            if eval_new < best.evaluation() {
                info!("Found new best solution with {:?}", eval_new);
                best = new.clone();
            }

            if eval_new.shortage < eval_old.shortage + noise {
                solution = new;
            }

            t *= c;
        }

        info!("Best solution found by SISRs was {:?}", &best);
        solution
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::{Compartment, FixedInventory, Inventory, Node, Problem, Vessel};

    use super::Config;

    static TIMESTEPS: usize = 360;
    static PRODUCTS: usize = 2;
    static N_VESSELS: usize = 2;
    static N_FARMS: usize = 3;
    static N_FACTORIES: usize = 2;

    pub fn test_instance() -> Problem {
        let vessel = Vessel::new(
            vec![Compartment(100.0), Compartment(100.0), Compartment(100.0)],
            1.0,
            1.0,
            0.7,
            0.5,
            0.3,
            0,
            Inventory::new(&vec![0.0; PRODUCTS]).unwrap().into(),
            0,
            "Class 1".to_owned(),
        );
        todo!()
    }

    #[test]
    fn uncovered_is_empty_with_nosplit_delivery() {
        // TODO
    }

    #[test]
    fn uncovered_is_empty_with_split_delivery() {
        // TODO
    }

    #[test]
    fn uncovered_identifies_unfulfilled_delivery() {
        // TODO
    }

    #[test]
    fn uncovered_works_with_delivery_not_belonging_to_any_order() {
        // TODO
    }

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
