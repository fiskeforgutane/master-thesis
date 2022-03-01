use std::{
    collections::{HashMap, HashSet},
    ops::{Range, RangeInclusive},
};

use rand::{self, prelude::Distribution};
use rand::{distributions::Uniform, Rng};

use crate::{
    problem::{NodeIndex, Problem, ProductIndex, Quantity, TimeIndex, VesselIndex},
    quants::Order,
};

/// Routes for each vessel, where each route is
pub struct Solution(Vec<Vec<(TimeIndex, NodeIndex)>>);

pub struct SlackInductionByStringRemoval<'p, 'o> {
    /// The problem we're trying to solve
    problem: &'p Problem,
    /// The orders we're trying to satisfy
    orders: &'o [Order],
    /// The current solution
    solution: Vec<Vec<Visit>>,
}

pub struct SortingWeights {
    /// Sort randomly
    pub random: f64,
    /// Sort by earliest start time
    pub earliest: f64,
    /// Sort by furthest distance from existing tours (within the time window)
    pub furthest: f64,
    /// Sort by closest distance from existing tours (within the time window)
    pub closest: f64,
}

pub enum GreedyBy {
    Random,
    Earliest,
    Furthest,
    Closest,
}

impl Default for SortingWeights {
    fn default() -> Self {
        Self {
            random: 4.0,
            earliest: 4.0,
            furthest: 2.0,
            closest: 1.0,
        }
    }
}

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Visit {
    /// The node we're visiting.
    node: NodeIndex,
    /// The product being delivered.
    product: ProductIndex,
    /// The time at which delivery starts.
    time: TimeIndex,
    /// The quantity delivered.
    quantity: Quantity,
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

impl<'p, 'o> SlackInductionByStringRemoval<'p, 'o> {
    /// Create a new instance of SISRs for the given problem and set of orders
    pub fn new(problem: &'p Problem, orders: &'o [Order]) -> Self {
        Self {
            problem,
            orders,
            solution: vec![Vec::new(); problem.vessels().len()],
        }
    }

    /// Warm start a SISRs from a previous solution
    pub fn warm_start(
        problem: &'p Problem,
        orders: &'o [Order],
        solution: Vec<Vec<Visit>>,
    ) -> Self {
        Self {
            problem,
            orders,
            solution,
        }
    }

    /// Determine the set of orders that are not covered by visits.
    /// Returns the indices of the orders that are not covered by the solution.
    pub fn uncovered(config: &Config, solution: &[Vec<Visit>], orders: &[Order]) -> Vec<usize> {
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

        // We will flatten and sort the set of visits
        let mut sorted = solution
            .iter()
            .flat_map(|xs| xs)
            .cloned()
            .collect::<Vec<_>>();
        sorted.sort_by_key(|v| (v.node, v.product, v.time));

        // The amount delivered for each order. (Same order)
        let mut delivered = vec![0.0; orders.len()];

        for (order, d) in orders.iter().zip(&mut delivered) {
            let open = (order.node(), order.product(), order.open());
            let close = (order.node(), order.product(), order.close());
            // `start` is the first element containing relevant deliveries, while `end` is the (exclusive) end.
            // Note that `start` == `end` iff there are not deliveries that are relevant for the order.
            let start = sorted.partition_point(|v| (v.node, v.product, v.time) < open);
            let end = sorted.partition_point(|v| (v.node, v.product, v.time) <= close);
            // The total amount delivered is simply the sum of deliveries to the relevant (node, product)-pair over the
            // course of the time window specified by the order.
            *d = sorted[start..end].iter().map(|v| v.quantity).sum();
        }

        delivered
            .iter()
            .zip(orders)
            .enumerate()
            .filter_map(|(i, (&x, order))| match x >= order.quantity() {
                true => Some(i),
                false => None,
            })
            .collect()
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
        solution: &[Vec<Visit>],
    ) -> Vec<(VesselIndex, usize)> {
        Vec::new()
    }

    /// The method used to select strings for removal
    pub fn select_strings(
        config: &Config,
        solution: &[Vec<Visit>],
    ) -> Vec<(VesselIndex, Range<usize>)> {
        let ls_max = (config.max_cardinality as f64).min(
            SlackInductionByStringRemoval::average_tour_cardinality(solution),
        );
        let ks_max = (4.0 * config.average_removal as f64) / (1.0 + ls_max) - 1.0;
        // The number of strings that will be removed.
        let k_s =
            Uniform::new_inclusive(1.0, ks_max + 1.0).sample(&mut rand::thread_rng()) as usize;

        let (seed_vehicle, seed_index) =
            SlackInductionByStringRemoval::select_random_visit(solution);
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
            let adjacents = SlackInductionByStringRemoval::adjacent(
                seed.node,
                &vehicles_used,
                &time_periods[time_periods.len() - 1],
                solution,
            );

            if adjacents.is_empty() {
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

    /// Attempt to repair a solution
    pub fn repair(config: &Config, solution: &[Vec<Visit>], orders: &[Order]) {}

    /// Run SISRs with the given configuration
    pub fn run(&mut self, config: &Config) {
        for _ in 0..config.iterations {
            // Select strings for removal, and create a new solution without them
            let strings = SlackInductionByStringRemoval::select_strings(config, &self.solution);
            let mut solution = self.solution.clone();
            // Note: since there is at most one string drawn from every vessel's tour, this is working as intended.
            // There can not occur any case where one range is "displaced" due to another range being removed from the same Vec.
            for (vessel, range) in strings {
                drop(solution[vessel].drain(range));
            }

            // Determine the orders that are uncovered
            let uncovered =
                SlackInductionByStringRemoval::uncovered(config, &solution, self.orders);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::problem::{Problem, Vessel};

    use super::Config;

    pub fn test_instance() {}

    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
