use std::{
    collections::{HashMap, HashSet},
    ops::{Range, RangeInclusive},
};

use rand::distributions::Uniform;
use rand::{self, prelude::Distribution};

use crate::problem::{NodeIndex, Problem, ProductIndex, Quantity, TimeIndex, VesselIndex};

/// Routes for each vessel, where each route is
pub struct Solution(Vec<Vec<(TimeIndex, NodeIndex)>>);

pub struct SlackInductionByStringRemoval<'p, 'o> {
    /// The problem we're trying to solve
    problem: &'p Problem,
    /// The orders we're trying to satisfy
    orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, Quantity)],
    /// The current solution
    solution: Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, Quantity)>>,
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
    pub fn new(
        problem: &'p Problem,
        orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, Quantity)],
    ) -> Self {
        Self {
            problem,
            orders,
            solution: vec![Vec::new(); problem.vessels().len()],
        }
    }

    /// Warm start a SISRs from a previous solution
    pub fn warm_start(
        problem: &'p Problem,
        orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, Quantity)],
        solution: Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, Quantity)>>,
    ) -> Self {
        Self {
            problem,
            orders,
            solution,
        }
    }

    /// Determine which of the orders that are *not* fulfilled
    pub fn unfulfilled(
        orders: &[(NodeIndex, TimeIndex, TimeIndex, ProductIndex, Quantity)],
        solution: &Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, Quantity)>>,
    ) {
    }

    pub fn average_tour_cardinality(&self) -> f64 {
        let total_length = self.solution.iter().map(|xs| xs.len()).sum::<usize>();
        let tour_count = self.solution.len();

        (total_length as f64) / (tour_count as f64)
    }

    pub fn select_random_visit(&self) -> (VesselIndex, usize) {
        // Choose a vessel whose solution we will draw from, and then an index from that vessel's solution
        let v = Uniform::new(0, self.solution.len()).sample(&mut rand::thread_rng());
        let i = Uniform::new(0, self.solution[v].len()).sample(&mut rand::thread_rng());

        (v, i)
    }

    /// Returns the node that is closest to `node` that is visited by an uncovered vehicle during `time_period`
    pub fn adjacent(
        &self,
        node: NodeIndex,
        vehicles_used: &HashSet<VesselIndex>,
        time_period: RangeInclusive<TimeIndex>,
    ) -> Vec<(VesselIndex, usize)> {
        Vec::new()
    }

    /// The method used to select strings for removal
    fn select_strings(&self, config: &Config) -> Vec<(VesselIndex, Range<usize>)> {
        let ls_max = (config.max_cardinality as f64).min(self.average_tour_cardinality());
        let ks_max = (4.0 * config.average_removal as f64) / (1.0 + ls_max) - 1.0;
        // The number of strings that will be removed.
        let k_s =
            Uniform::new_inclusive(1.0, ks_max + 1.0).sample(&mut rand::thread_rng()) as usize;

        let (seed_vehicle, seed_index) = self.select_random_visit();
        let (seed_time, seed_node, seed_product, seed_quantity) =
            self.solution[seed_vehicle][seed_index];

        // The strings we will remove, indexed as (vehicle, index range)
        let mut strings = Vec::with_capacity(k_s);
        // A list of the vehicle's who's tour we have removed.
        let mut vehicles_covered = HashSet::with_capacity(k_s);
        // A list of the time periods covered by the strings that will be removed.
        let mut time_periods = {
            let mut v = Vec::with_capacity(k_s);
            v.push(seed_time..=seed_time);
            v
        };

        // The SISRs paper by Christiaens et al. only considers adjacency based on distance, which makes sense when there is no time aspect.
        // However, we need to considers adjacency in both space and time. It makes sense to find a nearby node that is visited in the same
        // time period as the time period of the strings that have been selected for removal so far.
        while strings.len() < k_s {}

        strings
    }

    /// Run SISRs with the given configuration
    pub fn run(&mut self, config: &Config) {}
}
