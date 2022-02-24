use crate::problem::{InventoryType, NodeIndex, Problem, ProductIndex, TimeIndex};

/// Routes for each vessel, where each route is
pub struct Solution(Vec<Vec<(TimeIndex, NodeIndex)>>);

pub struct SlackInductionByStringRemoval<'p, 'o> {
    /// The problem we're trying to solve
    problem: &'p Problem,
    /// The orders we're trying to satisfy
    orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, InventoryType)],
    /// The current solution
    solution: Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, InventoryType)>>,
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
        orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, InventoryType)],
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
        orders: &'o [(NodeIndex, TimeIndex, TimeIndex, ProductIndex, InventoryType)],
        solution: Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, InventoryType)>>,
    ) -> Self {
        Self {
            problem,
            orders,
            solution,
        }
    }

    /// Determine which of the orders that are *not* fulfilled
    pub fn unfulfilled(
        orders: &[(NodeIndex, TimeIndex, TimeIndex, ProductIndex, InventoryType)],
        solution: &Vec<Vec<(TimeIndex, NodeIndex, ProductIndex, InventoryType)>>,
    ) {
    }

    /// Run SISRs with the given configuration
    pub fn run(&mut self, config: &Config) {}
}
