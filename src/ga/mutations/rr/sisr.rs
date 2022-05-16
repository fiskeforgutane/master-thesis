use std::{
    collections::HashSet,
    ops::{Range, RangeInclusive},
};

use float_ord::FloatOrd;
use itertools::Itertools;
use log::{debug, trace, warn};
use rand::{self, prelude::Distribution};
use rand::{distributions::Uniform, Rng};
use serde::{Deserialize, Serialize};

use crate::{
    ga::{initialization::GreedyWithBlinks, Fitness, Mutation},
    problem::{NodeIndex, Problem, TimeIndex, VesselIndex},
    solution::{
        routing::{Improvement, RoutingSolution},
        Visit,
    },
};
/// Implements a variant of the SISRs R&R algorithm presented by J. Christiaens and
/// G. V. Berge adapted for use in a VRP variant with MIRP-style time windows.
pub struct SlackInductionByStringRemoval {
    /// The configuration of the SISRs algorithm
    pub config: Config,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    /// We only consider the first `n` times in each continuous range of insertion points
    pub first_n: usize,
    /// The epsilon used for insertions
    pub epsilon: Improvement,
    // The initial temperature
    // pub t0: f64,
    // The end temperature
    // pub tk: f64,
    // The number of iterations to run
    // pub iterations: usize,
}

impl SlackInductionByStringRemoval {
    /// Create a new version of the SISR mutation
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub fn average_tour_cardinality(solution: &RoutingSolution) -> f64 {
        let total_length = solution.iter().map(|xs| xs.len()).sum::<usize>();
        let tour_count = solution.len();

        (total_length as f64) / (tour_count as f64)
    }

    pub fn select_random_visit(solution: &RoutingSolution) -> Option<(VesselIndex, usize)> {
        // Choose a vessel whose solution we will draw from, and then an index from that vessel's solution
        if solution.len() == 0 {
            return None;
        }

        let v = Uniform::new(0, solution.len()).sample(&mut rand::thread_rng());

        if solution[v].len() == 0 {
            return None;
        }

        let i = Uniform::new(0, solution[v].len()).sample(&mut rand::thread_rng());

        Some((v, i))
    }

    /// Returns the node that is closest to `node` that is visited by an uncovered vehicle during `time_period`
    pub fn adjacent(
        node: NodeIndex,
        vehicles_used: &HashSet<VesselIndex>,
        time_period: &RangeInclusive<TimeIndex>,
        solution: &RoutingSolution,
    ) -> Vec<(VesselIndex, usize)> {
        debug!("vehicles used = {:?}", vehicles_used);

        let key = |&(v, i): &(usize, usize)| {
            let visit = solution[v][i];
            let distance = solution.problem().distance(node, visit.node);
            let skew_start = visit.time - time_period.start();
            let skew_end = time_period.end() - visit.time;
            let time_skew = skew_start.max(skew_end) - skew_start.min(skew_end);

            (FloatOrd(distance), time_skew)
        };

        solution
            .iter()
            .enumerate()
            .filter(|(v, _)| !vehicles_used.contains(&v))
            .flat_map(|(v, route)| {
                route[1..].iter().enumerate().filter_map(move |(i, visit)| {
                    match time_period.contains(&visit.time) {
                        true => Some((v, i)),
                        false => None,
                    }
                })
            })
            .sorted_unstable_by_key(key)
            .collect()
    }

    /// The method used to select strings for removal
    pub fn select_strings(
        config: &Config,
        solution: &RoutingSolution,
    ) -> Vec<(VesselIndex, Range<usize>)> {
        let mut rng = rand::thread_rng();
        let problem = solution.problem();

        let ls_max = (config.max_cardinality as f64).min(Self::average_tour_cardinality(solution));
        let ks_max = (4.0 * config.average_removal as f64) / (1.0 + ls_max) - 1.0;
        // The number of strings that will be removed.
        let k_s = Uniform::new_inclusive(1.0, ks_max + 1.0).sample(&mut rng) as usize;

        trace!("ls_max = {}, ks_max = {}, k_s = {}", ls_max, ks_max, k_s);

        let (seed_vehicle, seed_index) = match Self::select_random_visit(solution) {
            Some(x) => x,
            None => return Vec::new(),
        };

        let seed = solution[seed_vehicle][seed_index];

        trace!("Seed: visit {} of vehicle {}", seed_index, seed_vehicle);

        // The strings we will remove, indexed as (vehicle, index range)
        let mut strings = Vec::with_capacity(k_s);
        // A list of the vehicle's who's tour we have removed.
        let mut vehicles_used = HashSet::with_capacity(k_s);
        // A list of the time periods covered by the strings that will be removed.
        let mut time_periods = Vec::with_capacity(k_s);
        time_periods.push(seed.time..=seed.time);

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

            let (v, idx) = match adjacents.first() {
                Some(&x) => x,
                None => {
                    warn!("No more adjacents found");
                    break;
                }
            };

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
            let lb = ((idx + l - t) as isize).max(1) as usize;
            // The range of allowed offsets that also gives a slice of size `l`
            let range = lb..ub;

            trace!("L = {}, idx = {}, allowed offsets = {:?}", l, idx, range);

            let chosen = match range.is_empty() {
                true => 1..t,
                false => {
                    let offset = rand::thread_rng().gen_range(range);
                    idx - offset..idx + l - offset
                }
            };

            // This should hold, unless there's a bug in the above calculations.
            assert!(
                ((lb..ub).is_empty() && chosen.len() == t - 1)
                    | (!(lb..ub).is_empty() && chosen.len() == l)
            );

            debug!("v = {}, vehicles used = {:?}", v, vehicles_used);

            vehicles_used.insert(v);
            time_periods.push(solution[v][chosen.start].time..=solution[v][chosen.end - 1].time);
            strings.push((v, chosen));
        }

        strings
    }

    fn ruin(&self, solution: &mut RoutingSolution) {
        // Select strings for removal, and create a new solution without them
        debug!("Ruining solution.");
        let strings = SlackInductionByStringRemoval::select_strings(&self.config, solution);

        debug!("Dropping strings {:?}", &strings);
        // Note: since there is at most one string drawn from every vessel's tour, this is working as intended.
        // There can not occur any case where one range is "displaced" due to another range being removed from the same Vec.
        let mut solution = solution.mutate();
        for (vessel, range) in strings {
            let mut plan = solution[vessel].mutate();
            drop(plan.drain(range));
        }
    }

    fn candidates(&self, solution: &RoutingSolution) -> Vec<(usize, Visit)> {
        solution
            .available()
            .flat_map(|(v, it)| {
                it.flat_map(move |(node, range)| {
                    range
                        .take(self.config.first_n)
                        .map(move |time| (v, Visit { node, time }))
                })
            })
            .collect()
    }
}

impl Mutation for SlackInductionByStringRemoval {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        self.ruin(solution);

        let greedy = GreedyWithBlinks::new(self.config.blink_rate);

        let mut best = solution.evaluation();

        debug!("SISR start = {:?}", solution.to_vec());
        loop {
            let candidates = self.candidates(solution);
            debug!("\t#Candidates = {}", candidates.len());

            match greedy.insert_best(solution, self.config.epsilon, &candidates, best) {
                Some(((v, visit), obj)) => {
                    debug!("\tinserted v = {v}: {visit:?}");
                    best = obj;
                }
                None => {
                    debug!("\tno viable insertion");
                    break;
                }
            }
        }
        debug!("SISR end");
    }
}
