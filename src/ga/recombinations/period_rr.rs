use std::ops::Range;

use float_ord::FloatOrd;
use itertools::Itertools;
use rand::{
    prelude::{IteratorRandom, StdRng},
    SeedableRng,
};

use crate::{
    ga::{initialization::GreedyWithBlinks, mutations::rr::Dropout, Recombination},
    problem::{Problem, Vessel},
    solution::{
        routing::{Plan, RoutingSolution},
        Visit,
    },
};

#[derive(Debug)]
pub struct PeriodRR {
    rng: StdRng,
    blink_rate: f64,
    removal_rate: f64,
    epsilon: (f64, f64),
    max_size: usize,
    c: usize,
}

impl PeriodRR {
    pub fn new(blink_rate: f64, removal_rate: f64, max_size: usize, c: usize) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            blink_rate,
            removal_rate,
            epsilon: (1.0, 1.0),
            max_size,
            c,
        }
    }

    pub fn rebuild(&self, solution: &mut RoutingSolution, period: &Range<usize>) {
        let mut best = (
            solution.warp(),
            FloatOrd(solution.violation()),
            FloatOrd(solution.cost() - solution.revenue()),
        );

        // The indices of the last visit evaluated in every plan
        let mut indices = solution
            .problem()
            .indices::<Vessel>()
            .map(|v| {
                match solution[v].binary_search_by_key(&period.start, |visit| visit.time) {
                    Ok(x) => x,
                    // since the index where an element can be inserted is returned if the key is not found, this will be the first visit time > period.
                    // therefore the one before is x-1
                    Err(x) => x.max(1) - 1,
                }
            })
            .collect::<Vec<_>>();

        // gets candidates and filteres out the ones outside the range.
        let get_candidates =
            |v, solution: &crate::solution::routing::RoutingSolution, indices: &Vec<usize>| {
                solution
                    .candidates(indices[v], v, self.c)
                    .filter(|(_, v)| period.contains(&v.time))
                    .collect::<Vec<_>>()
            };

        let mut candidates = solution
            .problem()
            .indices::<Vessel>()
            .map(|v| get_candidates(v, solution, &indices))
            .collect::<Vec<_>>();

        let greedy = GreedyWithBlinks::new(self.blink_rate);
        while let Some((idx, obj)) = greedy.insert_best(
            solution,
            self.epsilon,
            &candidates.iter().flatten().cloned().collect(),
            best,
        ) {
            best = obj;
            indices[idx.0] += 1;
            // the index of the vessel changed, this is the one we must get new candidates for
            let v = idx.0;
            candidates[v] = get_candidates(v, solution, &indices);
        }
    }
}

impl Recombination for PeriodRR {
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        let t = problem.timesteps();
        let half = self.max_size / 2;
        let middle = (0..t).choose(&mut self.rng).unwrap();
        let period = middle.max(half) - half..(middle + half).min(t);

        // Ruin left and right
        let ruin = |solution: &mut RoutingSolution| {
            let mut solution = solution.mutate();
            for mut plan in solution.iter_mut().map(Plan::mutate) {
                let origin = plan.origin();
                let eligible = |v: Visit| v != origin && period.contains(&v.time);
                plan.dropout(eligible, self.removal_rate);
            }
        };
        ruin(left);
        ruin(right);

        // extract visits satisfying the given predicate
        let extract = |solution: &mut RoutingSolution| {
            let mut sol_mut = solution.mutate();
            sol_mut
                .iter_mut()
                .map(|plan| {
                    let i = plan
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find_or_last(|(_, x)| x.time > period.start)
                        .map(|(i, _)| i);

                    let mut plan_mut = plan.mutate();
                    match i {
                        Some(i) => plan_mut.drain(1..i).collect(),
                        None => Vec::new(),
                    }
                })
                .collect::<Vec<_>>()
        };

        let add = |solution: &mut RoutingSolution, to_add: Vec<Vec<Visit>>| {
            let mut sol_mut = solution.mutate();
            for vessel in problem.indices::<Vessel>() {
                let mut plan_mut = sol_mut[vessel].mutate();
                to_add[vessel]
                    .iter()
                    .for_each(|visit| plan_mut.push(visit.clone()));
            }
        };

        // swap
        let left_out = extract(left);
        let right_out = extract(right);
        left.iter().for_each(|plan| {
            plan.iter()
                .skip(1)
                .for_each(|x| assert!(x.time > period.start))
        });
        right.iter().for_each(|plan| {
            plan.iter()
                .skip(1)
                .for_each(|x| assert!(x.time > period.start))
        });

        add(left, right_out);
        add(right, left_out);

        self.rebuild(left, &period);
        self.rebuild(right, &period);
    }
}
