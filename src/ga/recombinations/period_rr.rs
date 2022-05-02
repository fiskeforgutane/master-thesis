use std::ops::Range;

use float_ord::FloatOrd;
use rand::prelude::{IteratorRandom, StdRng};

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
    pub fn rebuild(&self, solution: &mut RoutingSolution, period: Range<usize>) {
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

        self.rebuild(left, period);
        self.rebuild(right, period);

        // perform the swap part
        for vessel in 0..problem.vessels().len() {
            for visit in left[vessel].iter().skip(1) {
                if v.time < period.start {}
            }
        }

        let mut best = (
            solution.warp(),
            FloatOrd(solution.violation()),
            FloatOrd(solution.cost() - solution.revenue()),
        );

        // The indices of the last visit evaluated in every plan
        let mut indices = problem
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

        let mut candidates = problem
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
