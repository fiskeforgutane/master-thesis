use float_ord::FloatOrd;

use rand::{
    prelude::{IteratorRandom, StdRng},
    SeedableRng,
};

use crate::{
    ga::{initialization::GreedyWithBlinks, Mutation},
    problem::Vessel,
    solution::{routing::Plan, Visit},
};

use super::Dropout;

/// A ruin and recreate method that removes visits within a random time period, and repairs the solution by greedily inserting visits using greedy insertion with blinks
pub struct Period {
    rng: StdRng,
    blink_rate: f64,
    removal_rate: f64,
    epsilon: (f64, f64),
    max_size: usize,
    c: usize,
}

impl Period {
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
}

impl Mutation for Period {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
    ) {
        let t = problem.timesteps();
        let half = self.max_size / 2;
        let middle = (0..t).choose(&mut self.rng).unwrap();
        let period = middle.max(half) - half..(middle + half).min(t);

        // Ruin the solution
        {
            let mut solution = solution.mutate();
            for mut plan in solution.iter_mut().map(Plan::mutate) {
                let origin = plan.origin();
                let eligible = |v: Visit| v != origin && period.contains(&v.time);
                plan.dropout(eligible, self.removal_rate);
            }
        }

        let mut best = (
            solution.warp(),
            FloatOrd(solution.violation()),
            FloatOrd(solution.cost() - solution.revenue()),
        );

        let mut indices = problem
            .indices::<Vessel>()
            .map(|v| {
                match solution[v].binary_search_by_key(&period.start, |visit| visit.time) {
                    Ok(x) => x,
                    // since the index where an element can be inserted is returned if the key is not found, this will be the first visit time > period.
                    // therefore the one before is x-1
                    Err(x) => 0.max(x - 1),
                }
            })
            .collect::<Vec<_>>();
        let mut candidates = problem
            .indices::<Vessel>()
            .map(|v| {
                solution
                    .candidates(indices[v], v, self.c)
                    .filter(|(_, v)| v.time < period.end)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut v = 0;
        loop {
            let greedy = GreedyWithBlinks::new(self.blink_rate);
            candidates[v] = solution
                .candidates(indices[v], v, self.c)
                .filter(|(_, v)| v.time < period.end)
                .collect();
            // choose the best among the candidates
            match greedy.insert_best(
                solution,
                self.epsilon,
                &candidates.iter().flatten().cloned().collect(),
                best,
            ) {
                Some((idx, obj)) => {
                    best = obj;
                    indices[idx.0] += 1;
                    v = idx.0
                }
                None => return,
            }
        }
    }
}
