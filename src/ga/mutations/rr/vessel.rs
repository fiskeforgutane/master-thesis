use rand::{
    prelude::{IteratorRandom, StdRng},
    Rng, SeedableRng,
};

use crate::ga::{initialization::GreedyWithBlinks, Fitness, Mutation};

pub trait Dropout<T> {
    fn dropout<F: Fn(T) -> bool>(&mut self, eligible: F, removal_rate: f64);
}

impl<T> Dropout<T> for Vec<T>
where
    T: Copy,
{
    fn dropout<F: Fn(T) -> bool>(&mut self, eligible: F, removal_rate: f64) {
        let mut i = 0;
        while i < self.len() {
            if eligible(self[i]) && rand::thread_rng().gen_bool(removal_rate) {
                self.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// A ruin and recreate method that removes a single vessel's path, and inserts in in a greedily using greedy insertion with blinks
pub struct Vessel {
    rng: StdRng,
    pub blink_rate: f64,
    pub removal_rate: f64,
    pub epsilon: f64,
    /// The number of candidates generated for one node out from the previous, typically set to 2-4.
    pub c: usize,
}

impl Vessel {
    pub fn new(blink_rate: f64, removal_rate: f64, c: usize) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            blink_rate,
            removal_rate,
            // Note: was previously (1.0, 1.0) for violation and loss
            epsilon: 1e-7,
            c,
        }
    }
}

impl Mutation for Vessel {
    fn apply(
        &mut self,
        problem: &crate::problem::Problem,
        solution: &mut crate::solution::routing::RoutingSolution,
        fitness: &dyn Fitness,
    ) {
        // Choose a random vessel index.
        let vessel = (0..problem.vessels().len()).choose(&mut self.rng).unwrap();
        let length = solution[vessel].len();
        // Ruin it by removing a part of the vessel's plan
        {
            let mut mutator = solution.mutate();
            let mut plan = mutator[vessel].mutate();
            let origin = plan.origin();
            plan.dropout(|x| x != origin, self.removal_rate);
        }

        let mut best = fitness.of(solution.problem(), solution);
        let mut idx = 0;
        // Gredily construct a new vessel plan based on greedy insertion with blinks
        let greedy = GreedyWithBlinks::new(self.blink_rate);
        let mut candidates = solution.candidates(idx, vessel, self.c).collect::<Vec<_>>();

        // Do at most 2 * (expected removal + 1) insertions
        for _ in 0..2 * ((length as f64 * self.removal_rate) as usize + 1) {
            match greedy.insert_best(solution, self.epsilon, &candidates, best, fitness) {
                Some((_, obj)) => {
                    best = obj;
                    idx += 1;
                    candidates = solution.candidates(idx, vessel, self.c).collect::<Vec<_>>();
                }
                None => return,
            }
        }
    }
}
