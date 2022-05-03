use float_ord::FloatOrd;
use rand::prelude::{IteratorRandom, StdRng};

use crate::{
    ga::{initialization::GreedyWithBlinks, mutations::rr::Dropout, Recombination},
    problem::{Problem, Vessel},
    solution::routing::RoutingSolution,
};

#[derive(Debug)]
pub struct VesselRR {
    rng: StdRng,
    blink_rate: f64,
    removal_rate: f64,
    epsilon: (f64, f64),
    c: usize,
}

impl VesselRR {
    pub fn rebuild(&self, solution: &mut RoutingSolution, vessel: usize) {
        let mut best = (
            solution.warp(),
            FloatOrd(solution.violation()),
            FloatOrd(solution.cost() - solution.revenue()),
        );
        let mut idx = 0;
        // Gredily construct a new vessel plan based on greedy insertion with blinks
        let greedy = GreedyWithBlinks::new(self.blink_rate);
        let mut candidates = solution.candidates(idx, vessel, self.c).collect::<Vec<_>>();
        while let Some((_, obj)) = greedy.insert_best(solution, self.epsilon, &candidates, best) {
            best = obj;
            idx += 1;
            candidates = solution.candidates(idx, vessel, self.c).collect::<Vec<_>>();
        }
    }
}

impl Recombination for VesselRR {
    fn apply(
        &mut self,
        problem: &Problem,
        left: &mut RoutingSolution,
        right: &mut RoutingSolution,
    ) {
        // Choose a random vessel index.
        let vessel = (0..problem.vessels().len()).choose(&mut self.rng).unwrap();

        // Ruin left and right
        let ruin = |solution: &mut RoutingSolution| {
            let mut solution = solution.mutate();
            let mut plan = solution[vessel].mutate();
            let origin = plan.origin();
            plan.dropout(|x| x != origin, self.removal_rate);
        };

        ruin(left);
        ruin(right);

        let mut left_mut = left.mutate();
        let mut right_mut = right.mutate();

        for v in problem.indices::<Vessel>() {
            if v < vessel {
                std::mem::swap(&mut left_mut[v], &mut right_mut[v])
            }
        }

        drop(left_mut);
        drop(right_mut);

        self.rebuild(left, vessel);
        self.rebuild(right, vessel);
    }
}
