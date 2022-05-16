use float_ord::FloatOrd;

use rand::prelude::*;

use crate::{
    ga::{Fitness, Mutation},
    problem::{Problem, Vessel},
    solution::routing::RoutingSolution,
};

/// Consider a random triplet of visits (a, b, c), and replace the middle one.
/// The choice of which node `d` to replace `b` with is chosen based on the distance `a -> d -> c`,
/// choosing greedily with a blink rate. In addition, we only look at those `d` that have
/// the same kind as `b`.
pub struct ReplaceNode {
    rng: StdRng,
    pub blink_rate: f64,
}

impl ReplaceNode {
    pub fn new(blink_rate: f64) -> Self {
        ReplaceNode {
            rng: StdRng::from_entropy(),
            blink_rate,
        }
    }
}

impl Mutation for ReplaceNode {
    fn apply(&mut self, problem: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        let mut solution = solution.mutate();
        // Choose a random vessel's plan
        let v = problem.indices::<Vessel>().choose(&mut self.rng).unwrap();
        let mut plan = solution[v].mutate();
        // Choose a triplet from a the vessel's solution
        let (prev, mid, next) = match (1..plan.len() - 1).choose(&mut self.rng) {
            Some(mid) => (plan[mid - 1], mid, plan[mid + 1]),
            None => return,
        };

        // The new node must have the same kind as the one we will replace.
        let kind = problem.nodes()[plan[mid].node].r#type();
        let relevant = problem.nodes().iter().filter(|n| n.r#type() == kind);

        // The new node is the one among the relevant ones with the least distance (unless evaluation blinked)
        plan[mid].node = relevant
            .min_by_key(|node| {
                let blink = self.rng.gen_bool(self.blink_rate);
                let one = problem.distance(prev.node, node.index());
                let two = problem.distance(node.index(), next.node);
                (blink, FloatOrd(one + two))
            })
            .expect("there should be at least one of each node type")
            .index();
    }
}
