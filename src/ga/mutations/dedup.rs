use crate::{
    ga::{Mutation, Fitness},
    problem::Problem,
    solution::routing::{Plan, RoutingSolution},
};

/// How to decide on what visit to keep amount a string of visits.
pub enum DedupPolicy {
    KeepFirst,
    KeepLast,
}
/// Remove duplicate visits in a solution
pub struct Dedup(pub DedupPolicy);

impl Mutation for Dedup {
    fn apply(&mut self, _: &Problem, solution: &mut RoutingSolution, _: &dyn Fitness) {
        let mut mutator = solution.mutate();

        for mut plan in mutator.iter_mut().map(Plan::mutate) {
            match self.0 {
                DedupPolicy::KeepFirst => plan.dedup_by_key(|x| x.node),
                DedupPolicy::KeepLast => {
                    // Note: we need special handling of the origin, since that can be removed here
                    // if the second visit is also to the origin node. (since that will make the origin the last one in
                    // the reversed vec.)
                    let origin = plan.origin();
                    plan.reverse();
                    plan.dedup_by_key(|x| x.node);
                    plan.last_mut().map(|x| *x = origin);
                }
            }
        }
    }
}
