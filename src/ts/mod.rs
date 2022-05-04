use std::collections::VecDeque;

use float_ord::FloatOrd;

use crate::{
    ga::{Fitness, Mutation},
    solution::routing::RoutingSolution,
    time,
};

pub struct TabuSearch<M> {
    // The mutation used to generate neighboring solutions
    pub mutation: M,
    // The list of tabu'ed elements
    pub tabu: VecDeque<RoutingSolution>,
    // The maximum size of the tabu list
    pub tabu_max_size: usize,
}

pub struct Evaluation {
    // The time warp of the solution
    warp: usize,
    // The amount of violation in the solution
    violation: f64,
    // The loss (i.e. - profit) of the solution.
    loss: f64,
}

impl<'a> From<&'a RoutingSolution> for Evaluation {
    fn from(solution: &'a RoutingSolution) -> Self {
        Evaluation {
            warp: solution.warp(),
            violation: solution.violation(),
            loss: solution.cost() - solution.revenue(),
        }
    }
}

pub fn tabu_search<M: Mutation, F: Fitness>(
    mut solution: RoutingSolution,
    mut mutation: M,
    fitness: F,
    tabu_max_size: usize,
    neighborhood_size: usize,
    epochs: usize,
) {
    // Tabu list
    // TODO: It would probably be useful to implement some kind of edit distance to use as a distance metric,
    // rather than using direct equality
    let mut tabu = VecDeque::from(vec![solution.to_vec(); tabu_max_size]);
    // The best solution found so far
    let mut best = solution.clone();

    // The new solutions, created as offsprings of the
    let mut neighbors = vec![solution.clone(); neighborhood_size];

    for epoch in 0..epochs {
        let cur = Evaluation::from(&solution);
        let eval = Evaluation::from(&best);
        println!(
            "{epoch:08}: warp = {: >5}, violation = {: >10}, loss = {: >10}; (BEST warp = {: >5}, violation = {: >10}, loss = {: >10})",
            cur.warp, cur.violation, cur.loss,
            eval.warp, eval.violation, eval.loss
        );
        for neighbor in &mut neighbors {
            neighbor.clone_from(&solution);

            time!("mutation", {
                mutation.apply(solution.problem(), neighbor);
            });
        }

        if let Some(candidate) = neighbors
            .iter()
            .filter(|x| !tabu.contains(&x.to_vec()))
            .min_by_key(|x| FloatOrd(fitness.of(solution.problem(), x)))
        {
            // Always replace `best` if the candidate is better
            if fitness.of(solution.problem(), candidate) < fitness.of(solution.problem(), &best) {
                best.clone_from(candidate);
            }

            // Replace the incumbent and insert the new candidate into the tabu list
            tabu.pop_front();
            tabu.push_back(candidate.to_vec());
            solution.clone_from(candidate);
        }
    }
}
